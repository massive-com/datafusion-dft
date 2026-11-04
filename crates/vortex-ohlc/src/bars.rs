//! `vortex_ohlc.bars`, the layout at the root of a table of bars, with the source's dtype:
//! four decimal prices where the stored table has the [`OHLC`] struct. Its children are the table
//! as stored and the source's zones: each price's min and max per zone of its rows, as the source
//! has them, which the stored table's zone maps (of the parts) can't give. Its reader rewrites
//! every expression over the source's columns into one over the stored table (a price into
//! [`ohlc_price`] over the struct, the whole row into a `pack`), and prunes a filter on prices on
//! the source's zones itself.

use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, StreamExt};
use vortex::array::aggregate_fn::AggregateFnRef;
use vortex::array::arrays::decimal::converted_buffer;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::{DecimalArray, StructArray};
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::expr::stats::Stat;
use vortex::array::stats::bind::{bind_stats, StatBinder};
use vortex::array::validity::Validity;
use vortex::array::{
    ArrayRef, ExecutionCtx, IntoArray, MaskFuture, RawMetadata, VortexSessionExecute,
};
use vortex::buffer::BufferMut;
use vortex::dtype::{
    DType, DecimalDType, FieldMask, FieldName, FieldPath, Nullability, StructFields,
};
use vortex::error::{vortex_bail, vortex_ensure, vortex_err, VortexResult};
use vortex::expr::{get_item, pack, root, BoundExpression, ExactBoundExpr, Expression};
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::segments::{SegmentSinkRef, SegmentSource};
use vortex::layout::sequence::{
    SendableSequentialStream, SequencePointer, SequentialArrayStreamExt, SequentialStreamAdapter,
    SequentialStreamExt,
};
use vortex::layout::{
    layout_children, ArrayFuture, Layout, LayoutChildType, LayoutDeserializeArgs, LayoutId,
    LayoutParts, LayoutReader, LayoutReaderContext, LayoutReaderRef, LayoutRef, LayoutStrategy,
    LayoutWriterContext, RowSplits, SplitRange, VTable,
};
use vortex::mask::Mask;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

use crate::dollar_volume::{dollar_volume, residual, shift, Residual};
use crate::ohlc::{ohlc_to_parts, Part, Price};
use crate::price_fn::{ohlc_price, OhlcPriceFn};

/// The stored table's struct column holding the four prices as their [`Part`]s, where open is in
/// the source.
pub const OHLC: &str = "ohlc";

/// The most digits a part may have: an `i128`'s.
const MAX_PART_PRECISION: u8 = 38;

/// The source's dollar volume, which the stored table has as [`DOLLAR_VOLUME_RESIDUAL`] when it
/// also has a [`VOLUME`] (see [`crate::dollar_volume`]).
pub const DOLLAR_VOLUME: &str = "dollar_volume";

/// The source's volume.
pub const VOLUME: &str = "volume";

/// The stored table's column for the source's dollar volume, less its volume times its open. A
/// name no source column has, so that its zone map is never taken for the dollar volume's.
pub const DOLLAR_VOLUME_RESIDUAL: &str = "dollar_volume_residual";

/// Register the layout, [`OhlcPriceFn`] and [`DollarVolumeFn`](crate::dollar_volume::DollarVolumeFn),
/// so files that use them can be read back.
pub fn register(session: &VortexSession) {
    use vortex::layout::session::LayoutSessionExt;
    use vortex::layout::LayoutEncoding;
    use vortex::scalar_fn::session::ScalarFnSessionExt;
    session
        .layouts()
        .register(&BarsLayout as &dyn LayoutEncoding);
    session.scalar_fns().register(OhlcPriceFn);
    session
        .scalar_fns()
        .register(crate::dollar_volume::DollarVolumeFn);
}

#[derive(Clone, Debug)]
pub struct BarsLayout;

/// The parts' decimal type and whether the dollar volume is stored as a residual, which with the
/// layout's dtype give the stored table's, and the rows per zone of the source's zones.
#[derive(Clone, Debug)]
pub struct BarsData {
    parts: DecimalDType,
    zone_len: u64,
    residual: bool,
}

impl VTable for BarsLayout {
    type LayoutData = BarsData;
    type Metadata = RawMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex_ohlc.bars");
        *ID
    }

    /// The parts' precision and scale, the zone length as a little-endian `u64`, then 1 if the
    /// dollar volume is stored as a residual and 0 if not.
    fn metadata(layout: &Layout<Self>) -> RawMetadata {
        let mut metadata = vec![layout.parts.precision(), layout.parts.scale() as u8];
        metadata.extend(layout.zone_len.to_le_bytes());
        metadata.push(u8::from(layout.residual));
        RawMetadata(metadata)
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &Vec<u8>,
    ) -> VortexResult<BarsData> {
        let [precision, scale, zone_len @ .., residual] = metadata.as_slice() else {
            vortex_bail!("a bars layout's metadata is too short");
        };
        let zone_len = <[u8; 8]>::try_from(zone_len)
            .map(u64::from_le_bytes)
            .map_err(|_| vortex_err!("a bars layout's metadata is 11 bytes"))?;
        vortex_ensure!(zone_len > 0, "a bars layout's zones have rows");
        let parts = DecimalDType::new(*precision, *scale as i8);
        let residual = match residual {
            0 => false,
            1 => true,
            other => vortex_bail!("a bars layout's residual flag is 0 or 1, not {other}"),
        };
        vortex_ensure!(
            !residual || dollar_volume_residual(args.dtype).is_some(),
            "a bars layout stores a dollar volume its table can't have as a residual"
        );
        vortex_ensure!(
            args.children.nchildren() == 2,
            "a bars layout has two children, its table and the source's zones"
        );
        vortex_ensure!(args.children.child_row_count(0) == args.row_count);
        vortex_ensure!(args.children.child_row_count(1) == args.row_count.div_ceil(zone_len));
        args.children
            .child(0, &stored_dtype(args.dtype, parts, residual)?)?;
        args.children
            .child(1, &zones_dtype(args.dtype, residual)?)?;
        Ok(BarsData {
            parts,
            zone_len,
            residual,
        })
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => stored_dtype(layout.dtype(), layout.parts, layout.residual),
            1 => zones_dtype(layout.dtype(), layout.residual),
            _ => vortex_bail!("a bars layout has no child {slot}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        match slot {
            0 => LayoutChildType::Transparent("data".into()),
            _ => LayoutChildType::Auxiliary("zones".into()),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let data = data_child(&layout.to_layout())?;
        let inner =
            data.new_reader(Arc::clone(&name), Arc::clone(&segment_source), session, ctx)?;
        let zones = Arc::clone(&layout.to_layout().children()?[1]);
        Ok(Arc::new(BarsReader::new(
            name,
            layout.dtype().clone(),
            inner,
            (zones, layout.zone_len),
            layout.residual,
            segment_source,
            session.clone(),
        )?))
    }
}

/// The table a bars layout wraps.
pub fn data_child(layout: &LayoutRef) -> VortexResult<LayoutRef> {
    Ok(Arc::clone(&layout.children()?[0]))
}

/// A bars layout's source zones (see [`zones_dtype`]) and the rows per zone; `None` for any other
/// layout.
pub fn source_zones(layout: &LayoutRef) -> Option<(LayoutRef, u64)> {
    let bars = layout.as_opt::<BarsLayout>()?;
    Some((Arc::clone(&layout.children().ok()?[1]), bars.zone_len))
}

/// The field of the source's zones (see [`zones_dtype`]) with `column`'s min, or its max.
pub fn bound_name(column: &str, max: bool) -> String {
    let bound = if max { "max" } else { "min" };
    format!("{column}.{bound}")
}

/// The dtype of the source's zones for a table of `source`'s dtype, with the dollar volume stored
/// as a `residual` or not: a row per zone, and the min and max there ([`bound_name`]) of each
/// column the stored table doesn't have as it is (see [`zoned_columns`]), at its type, null where
/// a zone has none.
pub fn zones_dtype(source: &DType, residual: bool) -> VortexResult<DType> {
    let mut names: Vec<FieldName> = Vec::new();
    let mut dtypes = Vec::new();
    for (name, decimal) in zoned_columns(source, residual)? {
        for max in [false, true] {
            names.push(bound_name(name, max).into());
            dtypes.push(DType::Decimal(decimal, Nullability::Nullable));
        }
    }
    Ok(DType::Struct(
        StructFields::new(names.into(), dtypes),
        Nullability::NonNullable,
    ))
}

/// The columns of a table of `source`'s dtype that the stored table doesn't have as they are,
/// with their decimal types: the four prices, and the dollar volume if it's stored as a
/// `residual` (see [`dollar_volume_residual`]).
fn zoned_columns(
    source: &DType,
    residual: bool,
) -> VortexResult<Vec<(&'static str, DecimalDType)>> {
    let mut columns = Vec::new();
    for price in Price::ALL {
        let Some(DType::Decimal(decimal, _)) = field(source, price.name()) else {
            vortex_bail!("{} is not a decimal", price.name());
        };
        columns.push((price.name(), decimal));
    }
    if let Some(decimal) = dollar_volume_residual(source).filter(|_| residual) {
        columns.push((DOLLAR_VOLUME, decimal));
    }
    Ok(columns)
}

/// `source`'s field `name`, if it has one.
fn field(source: &DType, name: &str) -> Option<DType> {
    source.as_struct_fields_opt()?.field(name)
}

/// The dollar volume's decimal type, if a table of `source`'s dtype can store it as a residual:
/// if it has a decimal [`DOLLAR_VOLUME`] and [`VOLUME`] whose product with the open fits its
/// scale. Whether it does is the writer's choice (see
/// [`BarsStrategy::with_dollar_volume_residual`]).
fn dollar_volume_residual(source: &DType) -> Option<DecimalDType> {
    let (Some(DType::Decimal(dollar_volume, _)), Some(DType::Decimal(volume, _))) =
        (field(source, DOLLAR_VOLUME), field(source, VOLUME))
    else {
        return None;
    };
    let Some(DType::Decimal(open, _)) = field(source, Price::Open.name()) else {
        return None;
    };
    shift(dollar_volume.scale(), volume.scale(), open.scale())?;
    Some(dollar_volume)
}

/// Each zone's min and max of some of a table's columns, from the chunks written so far.
struct ZoneBounds {
    zone_len: u64,
    rows: u64,
    /// Per column, per zone: the least and the greatest valid value.
    bounds: Vec<Vec<(Option<i128>, Option<i128>)>>,
}

impl ZoneBounds {
    fn new(zone_len: u64, columns: usize) -> Self {
        Self {
            zone_len,
            rows: 0,
            bounds: vec![Vec::new(); columns],
        }
    }

    /// Add the next chunk's columns: each one's unscaled values and which are valid.
    fn add(&mut self, columns: &[(&[i128], &Mask)]) {
        let len = columns.first().map_or(0, |(values, _)| values.len());
        let nzones = (self.rows + len as u64).div_ceil(self.zone_len) as usize;
        let mut start = 0;
        while start < len {
            let row = self.rows + start as u64;
            let zone = (row / self.zone_len) as usize;
            let end = len.min(start + (self.zone_len - row % self.zone_len) as usize);
            for ((values, valid), bounds) in columns.iter().zip(&mut self.bounds) {
                bounds.resize(nzones, (None, None));
                let (lo, hi) = &mut bounds[zone];
                for (i, &value) in values[start..end].iter().enumerate() {
                    if valid.value(start + i) {
                        *lo = Some(lo.map_or(value, |lo| lo.min(value)));
                        *hi = Some(hi.map_or(value, |hi| hi.max(value)));
                    }
                }
            }
            start = end;
        }
        self.rows += len as u64;
    }

    /// The bounds as the source's zones (see [`zones_dtype`]) of the columns `columns` (see
    /// [`zoned_columns`]).
    fn table(&self, columns: &[(&str, DecimalDType)]) -> VortexResult<ArrayRef> {
        let nzones = self.rows.div_ceil(self.zone_len) as usize;
        let mut fields = Vec::new();
        for (bounds, &(_, decimal)) in self.bounds.iter().zip(columns) {
            for max in [false, true] {
                let values = (0..nzones).map(|zone| {
                    let (lo, hi) = bounds.get(zone).copied().unwrap_or_default();
                    if max {
                        hi
                    } else {
                        lo
                    }
                });
                fields.push(DecimalArray::from_option_iter(values, decimal).into_array());
            }
        }
        let names = (columns.iter())
            .flat_map(|&(name, _)| [false, true].map(|max| FieldName::from(bound_name(name, max))));
        Ok(
            StructArray::try_new(names.collect(), fields, nzones, Validity::NonNullable)?
                .into_array(),
        )
    }
}

/// The stored table's dtype, given the source's, the parts' decimal type and whether the dollar
/// volume is stored as a `residual`: the [`OHLC`] struct where open is, no other price, and
/// [`DOLLAR_VOLUME_RESIDUAL`] where the dollar volume is, if it's stored as one.
fn stored_dtype(source: &DType, parts: DecimalDType, residual: bool) -> VortexResult<DType> {
    let fields = source
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("a table of bars is a struct, not {source}"))?;
    let (mut names, mut dtypes): (Vec<FieldName>, Vec<DType>) = (Vec::new(), Vec::new());
    for (name, dtype) in fields.names().iter().zip(fields.fields()) {
        if residual && name.as_ref() == DOLLAR_VOLUME {
            names.push(DOLLAR_VOLUME_RESIDUAL.into());
            dtypes.push(dtype);
            continue;
        }
        match Price::from_name(name.as_ref()) {
            Some(Price::Open) => {
                let part = DType::Decimal(parts, Nullability::NonNullable);
                let part_names: Vec<FieldName> =
                    Part::ALL.iter().map(|p| p.name().into()).collect();
                let ohlc = StructFields::new(part_names.into(), vec![part; Part::ALL.len()]);
                names.push(OHLC.into());
                dtypes.push(DType::Struct(ohlc, dtype.nullability()));
            }
            Some(_) => {}
            None => {
                names.push(name.clone());
                dtypes.push(dtype);
            }
        }
    }
    let fields = StructFields::new(names.into(), dtypes);
    Ok(DType::Struct(fields, source.nullability()))
}

/// A chunk of a table of bars as stored, with `parts` (see [`parts_dtype`]): the four prices as
/// the [`OHLC`] struct's parts, where open is in the source, and the dollar volume as its residual
/// if it's stored as one (see [`stored_dtype`]). Its prices, and its dollar volume if it's a
/// residual, go into `zones`.
fn stored_chunk(
    chunk: ArrayRef,
    parts: DecimalDType,
    residual: bool,
    zones: &Mutex<ZoneBounds>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let mut ctx = session.create_execution_ctx();
    let table = chunk.execute::<StructArray>(&mut ctx)?;
    let len = table.len();
    let mut prices = Vec::with_capacity(Price::ALL.len());
    for price in Price::ALL {
        let column = (table.unmasked_field_by_name(price.name())?.clone())
            .execute::<DecimalArray>(&mut ctx)?;
        let valid = column.validity()?.execute_mask(len, &mut ctx)?;
        vortex_ensure!(
            valid.all_true(),
            "{} has nulls, which the bars layout can't store yet",
            price.name()
        );
        prices.push(converted_buffer::<i128>(&column, &valid)?);
    }
    let [open, high, low, close] = [&prices[0], &prices[1], &prices[2], &prices[3]];
    let valid = Mask::new_true(len);
    let residual = match residual {
        true => Some(residual_column(&table, open, &mut ctx)?),
        false => None,
    };
    let mut zoned: Vec<(&[i128], &Mask)> = vec![
        (open, &valid),
        (high, &valid),
        (low, &valid),
        (close, &valid),
    ];
    if let Some(residual) = &residual {
        zoned.push((&residual.dollar_volume, &residual.valid));
    }
    zones.lock().unwrap().add(&zoned);
    let limit = 10i128.pow(u32::from(parts.precision()));
    let mut columns: [BufferMut<i128>; 4] = std::array::from_fn(|_| BufferMut::with_capacity(len));
    for i in 0..len {
        let bar = ohlc_to_parts(open[i], high[i], low[i], close[i])
            .filter(|bar| bar.iter().all(|part| part.abs() < limit))
            .ok_or_else(|| vortex_err!("a bar's parts don't fit {} digits", parts.precision()))?;
        for (column, part) in columns.iter_mut().zip(bar) {
            column.push(part);
        }
    }
    let nullable = table
        .unmasked_field_by_name(Price::Open.name())?
        .dtype()
        .is_nullable();
    let ohlc = StructArray::try_new(
        Part::ALL
            .map(|part| FieldName::from(part.name()))
            .to_vec()
            .into(),
        columns.map(|column| {
            DecimalArray::new(column.freeze(), parts, Validity::NonNullable).into_array()
        }),
        len,
        match nullable {
            true => Validity::AllValid,
            false => Validity::NonNullable,
        },
    )?;
    let (mut names, mut fields): (Vec<FieldName>, Vec<ArrayRef>) = (Vec::new(), Vec::new());
    for (name, field) in table.names().iter().zip(table.iter_unmasked_fields()) {
        if let Some(residual) = residual.as_ref().filter(|_| name.as_ref() == DOLLAR_VOLUME) {
            names.push(DOLLAR_VOLUME_RESIDUAL.into());
            fields.push(residual.residual.clone());
            continue;
        }
        match Price::from_name(name.as_ref()) {
            Some(Price::Open) => {
                names.push(OHLC.into());
                fields.push(ohlc.clone().into_array());
            }
            Some(_) => {}
            None => {
                names.push(name.clone());
                fields.push(field.clone());
            }
        }
    }
    Ok(StructArray::try_new(names.into(), fields, len, table.struct_validity())?.into_array())
}

/// The [`Residual`] of `table`'s dollar volume, a chunk of a table of bars that stores it as one,
/// given its unscaled `open` prices.
fn residual_column(
    table: &StructArray,
    open: &[i128],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Residual> {
    let decimal = |name: &str, ctx: &mut ExecutionCtx| {
        (table.unmasked_field_by_name(name)?.clone()).execute::<DecimalArray>(ctx)
    };
    let open_scale = match field(table.dtype(), Price::Open.name()) {
        Some(DType::Decimal(decimal, _)) => decimal.scale(),
        _ => vortex_bail!("open is not a decimal"),
    };
    let (dollar_volume, volume) = (decimal(DOLLAR_VOLUME, ctx)?, decimal(VOLUME, ctx)?);
    residual(&dollar_volume, &volume, open, open_scale, ctx)
}

/// The parts' decimal type for a table of `source`'s dtype, if it is a table of bars: one with
/// the four prices, all decimals of one type. The parts keep the prices' scale, with a digit more
/// (a part is a price or the difference of two), up to [`MAX_PART_PRECISION`].
fn parts_dtype(source: &DType) -> Option<DecimalDType> {
    let fields = source.as_struct_fields_opt()?;
    let [open, high, low, close] = Price::ALL.map(|price| fields.field(price.name()));
    let Some(DType::Decimal(price, nullability)) = open else {
        return None;
    };
    [high, low, close]
        .iter()
        .all(|dtype| *dtype == Some(DType::Decimal(price, nullability)))
        .then(|| {
            let precision = (price.precision() + 1).min(MAX_PART_PRECISION);
            DecimalDType::new(precision, price.scale())
        })
}

/// Writes a table of bars (see [`parts_dtype`]) with `inner`, its four prices as the [`OHLC`]
/// struct's parts, and wraps it in a [`BarsLayout`] with the source's dtype and the source's
/// zones of `zone_len` rows (see [`zones_dtype`]). Writes any other table with `inner` as it is.
pub struct BarsStrategy {
    inner: Arc<dyn LayoutStrategy>,
    zone_len: u64,
    residual: bool,
}

impl BarsStrategy {
    /// The strategy, storing the dollar volume as a residual where it can (see
    /// [`Self::with_dollar_volume_residual`]).
    pub fn new(inner: Arc<dyn LayoutStrategy>, zone_len: u64) -> Self {
        assert!(zone_len > 0, "a zone has rows");
        Self {
            inner,
            zone_len,
            residual: true,
        }
    }

    /// Whether to store the dollar volume as a residual against `volume × open` where the table
    /// can (see [`crate::dollar_volume`]). It's smaller, 29.7 MiB less for a month of minute bars
    /// (−12.6%), but reading the dollar volume then decodes the volume and the open as well:
    /// filters on it were 14–23% slower, and a scan of it alone 35–85%.
    pub fn with_dollar_volume_residual(mut self, residual: bool) -> Self {
        self.residual = residual;
        self
    }
}

#[async_trait]
impl LayoutStrategy for BarsStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let source = stream.dtype().clone();
        let Some(parts) = parts_dtype(&source) else {
            return (self.inner)
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        };
        let residual = self.residual && dollar_volume_residual(&source).is_some();
        let stored = stored_dtype(&source, parts, residual)?;
        let zoned = zoned_columns(&source, residual)?;
        let zone_bounds = Arc::new(Mutex::new(ZoneBounds::new(self.zone_len, zoned.len())));
        let (chunk_zones, chunk_session) = (Arc::clone(&zone_bounds), session.clone());
        let stream = SequentialStreamAdapter::new(
            stored,
            stream.map(move |chunk| {
                let (sequence, chunk) = chunk?;
                let chunk = stored_chunk(chunk, parts, residual, &chunk_zones, &chunk_session)?;
                Ok((sequence, chunk))
            }),
        )
        .sendable();
        // The data's segments come before the zones', as a zoned layout's do.
        let data_eof = eof.split_off();
        let data = self
            .inner
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;
        let table = zone_bounds.lock().unwrap().table(&zoned)?;
        let table = table.to_array_stream().sequenced(eof.split_off());
        let zones = FlatLayoutStrategy::default()
            .write_stream(ctx, segment_sink, table, eof, session)
            .await?;
        Ok(LayoutParts::new(
            BarsLayout,
            source,
            data.row_count(),
            Vec::new(),
            layout_children(vec![data, zones]),
            BarsData {
                parts,
                zone_len: self.zone_len,
                residual,
            },
        )
        .into_layout())
    }
}

struct BarsReader {
    name: Arc<str>,
    /// The source's dtype.
    dtype: DType,
    /// The stored table's reader.
    inner: LayoutReaderRef,
    /// Each price's decimal type, in [`Price::ALL`] order.
    prices: [DecimalDType; 4],
    /// The dollar volume's decimal type, if it's stored as a residual.
    dollar_volume: Option<DecimalDType>,
    /// Expressions over the source's columns, rewritten over the stored table and bound.
    rewritten: Mutex<HashMap<ExactBoundExpr, BoundExpression>>,
    /// The source's zones and the rows per zone.
    zones: (LayoutRef, u64),
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    /// The prices' bounds per zone, read once per reader, on the first filter on a price.
    bounds: OnceLock<SharedBounds>,
    /// Per filter, the zones the prices' bounds rule out (see [`Self::ruled_out`]).
    ruled_out: Mutex<HashMap<ExactBoundExpr, Option<SharedRuledOut>>>,
}

impl BarsReader {
    fn new(
        name: Arc<str>,
        dtype: DType,
        inner: LayoutReaderRef,
        zones: (LayoutRef, u64),
        residual: bool,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("a table of bars is a struct"))?;
        let prices = Price::ALL.map(|price| match fields.field(price.name()) {
            Some(DType::Decimal(decimal, _)) => Ok(decimal),
            _ => Err(vortex_err!("{} is not a decimal", price.name())),
        });
        let [open, high, low, close] = prices;
        let dollar_volume = dollar_volume_residual(&dtype).filter(|_| residual);
        Ok(Self {
            name,
            dtype,
            inner,
            prices: [open?, high?, low?, close?],
            dollar_volume,
            rewritten: Mutex::new(HashMap::new()),
            zones,
            segment_source,
            session,
            bounds: OnceLock::new(),
            ruled_out: Mutex::new(HashMap::new()),
        })
    }

    fn price_decimal(&self, price: Price) -> DecimalDType {
        self.prices[price as usize]
    }

    /// `expr`, over the source's columns, as an expression over the stored table.
    fn to_stored(&self, expr: &BoundExpression) -> VortexResult<Expression> {
        if expr.is_root() {
            let fields = self.dtype.as_struct_fields_opt().expect("a struct");
            let elements = fields
                .names()
                .iter()
                .map(|name| (name.clone(), self.column(name.as_ref())));
            return Ok(pack(elements, self.dtype.nullability()));
        }
        if let Some(name) = expr.as_opt::<GetItem>() {
            if expr.child(0).is_root() {
                return Ok(self.column(name.as_ref()));
            }
        }
        let Some(scalar_fn) = expr.as_scalar() else {
            vortex_bail!("can't rewrite {expr}");
        };
        let children = expr
            .children()
            .iter()
            .map(|child| self.to_stored(child))
            .collect::<VortexResult<Vec<_>>>()?;
        Expression::try_new(scalar_fn.clone(), children)
    }

    /// The source's column `name`, over the stored table.
    fn column(&self, name: &str) -> Expression {
        if let Some(decimal) = self.dollar_volume.filter(|_| name == DOLLAR_VOLUME) {
            return dollar_volume(
                decimal,
                get_item(DOLLAR_VOLUME_RESIDUAL, root()),
                get_item(VOLUME, root()),
                get_item(Part::Open.name(), get_item(OHLC, root())),
            );
        }
        match Price::from_name(name) {
            Some(price) => ohlc_price(price, self.price_decimal(price), get_item(OHLC, root())),
            None => get_item(name, root()),
        }
    }

    /// [`Self::to_stored`], bound to the stored table, once per expression.
    fn stored(&self, expr: &BoundExpression) -> VortexResult<BoundExpression> {
        let key = ExactBoundExpr(expr.clone());
        if let Some(bound) = self.rewritten.lock().unwrap().get(&key) {
            return Ok(bound.clone());
        }
        let bound = self.to_stored(expr)?.bind(self.inner.dtype())?;
        self.rewritten.lock().unwrap().insert(key, bound.clone());
        Ok(bound)
    }

    fn bounds(&self) -> SharedBounds {
        self.bounds.get_or_init(|| self.read_bounds()).clone()
    }

    /// Read the source's zones.
    fn read_bounds(&self) -> SharedBounds {
        let (zones, zone_len) = (Arc::clone(&self.zones.0), self.zones.1);
        let row_count = self.inner.row_count();
        let n = zones.row_count();
        let read = zones
            .new_reader(
                format!("{}.zones", self.name).into(),
                Arc::clone(&self.segment_source),
                &self.session,
                &LayoutReaderContext::new(),
            )
            .and_then(|reader| {
                let all = root().bind(zones.dtype())?;
                reader.projection_evaluation(&(0..n), &all, MaskFuture::new_true(n as usize))
            });
        async move {
            let table = read.ok()?.await.ok()?;
            Some(Arc::new(PriceBounds {
                zone_len,
                row_count,
                table,
            }))
        }
        .boxed()
        .shared()
    }

    /// The zones the prices' bounds rule out for `expr`, once per filter: Vortex's own rules
    /// for proving a filter false from statistics ([`BoundExpression::falsify`]), with each
    /// price's min and max taken from the source's zones. `None` if `expr` has no proof that uses
    /// a price's.
    fn ruled_out(&self, expr: &BoundExpression) -> VortexResult<Option<SharedRuledOut>> {
        let key = ExactBoundExpr(expr.clone());
        if let Some(ruled_out) = self.ruled_out.lock().unwrap().get(&key) {
            return Ok(ruled_out.clone());
        }
        let binder = PriceStats {
            table: self.zones.0.dtype().clone(),
            bound: Cell::new(false),
        };
        let proof = match expr.falsify(&self.session)? {
            Some(proof) => {
                let proof = bind_stats(proof, &binder)?;
                binder.bound.get().then_some(proof)
            }
            None => None,
        };
        let ruled_out = proof.map(|proof| {
            let (bounds, session) = (self.bounds(), self.session.clone());
            async move {
                let bounds = bounds.await?;
                let mut ctx = session.create_execution_ctx();
                let proven = bounds.table.clone().apply_bound(&proof).ok()?;
                let zones = proven
                    .fill_null(false)
                    .ok()?
                    .execute::<Mask>(&mut ctx)
                    .ok()?;
                Some(Arc::new(RuledOut { bounds, zones }))
            }
            .boxed()
            .shared()
        });
        self.ruled_out
            .lock()
            .unwrap()
            .insert(key, ruled_out.clone());
        Ok(ruled_out)
    }
}

impl LayoutReader for BarsReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.inner.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // A price is read from the struct, and the dollar volume from its residual, the volume and
        // the struct.
        let masks: Vec<FieldMask> = field_mask
            .iter()
            .flat_map(|mask| match mask {
                FieldMask::Prefix(path) | FieldMask::Exact(path) => {
                    let first = path.parts().first().and_then(|field| field.as_name());
                    let prefix = |name: &str| FieldMask::Prefix(FieldPath::from_name(name));
                    match first {
                        Some(name) if self.dollar_volume.is_some() && name == DOLLAR_VOLUME => {
                            vec![prefix(DOLLAR_VOLUME_RESIDUAL), prefix(VOLUME), prefix(OHLC)]
                        }
                        Some(name) if Price::from_name(name).is_some() => vec![prefix(OHLC)],
                        _ => vec![mask.clone()],
                    }
                }
                FieldMask::All => vec![FieldMask::All],
            })
            .collect();
        self.inner.register_splits(&masks, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let Some(ruled_out) = self.ruled_out(expr)? else {
            return self
                .inner
                .pruning_evaluation(row_range, &self.stored(expr)?, mask);
        };
        let row_range = row_range.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            let Some(ruled_out) = ruled_out.await else {
                return Ok(mask);
            };
            let zone_len = ruled_out.bounds.zone_len;
            let end = row_range.end.min(ruled_out.bounds.row_count);
            let mut keep: Vec<(usize, usize)> = Vec::new();
            for zone in row_range.start / zone_len..end.div_ceil(zone_len) {
                if ruled_out.zones.value(zone as usize) {
                    continue;
                }
                let start = ((zone * zone_len).max(row_range.start) - row_range.start) as usize;
                let stop = (((zone + 1) * zone_len).min(end) - row_range.start) as usize;
                match keep.last_mut() {
                    Some((_, last)) if *last == start => *last = stop,
                    _ => keep.push((start, stop)),
                }
            }
            Ok(&mask & &Mask::from_slices(mask.len(), keep))
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.inner
            .filter_evaluation(row_range, &self.stored(expr)?, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.inner
            .projection_evaluation(row_range, &self.stored(expr)?, mask)
    }
}

/// Binds the min and max of a column in the source's zones (a price, the dollar volume) to its
/// field there, and no other statistic.
struct PriceStats {
    /// The table's dtype.
    table: DType,
    /// Whether a price's min or max was bound.
    bound: Cell<bool>,
}

impl StatBinder for PriceStats {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(name) = input.as_opt::<GetItem>() else {
            return Ok(None);
        };
        if !input.child(0).is_root() {
            return Ok(None);
        }
        let max = match Stat::from_aggregate_fn(aggregate_fn) {
            Some(Stat::Min) => false,
            Some(Stat::Max) => true,
            _ => return Ok(None),
        };
        let bound = bound_name(name.as_ref(), max);
        let fields = self.table.as_struct_fields_opt();
        if fields.and_then(|fields| fields.field(&bound)).is_none() {
            return Ok(None);
        }
        self.bound.set(true);
        Ok(Some(get_item(bound, root()).bind(&self.table)?))
    }
}

/// The zones a filter's proof rules out (`true`), one per zone of `bounds`.
struct RuledOut {
    bounds: Arc<PriceBounds>,
    zones: Mask,
}

type SharedRuledOut = Shared<BoxFuture<'static, Option<Arc<RuledOut>>>>;

/// The source's zones (see [`zones_dtype`]) as read, with their rows per zone.
struct PriceBounds {
    zone_len: u64,
    row_count: u64,
    table: ArrayRef,
}

type SharedBounds = Shared<BoxFuture<'static, Option<Arc<PriceBounds>>>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunks that don't line up with the zones (3, 4 and 1 rows, zones of 3) give each zone
    /// its own rows' bounds, without the nulls, and a zone of nulls none.
    #[test]
    fn zone_bounds_follow_rows_not_chunks() {
        let mut zones = ZoneBounds::new(3, 1);
        let chunks: [(&[i128], Mask); 3] = [
            (&[5, 1, 9], Mask::new_true(3)),
            (&[4, 8, 7, 100], Mask::from_iter([true, true, true, false])),
            (&[2], Mask::new_true(1)),
        ];
        for (values, valid) in &chunks {
            zones.add(&[(values, valid)]);
        }
        assert_eq!(zones.rows, 8);
        assert_eq!(
            zones.bounds[0],
            vec![(Some(1), Some(9)), (Some(4), Some(8)), (Some(2), Some(2))]
        );
        let mut nulls = ZoneBounds::new(2, 1);
        nulls.add(&[(&[1, 2, 3], &Mask::from_iter([false, false, true]))]);
        assert_eq!(nulls.bounds[0], vec![(None, None), (Some(3), Some(3))]);
    }
}
