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
use vortex::error::{vortex_bail, vortex_ensure, vortex_err, VortexError, VortexResult};
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

use crate::decimal::unscaled;
use crate::ohlc::{ohlc_to_parts, Part, Price};
use crate::price_fn::{ohlc_price, OhlcPriceFn};
use crate::product_residual::{product_residual, residual, shift};
use crate::scaled::predecoded;

/// The stored table's struct column holding the four prices as their [`Part`]s, where open is in
/// the source.
pub const OHLC: &str = "ohlc";

/// The most digits a part may have: an `i128`'s.
const MAX_PART_PRECISION: u8 = 38;

/// A decimal column of the source that the layout stores as its residual against the product of
/// two others, `target - a × b` (see [`crate::product_residual`]), under [`Self::stored_name`].
///
/// A table can store it if all three are decimals, `a`'s and `b`'s scales add up to no more than
/// the target's, the target is no price and the factors no target, and the table has no column
/// of the stored name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProductResidual {
    pub target: FieldName,
    pub a: FieldName,
    pub b: FieldName,
}

impl ProductResidual {
    pub fn new(
        target: impl Into<FieldName>,
        a: impl Into<FieldName>,
        b: impl Into<FieldName>,
    ) -> Self {
        Self {
            target: target.into(),
            a: a.into(),
            b: b.into(),
        }
    }

    /// A bar's dollar volume against its volume × open. It's smaller, 29.7 MiB less for a month
    /// of minute bars (−12.6%), but reading the dollar volume then decodes the volume and the
    /// open as well: filters on it were 14–23% slower, and a scan of it alone 35–85%.
    pub fn dollar_volume() -> Self {
        Self::new("dollar_volume", "volume", Price::Open.name())
    }

    /// The stored table's column in the target's place, `<target>_residual`: a name no source
    /// column has, so that its zone map is never taken for the target's.
    pub fn stored_name(&self) -> FieldName {
        format!("{}_residual", self.target).into()
    }

    /// The target, `a` and `b`.
    fn names(&self) -> [&FieldName; 3] {
        [&self.target, &self.a, &self.b]
    }
}

/// Register the layout, [`OhlcPriceFn`] and
/// [`ProductResidualFn`](crate::product_residual::ProductResidualFn), so files that use them can
/// be read back.
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
        .register(crate::product_residual::ProductResidualFn);
}

#[derive(Clone, Debug)]
pub struct BarsLayout;

/// The parts' decimal type and the columns stored as residuals, which with the layout's dtype give
/// the stored table's, and the rows per zone of the source's zones.
#[derive(Clone, Debug)]
pub struct BarsData {
    parts: DecimalDType,
    zone_len: u64,
    residuals: Vec<ProductResidual>,
}

impl VTable for BarsLayout {
    type LayoutData = BarsData;
    type Metadata = RawMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex_ohlc.bars");
        *ID
    }

    /// The parts' precision and scale, the zone length as a little-endian `u64`, then for each
    /// column stored as a residual its target's name, `a`'s and `b`'s, each as its length in
    /// bytes (a little-endian `u32`) and its UTF-8.
    fn metadata(layout: &Layout<Self>) -> RawMetadata {
        let mut metadata = vec![layout.parts.precision(), layout.parts.scale() as u8];
        metadata.extend(layout.zone_len.to_le_bytes());
        for name in layout.residuals.iter().flat_map(ProductResidual::names) {
            let len = u32::try_from(name.as_ref().len()).expect("a column name under 4 GiB");
            metadata.extend(len.to_le_bytes());
            metadata.extend(name.as_ref().as_bytes());
        }
        RawMetadata(metadata)
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &Vec<u8>,
    ) -> VortexResult<BarsData> {
        let short = || vortex_err!("a bars layout's metadata is cut short");
        let [precision, scale, rest @ ..] = metadata.as_slice() else {
            return Err(short());
        };
        let (zone_len, mut rest) = rest.split_first_chunk::<8>().ok_or_else(short)?;
        let zone_len = u64::from_le_bytes(*zone_len);
        vortex_ensure!(zone_len > 0, "a bars layout's zones have rows");
        let parts = DecimalDType::new(*precision, *scale as i8);
        let mut names = Vec::new();
        while let Some((len, tail)) = rest.split_first_chunk::<4>() {
            let len = u32::from_le_bytes(*len) as usize;
            let name = tail.get(..len).ok_or_else(short)?;
            let name = std::str::from_utf8(name)
                .map_err(|err| vortex_err!("a bars layout's column name isn't UTF-8: {err}"))?;
            names.push(FieldName::from(name));
            rest = &tail[len..];
        }
        let (triples, []) = names.as_chunks::<3>() else {
            return Err(short());
        };
        if !rest.is_empty() {
            return Err(short());
        }
        let residuals: Vec<ProductResidual> = (triples.iter())
            .map(|[target, a, b]| ProductResidual::new(target.clone(), a.clone(), b.clone()))
            .collect();
        residual_decimals(args.dtype, &residuals)?;
        vortex_ensure!(
            args.children.nchildren() == 2,
            "a bars layout has two children, its table and the source's zones"
        );
        vortex_ensure!(args.children.child_row_count(0) == args.row_count);
        vortex_ensure!(args.children.child_row_count(1) == args.row_count.div_ceil(zone_len));
        args.children
            .child(0, &stored_dtype(args.dtype, parts, &residuals)?)?;
        args.children
            .child(1, &zones_dtype(args.dtype, &residuals)?)?;
        Ok(BarsData {
            parts,
            zone_len,
            residuals,
        })
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => stored_dtype(layout.dtype(), layout.parts, &layout.residuals),
            1 => zones_dtype(layout.dtype(), &layout.residuals),
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
            layout.residuals.clone(),
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
pub fn source_zones(layout: &LayoutRef) -> VortexResult<Option<(LayoutRef, u64)>> {
    let Some(bars) = layout.as_opt::<BarsLayout>() else {
        return Ok(None);
    };
    Ok(Some((Arc::clone(&layout.children()?[1]), bars.zone_len)))
}

/// The field of the source's zones (see [`zones_dtype`]) with `column`'s min, or its max.
pub fn bound_name(column: &str, max: bool) -> String {
    let bound = if max { "max" } else { "min" };
    format!("{column}.{bound}")
}

/// The dtype of the source's zones for a table of `source`'s dtype, with `residuals`: a row per
/// zone, and the min and max there ([`bound_name`]) of each column the stored table doesn't have
/// as it is (see [`zoned_columns`]), at its type, null where a zone has none.
pub fn zones_dtype(source: &DType, residuals: &[ProductResidual]) -> VortexResult<DType> {
    let mut names: Vec<FieldName> = Vec::new();
    let mut dtypes = Vec::new();
    for (name, decimal) in zoned_columns(source, residuals)? {
        for max in [false, true] {
            names.push(bound_name(name.as_ref(), max).into());
            dtypes.push(DType::Decimal(decimal, Nullability::Nullable));
        }
    }
    Ok(DType::Struct(
        StructFields::new(names.into(), dtypes),
        Nullability::NonNullable,
    ))
}

/// The columns of a table of `source`'s dtype that the stored table doesn't have as they are,
/// with their decimal types: the four prices, and the targets of `residuals`.
fn zoned_columns(
    source: &DType,
    residuals: &[ProductResidual],
) -> VortexResult<Vec<(FieldName, DecimalDType)>> {
    let mut columns = Vec::new();
    for price in Price::ALL {
        let Some(DType::Decimal(decimal, _)) = field(source, price.name()) else {
            vortex_bail!("{} is not a decimal", price.name());
        };
        columns.push((price.name().into(), decimal));
    }
    let targets = residuals.iter().map(|residual| residual.target.clone());
    columns.extend(targets.zip(residual_decimals(source, residuals)?));
    Ok(columns)
}

/// `source`'s field `name`, if it has one.
fn field(source: &DType, name: &str) -> Option<DType> {
    source.as_struct_fields_opt()?.field(name)
}

/// The targets' decimal types, if a table of `source`'s dtype can store `residuals` (see
/// [`ProductResidual`]).
fn residual_decimals(
    source: &DType,
    residuals: &[ProductResidual],
) -> VortexResult<Vec<DecimalDType>> {
    let decimal = |name: &FieldName| match field(source, name.as_ref()) {
        Some(DType::Decimal(decimal, _)) => Ok(decimal),
        Some(other) => Err(vortex_err!("{name} is {other}, not a decimal")),
        None => Err(vortex_err!("the table has no {name}")),
    };
    let is_target = |name: &FieldName| residuals.iter().any(|other| other.target == *name);
    let mut decimals = Vec::with_capacity(residuals.len());
    for (i, residual) in residuals.iter().enumerate() {
        let ProductResidual { target, a, b } = residual;
        let (decimal, a_decimal, b_decimal) = (decimal(target)?, decimal(a)?, decimal(b)?);
        vortex_ensure!(
            Price::from_name(target.as_ref()).is_none(),
            "{target} is a price, which the bars layout stores as a part, not as a residual"
        );
        vortex_ensure!(
            !residuals[..i].iter().any(|other| other.target == *target),
            "{target} is stored as a residual twice"
        );
        vortex_ensure!(
            !is_target(a) && !is_target(b),
            "{target} is stored against {a} × {b}, but a factor is stored as a residual itself"
        );
        let stored = residual.stored_name();
        vortex_ensure!(
            field(source, stored.as_ref()).is_none(),
            "{target}'s residual would be stored as {stored}, a column the table has"
        );
        vortex_ensure!(
            shift(decimal.scale(), a_decimal.scale(), b_decimal.scale()).is_some(),
            "{target}'s scale is below {a} × {b}'s"
        );
        decimals.push(decimal);
    }
    Ok(decimals)
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
    fn table(&self, columns: &[(FieldName, DecimalDType)]) -> VortexResult<ArrayRef> {
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
        let names = (columns.iter()).flat_map(|(name, _)| {
            [false, true].map(|max| FieldName::from(bound_name(name.as_ref(), max)))
        });
        Ok(
            StructArray::try_new(names.collect(), fields, nzones, Validity::NonNullable)?
                .into_array(),
        )
    }
}

/// The dtype of the table a [`BarsStrategy`] with `residuals` hands its inner strategy for a
/// table of `source`'s dtype, so a writer can name the inner strategy's columns: `None` for a
/// table that isn't a table of bars, which the inner strategy writes as it is.
pub fn stored_table_dtype(
    source: &DType,
    residuals: &[ProductResidual],
) -> VortexResult<Option<DType>> {
    let Some(parts) = parts_dtype(source) else {
        return Ok(None);
    };
    stored_dtype(source, parts, &residuals_of(source, residuals)).map(Some)
}

/// The columns of `residuals` a table of `source`'s dtype stores: those whose three columns it
/// has.
fn residuals_of(source: &DType, residuals: &[ProductResidual]) -> Vec<ProductResidual> {
    (residuals.iter())
        .filter(|residual| {
            (residual.names().iter()).all(|name| field(source, name.as_ref()).is_some())
        })
        .cloned()
        .collect()
}

/// The stored table's dtype, given the source's, the parts' decimal type and the columns stored as
/// `residuals`: the [`OHLC`] struct where open is, no other price, and each residual's stored
/// column where its target is.
fn stored_dtype(
    source: &DType,
    parts: DecimalDType,
    residuals: &[ProductResidual],
) -> VortexResult<DType> {
    let fields = source
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("a table of bars is a struct, not {source}"))?;
    let (mut names, mut dtypes): (Vec<FieldName>, Vec<DType>) = (Vec::new(), Vec::new());
    for (name, dtype) in fields.names().iter().zip(fields.fields()) {
        if let Some(residual) = residuals.iter().find(|residual| residual.target == *name) {
            names.push(residual.stored_name());
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
/// the [`OHLC`] struct's parts, where open is in the source, and the targets of `residuals` as
/// their residuals (see [`stored_dtype`]). Its prices and those targets go into `zones`.
fn stored_chunk(
    chunk: ArrayRef,
    parts: DecimalDType,
    residuals: &[ProductResidual],
    zones: &Mutex<ZoneBounds>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let mut ctx = session.create_execution_ctx();
    let table = chunk.execute::<StructArray>(&mut ctx)?;
    let len = table.len();
    let mut prices = Vec::with_capacity(Price::ALL.len());
    for price in Price::ALL {
        let (values, valid) = unscaled(&decimal_column(&table, price.name(), &mut ctx)?, &mut ctx)?;
        vortex_ensure!(
            valid.all_true(),
            "{} has nulls, which the bars layout can't store yet",
            price.name()
        );
        prices.push(values);
    }
    let [open, high, low, close] = [&prices[0], &prices[1], &prices[2], &prices[3]];
    let valid = Mask::new_true(len);
    let mut stored_residuals = Vec::with_capacity(residuals.len());
    for column in residuals {
        let [target, a, b] = column
            .names()
            .map(|name| decimal_column(&table, name.as_ref(), &mut ctx));
        stored_residuals.push(residual(&target?, &a?, &b?, &mut ctx)?);
    }
    let mut zoned: Vec<(&[i128], &Mask)> = vec![
        (open, &valid),
        (high, &valid),
        (low, &valid),
        (close, &valid),
    ];
    zoned.extend((stored_residuals.iter()).map(|r| (r.values.as_slice(), &r.valid)));
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
        if let Some(i) = residuals.iter().position(|column| column.target == *name) {
            names.push(residuals[i].stored_name());
            fields.push(stored_residuals[i].residual.clone());
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

/// `table`'s column `name`, a decimal, decoded.
fn decimal_column(
    table: &StructArray,
    name: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalArray> {
    (table.unmasked_field_by_name(name)?.clone()).execute::<DecimalArray>(ctx)
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
    residuals: Vec<ProductResidual>,
}

impl BarsStrategy {
    /// The strategy, storing the dollar volume as a residual where the table has it (see
    /// [`Self::with_product_residuals`]).
    pub fn new(inner: Arc<dyn LayoutStrategy>, zone_len: u64) -> Self {
        assert!(zone_len > 0, "a zone has rows");
        Self {
            inner,
            zone_len,
            residuals: vec![ProductResidual::dollar_volume()],
        }
    }

    /// The columns to store as residuals. One whose three columns a table hasn't all is left out
    /// of its file; one it has but can't store (see [`ProductResidual`]) fails the write.
    pub fn with_product_residuals(mut self, residuals: Vec<ProductResidual>) -> Self {
        self.residuals = residuals;
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
        let residuals = residuals_of(&source, &self.residuals);
        let stored = stored_dtype(&source, parts, &residuals)?;
        let zoned = zoned_columns(&source, &residuals)?;
        let zone_bounds = Arc::new(Mutex::new(ZoneBounds::new(self.zone_len, zoned.len())));
        let (chunk_zones, chunk_session) = (Arc::clone(&zone_bounds), session.clone());
        let chunk_residuals = residuals.clone();
        let stream = SequentialStreamAdapter::new(
            stored,
            stream.map(move |chunk| {
                let (sequence, chunk) = chunk?;
                let chunk =
                    stored_chunk(chunk, parts, &chunk_residuals, &chunk_zones, &chunk_session)?;
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
                residuals,
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
    /// The columns stored as residuals, with their targets' decimal types.
    residuals: Vec<(ProductResidual, DecimalDType)>,
    /// Expressions over the source's columns, rewritten over the stored table and bound.
    rewritten: Mutex<HashMap<ExactBoundExpr, BoundExpression>>,
    /// Per projection, how its prices share the parts they decode (see [`Self::shared_parts`]).
    shared_parts: Mutex<HashMap<ExactBoundExpr, Option<SharedParts>>>,
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
        residuals: Vec<ProductResidual>,
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
        let decimals = residual_decimals(&dtype, &residuals)?;
        Ok(Self {
            name,
            dtype,
            inner,
            prices: [open?, high?, low?, close?],
            residuals: residuals.into_iter().zip(decimals).collect(),
            rewritten: Mutex::new(HashMap::new()),
            shared_parts: Mutex::new(HashMap::new()),
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

    /// The residual whose target is the source's column `name`, if it's stored as one.
    fn residual(&self, name: &str) -> Option<&(ProductResidual, DecimalDType)> {
        (self.residuals.iter()).find(|(residual, _)| residual.target.as_ref() == name)
    }

    /// The source's column `name`, over the stored table.
    fn column(&self, name: &str) -> Expression {
        if let Some((residual, decimal)) = self.residual(name) {
            return product_residual(
                *decimal,
                get_item(residual.stored_name(), root()),
                self.factor(residual.a.as_ref()),
                self.factor(residual.b.as_ref()),
            );
        }
        match Price::from_name(name) {
            Some(price) => ohlc_price(price, self.price_decimal(price), get_item(OHLC, root())),
            None => get_item(name, root()),
        }
    }

    /// A residual's factor `name`, over the stored table: as [`Self::column`] has it, but the
    /// open as its part, which is the open at a digit more precision.
    fn factor(&self, name: &str) -> Expression {
        match Price::from_name(name) {
            Some(Price::Open) => get_item(Part::Open.name(), get_item(OHLC, root())),
            _ => self.column(name),
        }
    }

    /// The stored table's columns that the source's column `name` is read from.
    fn stored_columns(&self, name: &str) -> Vec<FieldName> {
        if let Some((residual, _)) = self.residual(name) {
            let mut columns = vec![residual.stored_name()];
            columns.extend(self.stored_columns(residual.a.as_ref()));
            columns.extend(self.stored_columns(residual.b.as_ref()));
            return columns;
        }
        match Price::from_name(name) {
            Some(_) => vec![OHLC.into()],
            None => vec![name.into()],
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

    /// For a projection that rebuilds more than one price, how they share the parts they
    /// decode, once per expression; `None` for any other, which the stored table's reader
    /// evaluates whole.
    ///
    /// Each [`ohlc_price`] decodes the parts its price uses, and the stored table's reader
    /// evaluates each on its own: a row's four prices decoded the open four times and the body
    /// three, each time filtered to the rows read.
    fn shared_parts(&self, expr: &BoundExpression) -> VortexResult<Option<SharedParts>> {
        let key = ExactBoundExpr(expr.clone());
        if let Some(shared) = self.shared_parts.lock().unwrap().get(&key) {
            return Ok(shared.clone());
        }
        let shared = self.plan_shared_parts(expr)?;
        (self.shared_parts.lock().unwrap()).insert(key, shared.clone());
        Ok(shared)
    }

    fn plan_shared_parts(&self, expr: &BoundExpression) -> VortexResult<Option<SharedParts>> {
        let stored = self.to_stored(expr)?;
        let (mut prices, mut parts, mut columns) = (0, [false; 4], Vec::<FieldName>::new());
        let mut pending = vec![&stored];
        while let Some(node) = pending.pop() {
            if let Some(options) = node.as_opt::<OhlcPriceFn>() {
                prices += 1;
                for (part, used) in Part::ALL.into_iter().zip(&mut parts) {
                    *used |= options.price.uses(part);
                }
            } else if let Some(name) = node.as_opt::<GetItem>() {
                if node.child(0).is_root() {
                    if !columns.contains(name) {
                        columns.push(name.clone());
                    }
                    continue;
                }
            }
            if node.is_root() {
                // The whole stored row, which the stored table's reader is left to read.
                return Ok(None);
            }
            pending.extend(node.children());
        }
        if prices < 2 {
            return Ok(None);
        }
        let reads = columns
            .into_iter()
            .map(|name| (name.clone(), get_item(name, root())));
        let reads = pack(reads, self.inner.dtype().nullability()).bind(self.inner.dtype())?;
        let projection = stored.bind(reads.dtype())?;
        Ok(Some(SharedParts {
            reads,
            projection,
            parts,
        }))
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
        let bounds = async move {
            Ok(Arc::new(PriceBounds {
                zone_len,
                row_count,
                table: read?.await?,
            }))
        };
        async move { bounds.await.map_err(Arc::new) }
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
                let prove = || -> VortexResult<Mask> {
                    let mut ctx = session.create_execution_ctx();
                    let proven = bounds.table.clone().apply_bound(&proof)?;
                    proven.fill_null(false)?.execute::<Mask>(&mut ctx)
                };
                let zones = prove().map_err(Arc::new)?;
                Ok(Arc::new(RuledOut { bounds, zones }))
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
        // A price is read from the struct, and a residual's target from the residual and its
        // factors.
        let masks: Vec<FieldMask> = field_mask
            .iter()
            .flat_map(|mask| match mask {
                FieldMask::Prefix(path) | FieldMask::Exact(path) => {
                    let first = path.parts().first().and_then(|field| field.as_name());
                    let prefix = |name: &str| FieldMask::Prefix(FieldPath::from_name(name));
                    match first {
                        Some(name)
                            if self.residual(name).is_some()
                                || Price::from_name(name).is_some() =>
                        {
                            let columns = self.stored_columns(name);
                            columns
                                .iter()
                                .map(|column| prefix(column.as_ref()))
                                .collect()
                        }
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
            let ruled_out = (ruled_out.await)
                .map_err(|err| vortex_err!("ruling out zones by the prices' bounds: {err}"))?;
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
        let Some(shared) = self.shared_parts(expr)? else {
            return (self.inner).projection_evaluation(row_range, &self.stored(expr)?, mask);
        };
        let read = (self.inner).projection_evaluation(row_range, &shared.reads, mask)?;
        let session = self.session.clone();
        Ok(async move {
            let mut ctx = session.create_execution_ctx();
            let columns = read.await?.execute::<StructArray>(&mut ctx)?;
            let columns = with_parts_predecoded(&columns, shared.parts, &mut ctx)?;
            columns.into_array().apply_bound(&shared.projection)
        }
        .boxed())
    }
}

/// How a projection that rebuilds more than one price reads the [`OHLC`] struct once for all of
/// them (see [`BarsReader::shared_parts`]).
#[derive(Clone)]
struct SharedParts {
    /// The stored columns the projection reads, as a `pack` bound to the stored table.
    reads: BoundExpression,
    /// The projection, bound to what `reads` returns.
    projection: BoundExpression,
    /// The parts its prices use, in [`Part::ALL`] order.
    parts: [bool; 4],
}

/// `columns`, stored columns among them the [`OHLC`] struct, with the `parts` of that struct
/// [`predecoded`], so that the prices using a part share its decoding.
fn with_parts_predecoded(
    columns: &StructArray,
    parts: [bool; 4],
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    let mut fields = Vec::with_capacity(columns.names().len());
    for (name, field) in columns.names().iter().zip(columns.iter_unmasked_fields()) {
        if name.as_ref() != OHLC {
            fields.push(field.clone());
            continue;
        }
        let ohlc = field.clone().execute::<StructArray>(ctx)?;
        let mut decoded = Vec::with_capacity(ohlc.names().len());
        for (name, part) in ohlc.names().iter().zip(ohlc.iter_unmasked_fields()) {
            let used = (Part::ALL.into_iter().zip(parts))
                .any(|(each, used)| used && each.name() == name.as_ref());
            decoded.push(match used {
                true => predecoded(part.clone(), ctx)?,
                false => part.clone(),
            });
        }
        let (dtype, validity) = (ohlc.struct_fields().clone(), ohlc.struct_validity());
        let ohlc = StructArray::try_new_with_dtype(decoded, dtype, ohlc.len(), validity)?;
        fields.push(ohlc.into_array());
    }
    let (dtype, validity) = (columns.struct_fields().clone(), columns.struct_validity());
    StructArray::try_new_with_dtype(fields, dtype, columns.len(), validity)
}

/// Binds the min and max of a column in the source's zones (a price, a residual's target) to its
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

/// Shared by every split of a scan, so an error is shared as well.
type SharedRuledOut = Shared<BoxFuture<'static, Result<Arc<RuledOut>, Arc<VortexError>>>>;

/// The source's zones (see [`zones_dtype`]) as read, with their rows per zone.
struct PriceBounds {
    zone_len: u64,
    row_count: u64,
    table: ArrayRef,
}

type SharedBounds = Shared<BoxFuture<'static, Result<Arc<PriceBounds>, Arc<VortexError>>>>;

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
