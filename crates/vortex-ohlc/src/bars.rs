//! `vortex_ohlc.bars`, the layout at the root of a table of bars, with the source's dtype:
//! four decimal prices where the stored table has the [`OHLC`] struct. Its one child is the table
//! as stored. Its reader rewrites every expression over the source's columns into one over the
//! stored table (a price into [`ohlc_price`] over the struct, the whole row into a `pack`), and
//! prunes a comparison of a price with a literal on the parts' zone maps itself: Vortex's struct
//! reader only prunes an expression that reads one field, and a price reads two or three.

use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::Array;
use arrow::datatypes::DataType;
use async_trait::async_trait;
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use vortex::array::aggregate_fn::AggregateFnRef;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::{DecimalArray, StructArray};
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::expr::stats::Stat;
use vortex::array::stats::bind::{bind_stats, StatBinder};
use vortex::array::validity::Validity;
use vortex::array::{ArrayRef, IntoArray, MaskFuture, RawMetadata, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::{
    DType, DecimalDType, FieldMask, FieldName, FieldPath, Nullability, StructFields,
};
use vortex::error::{vortex_bail, vortex_ensure, vortex_err, VortexResult};
use vortex::expr::{get_item, pack, root, BoundExpression, ExactBoundExpr, Expression};
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::segments::{SegmentSinkRef, SegmentSource};
use vortex::layout::sequence::{SendableSequentialStream, SequencePointer};
use vortex::layout::{
    layout_children, ArrayFuture, Layout, LayoutChildType, LayoutDeserializeArgs, LayoutId,
    LayoutParts, LayoutReader, LayoutReaderContext, LayoutReaderRef, LayoutRef, LayoutStrategy,
    LayoutWriterContext, RowSplits, SplitRange, VTable,
};
use vortex::mask::Mask;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

use crate::ohlc::{Part, Price};
use crate::plan::OHLC;
use crate::price_fn::{ohlc_price, OhlcPriceFn};

/// Register the layout and [`OhlcPriceFn`], so files that use them can be read back.
pub fn register(session: &VortexSession) {
    use vortex::layout::session::LayoutSessionExt;
    use vortex::layout::LayoutEncoding;
    use vortex::scalar_fn::session::ScalarFnSessionExt;
    session
        .layouts()
        .register(&BarsLayout as &dyn LayoutEncoding);
    session.scalar_fns().register(OhlcPriceFn);
}

#[derive(Clone, Debug)]
pub struct BarsLayout;

/// The parts' decimal type, which with the layout's dtype gives the stored table's.
#[derive(Clone, Debug)]
pub struct BarsData {
    parts: DecimalDType,
}

impl VTable for BarsLayout {
    type LayoutData = BarsData;
    type Metadata = RawMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex_ohlc.bars");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> RawMetadata {
        RawMetadata(vec![layout.parts.precision(), layout.parts.scale() as u8])
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &Vec<u8>,
    ) -> VortexResult<BarsData> {
        let &[precision, scale] = metadata.as_slice() else {
            vortex_bail!(
                "a bars layout's metadata is 2 bytes, not {}",
                metadata.len()
            );
        };
        let parts = DecimalDType::new(precision, scale as i8);
        vortex_ensure!(
            args.children.nchildren() == 1,
            "a bars layout has exactly one child, its table"
        );
        vortex_ensure!(args.children.child_row_count(0) == args.row_count);
        args.children.child(0, &stored_dtype(args.dtype, parts)?)?;
        Ok(BarsData { parts })
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => stored_dtype(layout.dtype(), layout.parts),
            _ => vortex_bail!("a bars layout has no child {slot}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, _slot: usize) -> LayoutChildType {
        LayoutChildType::Transparent("data".into())
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
        Ok(Arc::new(BarsReader::new(
            name,
            layout.dtype().clone(),
            inner,
            PartZones::find(&data),
            segment_source,
            session.clone(),
        )?))
    }
}

/// The table a bars layout wraps.
pub fn data_child(layout: &LayoutRef) -> VortexResult<LayoutRef> {
    Ok(Arc::clone(&layout.children()?[0]))
}

/// The stored table's dtype, given the source's and the parts' decimal type: the [`OHLC`] struct
/// where open is, and no other price.
fn stored_dtype(source: &DType, parts: DecimalDType) -> VortexResult<DType> {
    let fields = source
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("a table of bars is a struct, not {source}"))?;
    let (mut names, mut dtypes): (Vec<FieldName>, Vec<DType>) = (Vec::new(), Vec::new());
    for (name, dtype) in fields.names().iter().zip(fields.fields()) {
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

/// The source's dtype, given the stored table's and the prices' decimal type, and the parts'
/// decimal type: the four prices where the [`OHLC`] struct is. The inverse of [`stored_dtype`].
fn source_dtype(stored: &DType, price: DecimalDType) -> VortexResult<(DType, DecimalDType)> {
    let fields = stored
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("a table of bars is a struct, not {stored}"))?;
    let (mut names, mut dtypes): (Vec<FieldName>, Vec<DType>) = (Vec::new(), Vec::new());
    let mut parts = None;
    for (name, dtype) in fields.names().iter().zip(fields.fields()) {
        if name.as_ref() != OHLC {
            names.push(name.clone());
            dtypes.push(dtype);
            continue;
        }
        let ohlc = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("{OHLC} is {dtype}, not a struct"))?;
        let Some(DType::Decimal(decimal, _)) = ohlc.field(Part::Open.name()) else {
            vortex_bail!("{OHLC} has no decimal open");
        };
        parts = Some(decimal);
        for p in Price::ALL {
            names.push(p.name().into());
            dtypes.push(DType::Decimal(price, dtype.nullability()));
        }
    }
    let parts = parts.ok_or_else(|| vortex_err!("the table has no {OHLC} struct"))?;
    let fields = StructFields::new(names.into(), dtypes);
    Ok((DType::Struct(fields, stored.nullability()), parts))
}

/// Writes the table with `inner` and wraps it in a [`BarsLayout`] whose prices are `price`.
pub struct BarsStrategy {
    inner: Arc<dyn LayoutStrategy>,
    price: DecimalDType,
}

impl BarsStrategy {
    pub fn new(inner: Arc<dyn LayoutStrategy>, price: DecimalDType) -> Self {
        Self { inner, price }
    }
}

#[async_trait]
impl LayoutStrategy for BarsStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let data = self
            .inner
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await?;
        let (source, parts) = source_dtype(data.dtype(), self.price)?;
        Ok(LayoutParts::new(
            BarsLayout,
            source,
            data.row_count(),
            Vec::new(),
            layout_children(vec![data]),
            BarsData { parts },
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
    /// Expressions over the source's columns, rewritten over the stored table and bound.
    rewritten: Mutex<HashMap<ExactBoundExpr, BoundExpression>>,
    zones: Option<PartZones>,
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
        zones: Option<PartZones>,
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
        Ok(Self {
            name,
            dtype,
            inner,
            prices: [open?, high?, low?, close?],
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
        self.bounds
            .get_or_init(|| match &self.zones {
                None => futures::future::ready(None).boxed().shared(),
                Some(zones) => {
                    zones.read(&self.name, &self.segment_source, &self.session, self.prices)
                }
            })
            .clone()
    }

    /// The zones the prices' bounds rule out for `expr`, once per filter: Vortex's own rules
    /// for proving a filter false from statistics ([`BoundExpression::falsify`]), with each
    /// price's min and max taken from [`PriceBounds`]. `None` if `expr` has no proof that uses a
    /// price's.
    fn ruled_out(&self, expr: &BoundExpression) -> VortexResult<Option<SharedRuledOut>> {
        let key = ExactBoundExpr(expr.clone());
        if let Some(ruled_out) = self.ruled_out.lock().unwrap().get(&key) {
            return Ok(ruled_out.clone());
        }
        let binder = PriceStats {
            table: price_bounds_dtype(self.prices),
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
        // A price is read from the struct.
        let masks: Vec<FieldMask> = field_mask
            .iter()
            .map(|mask| match mask {
                FieldMask::Prefix(path) | FieldMask::Exact(path) => {
                    let first = path.parts().first().and_then(|field| field.as_name());
                    match first.and_then(Price::from_name) {
                        Some(_) => FieldMask::Prefix(FieldPath::from_name(OHLC)),
                        None => mask.clone(),
                    }
                }
                FieldMask::All => FieldMask::All,
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

/// Binds a price's min and max to its field in the [`PriceBounds`] table, and no other statistic.
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
        let Some(price) = Price::from_name(name.as_ref()).filter(|_| input.child(0).is_root())
        else {
            return Ok(None);
        };
        let max = match Stat::from_aggregate_fn(aggregate_fn) {
            Some(Stat::Min) => false,
            Some(Stat::Max) => true,
            _ => return Ok(None),
        };
        self.bound.set(true);
        Ok(Some(
            get_item(bound_name(price, max), root()).bind(&self.table)?,
        ))
    }
}

/// The zones a filter's proof rules out (`true`), one per zone of `bounds`.
struct RuledOut {
    bounds: Arc<PriceBounds>,
    zones: Mask,
}

type SharedRuledOut = Shared<BoxFuture<'static, Option<Arc<RuledOut>>>>;

/// Where the [`OHLC`] struct's parts keep their zone maps, in [`Part::ALL`] order, all with the
/// same zones.
struct PartZones {
    zone_len: u64,
    row_count: u64,
    nzones: usize,
    parts: Vec<LayoutRef>,
}

/// One part's min and max per zone, unscaled at the parts' scale, `None` where the zone map
/// has no bound.
type ZoneBounds = (Vec<Option<i128>>, Vec<Option<i128>>);

/// Each price's bounds per zone, from its parts' (see [`price_bound`]): a table with a row per
/// zone and two fields per price, its min and max ([`bound_name`]) at its own type, null where a
/// part's zone map has no bound.
struct PriceBounds {
    zone_len: u64,
    row_count: u64,
    table: ArrayRef,
}

type SharedBounds = Shared<BoxFuture<'static, Option<Arc<PriceBounds>>>>;

/// The [`PriceBounds`] table's field for `price`'s min, or its max.
fn bound_name(price: Price, max: bool) -> String {
    let bound = if max { "max" } else { "min" };
    format!("{}.{bound}", price.name())
}

/// The [`PriceBounds`] table's dtype, for prices of the types `prices`.
fn price_bounds_dtype(prices: [DecimalDType; 4]) -> DType {
    let (names, dtypes): (Vec<FieldName>, Vec<DType>) = (Price::ALL.iter())
        .flat_map(|&price| [false, true].map(|max| (price, max)))
        .map(|(price, max)| {
            let dtype = DType::Decimal(prices[price as usize], Nullability::Nullable);
            (bound_name(price, max).into(), dtype)
        })
        .unzip();
    DType::Struct(
        StructFields::new(names.into(), dtypes),
        Nullability::NonNullable,
    )
}

/// The [`PriceBounds`] table from the parts' bounds `parts`, at the parts' `scale`.
fn price_bounds_table(parts: &[ZoneBounds], scale: i8, prices: [DecimalDType; 4]) -> ArrayRef {
    let nzones = parts[0].0.len();
    let fields = (Price::ALL.iter()).flat_map(|&price| {
        let decimal = prices[price as usize];
        let factor = 10i128.checked_pow((decimal.scale() - scale) as u32);
        [false, true].map(|max| {
            let bounds = (0..nzones).map(|zone| price_bound(parts, zone, price, max, factor));
            DecimalArray::from_option_iter(bounds, decimal).into_array()
        })
    });
    let names = (Price::ALL.iter())
        .flat_map(|&price| [false, true].map(|max| FieldName::from(bound_name(price, max))));
    StructArray::try_new(names.collect(), fields, nzones, Validity::NonNullable)
        .expect("a field per bound, a row per zone")
        .into_array()
}

/// `price`'s least value in `zone`, or its greatest (`max`), from its parts' bounds there and
/// `factor` (from the parts' scale to the price's): a price grows with every part but the lower
/// wick (see [`Price::from_parts`]). `None` where a part has no bound.
fn price_bound(
    parts: &[ZoneBounds],
    zone: usize,
    price: Price,
    max: bool,
    factor: Option<i128>,
) -> Option<i128> {
    let mut at = [0i128; 4];
    for (i, part) in Part::ALL.into_iter().enumerate() {
        if !price.uses(part) {
            continue;
        }
        let (min_bounds, max_bounds) = &parts[i];
        at[i] = match (part == Part::Lower) == max {
            true => min_bounds[zone]?,
            false => max_bounds[zone]?,
        };
    }
    Some(price.from_parts(at).saturating_mul(factor?))
}

impl PartZones {
    /// `None` unless the stored table `data` has an [`OHLC`] struct, every part of which has a
    /// zone map, all with the same zones.
    fn find(data: &LayoutRef) -> Option<Self> {
        let ohlc = find_child(data, OHLC)?;
        let mut parts = Vec::new();
        let mut shape = None;
        for part in Part::ALL {
            let zoned = find_zoned(&find_child(&ohlc, part.name())?)?;
            let typed = zoned.as_opt::<Zoned>()?;
            let this = (typed.zone_len() as u64, typed.nzones());
            if *shape.get_or_insert(this) != this {
                return None;
            }
            parts.push(Arc::clone(&zoned.children().ok()?[1]));
        }
        let (zone_len, nzones) = shape?;
        Some(Self {
            zone_len,
            row_count: data.row_count(),
            nzones,
            parts,
        })
    }

    /// Read every part's zone map.
    ///
    /// *Upstream*: Vortex's own `ZoneMap` reads a zone map, bounded statistics included, and
    /// prunes with it (vortex-layout 0.89 `zoned/zone_map.rs:74,144`). Building one needs the
    /// zoned layout's aggregate functions, which are `pub(crate)` (`ZonedData::aggregate_fns`,
    /// `zoned/mod.rs:380`), as is the zone map `ZonedReader` keeps (`zoned/pruning.rs:156`). If
    /// Vortex made either public, this and the pruner's `read_zone_map` could use it.
    fn read(
        &self,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        session: &VortexSession,
        prices: [DecimalDType; 4],
    ) -> SharedBounds {
        let n = self.nzones;
        let reads: Vec<_> = Part::ALL
            .iter()
            .zip(&self.parts)
            .map(|(part, zones)| {
                let reader = zones.new_reader(
                    format!("{name}.{}.zones", part.name()).into(),
                    Arc::clone(segment_source),
                    session,
                    &LayoutReaderContext::new(),
                )?;
                let all = root().bind(zones.dtype())?;
                reader.projection_evaluation(&(0..n as u64), &all, MaskFuture::new_true(n))
            })
            .collect();
        let session = session.clone();
        let (zone_len, row_count) = (self.zone_len, self.row_count);
        async move {
            let mut parts = Vec::new();
            let mut scale = None;
            for read in reads {
                let table = read.ok()?.await.ok()?;
                let mut ctx = session.create_execution_ctx();
                let table = table.execute::<StructArray>(&mut ctx).ok()?;
                let mut stat = |want: &str| -> Option<Vec<Option<i128>>> {
                    let array = table
                        .unmasked_field_by_name(format!("vortex.{want}()"))
                        .ok()?
                        .clone();
                    let array = session.arrow().execute_arrow(array, None, &mut ctx).ok()?;
                    let s = match array.data_type() {
                        DataType::Decimal32(_, s)
                        | DataType::Decimal64(_, s)
                        | DataType::Decimal128(_, s) => *s,
                        _ => return None,
                    };
                    scale = Some(s);
                    let wide = arrow::compute::cast(&array, &DataType::Decimal128(38, s)).ok()?;
                    let wide = wide.as_primitive::<Decimal128Type>();
                    Some(
                        (0..wide.len())
                            .map(|i| wide.is_valid(i).then(|| wide.value(i)))
                            .collect(),
                    )
                };
                parts.push((stat("min")?, stat("max")?));
            }
            Some(Arc::new(PriceBounds {
                zone_len,
                row_count,
                table: price_bounds_table(&parts, scale?, prices),
            }))
        }
        .boxed()
        .shared()
    }
}

/// The child named `name` of the first layout at or under `layout` that has one, depth first.
fn find_child(layout: &LayoutRef, name: &str) -> Option<LayoutRef> {
    let children = layout.children().ok()?;
    if let Some(index) = layout.child_names().position(|n| n.as_ref() == name) {
        return children.get(index).map(Arc::clone);
    }
    children.iter().find_map(|child| find_child(child, name))
}

/// The first zoned layout at or under `layout`, depth first.
fn find_zoned(layout: &LayoutRef) -> Option<LayoutRef> {
    if layout.is::<Zoned>() {
        return Some(Arc::clone(layout));
    }
    layout.children().ok()?.iter().find_map(find_zoned)
}
