//! `vortex_ohlc.bars`: the layout at the root of a table of bars, which wraps the table's own
//! layout with no data of its own. Its reader hands everything to the table's reader, except:
//!
//! - **Pruning on the prices.** A filter on high, low or close is an expression over two or three
//!   of the [`OHLC`] struct's parts (see [`OHLC_PARTS`](crate::plan::OHLC_PARTS)). Each part has
//!   its own zone map, but Vortex's struct reader only prunes an expression that reads a single
//!   field, so it scans every chunk for these. This reader also bounds the expression from the
//!   parts' zone maps: a part is within its zone's min and max, a sum within the sum of its
//!   operands' bounds, and a `zip` within the bounds of the branch the zone's bounds pick (or of
//!   either). A zone can be skipped when those bounds rule out every comparison the expression
//!   needs.
//! - **Projections wait for the filter.** Vortex's scan asks every split for its projection
//!   before the split's filter has run, so the reader can start its I/O early, and throws that
//!   work away for a split the filter rules out entirely: with a selective filter, most of them.
//!   This reader asks the table's reader for a projection only once the split's mask is known,
//!   so a ruled-out split costs nothing, and a kept split's I/O starts after its filter rather
//!   than alongside it.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::Array;
use arrow::datatypes::DataType;
use async_trait::async_trait;
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::StructArray;
use vortex::array::{EmptyMetadata, MaskFuture, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::{DType, FieldMask, FieldName};
use vortex::error::{vortex_bail, vortex_ensure, VortexResult};
use vortex::expr::{root, BoundExpression};
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::segments::{SegmentSinkRef, SegmentSource};
use vortex::layout::sequence::{SendableSequentialStream, SequencePointer};
use vortex::layout::{
    layout_children, ArrayFuture, Layout, LayoutChildType, LayoutDeserializeArgs, LayoutId,
    LayoutParts, LayoutReader, LayoutReaderContext, LayoutReaderRef, LayoutRef, LayoutStrategy,
    LayoutWriterContext, RowSplits, SplitRange, VTable,
};
use vortex::mask::Mask;
use vortex::metrics::{Counter, MetricBuilder, MetricsRegistry};
use vortex::scalar_fn::fns::between::{Between, StrictComparison};
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::cast::Cast;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::zip::Zip;
use vortex::session::registry::CachedId;
use vortex::session::{SessionExt, SessionVar, VortexSession};

use crate::plan::OHLC;

/// Register the layout, so files that use it can be read back.
pub fn register(session: &VortexSession) {
    use vortex::layout::session::LayoutSessionExt;
    use vortex::layout::LayoutEncoding;
    session
        .layouts()
        .register(&BarsLayout as &dyn LayoutEncoding);
}

/// Where the bars layout's readers record what their pruning did, when a session carries one
/// (see [`record_metrics`]):
///
/// - `vortex_ohlc.bars.zones`: the zones a pruning evaluation of a price filter looked at;
/// - `vortex_ohlc.bars.zones_kept`: those its bounds couldn't rule out.
#[derive(Clone)]
pub struct BarsMetrics {
    registry: Arc<dyn MetricsRegistry>,
}

impl std::fmt::Debug for BarsMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarsMetrics").finish_non_exhaustive()
    }
}

impl SessionVar for BarsMetrics {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Have the bars layout's readers on `session` record their pruning in `registry` (see
/// [`BarsMetrics`]). Readers made before this record nothing.
pub fn record_metrics(session: &VortexSession, registry: Arc<dyn MetricsRegistry>) {
    session.register(BarsMetrics { registry });
}

/// The counters one reader records into.
#[derive(Clone)]
struct PruneCounters {
    zones: Counter,
    kept: Counter,
}

impl PruneCounters {
    fn new(session: &VortexSession) -> Option<Self> {
        let metrics = session.get_opt::<BarsMetrics>()?;
        let counter = |name| MetricBuilder::new(metrics.registry.as_ref()).counter(name);
        Some(Self {
            zones: counter("vortex_ohlc.bars.zones"),
            kept: counter("vortex_ohlc.bars.zones_kept"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct BarsLayout;

impl VTable for BarsLayout {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex_ohlc.bars");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> EmptyMetadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<()> {
        vortex_ensure!(
            args.children.nchildren() == 1,
            "a bars layout has exactly one child, its table"
        );
        vortex_ensure!(args.children.child_row_count(0) == args.row_count);
        args.children.child(0, args.dtype)?;
        Ok(())
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => Ok(layout.dtype().clone()),
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
        let data = data_child(&layout.to_layout());
        let inner =
            data.new_reader(Arc::clone(&name), Arc::clone(&segment_source), session, ctx)?;
        Ok(Arc::new(BarsReader {
            name,
            inner,
            zones: PartZones::find(&data),
            segment_source,
            session: session.clone(),
            bounds: OnceLock::new(),
            counters: PruneCounters::new(session),
        }))
    }
}

/// The table a bars layout wraps.
pub fn data_child(layout: &LayoutRef) -> LayoutRef {
    Arc::clone(&layout.children().unwrap()[0])
}

/// Writes the table with `inner` and wraps it in a [`BarsLayout`].
pub struct BarsStrategy {
    inner: Arc<dyn LayoutStrategy>,
}

impl BarsStrategy {
    pub fn new(inner: Arc<dyn LayoutStrategy>) -> Self {
        Self { inner }
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
        Ok(LayoutParts::new(
            BarsLayout,
            data.dtype().clone(),
            data.row_count(),
            Vec::new(),
            layout_children(vec![data]),
            (),
        )
        .into_layout())
    }
}

/// Where the [`OHLC`] struct's parts keep their zone maps: the zones layout of each part, all
/// with the same zones.
struct PartZones {
    zone_len: u64,
    row_count: u64,
    nzones: usize,
    parts: Vec<(FieldName, LayoutRef)>,
}

impl PartZones {
    /// `None` unless the table `data` has an [`OHLC`] struct, every field of which has a zone
    /// map, all with the same zones.
    fn find(data: &LayoutRef) -> Option<Self> {
        let ohlc = find_child(data, OHLC)?;
        let fields = ohlc.dtype().as_struct_fields_opt()?;
        let names: HashMap<Arc<str>, LayoutRef> =
            ohlc.child_names().zip(ohlc.children().ok()?).collect();
        let mut parts = Vec::new();
        let mut shape = None;
        for name in fields.names().iter() {
            let zoned = find_zoned(names.get(name.as_ref())?)?;
            let typed = zoned.as_opt::<Zoned>()?;
            let this = (typed.zone_len() as u64, typed.nzones());
            if *shape.get_or_insert(this) != this {
                return None;
            }
            parts.push((name.clone(), Arc::clone(&zoned.children().ok()?[1])));
        }
        let (zone_len, nzones) = shape?;
        Some(Self {
            zone_len,
            row_count: data.row_count(),
            nzones,
            parts,
        })
    }
}

/// The child named `name` of the first layout at or under `layout` that has one, depth first:
/// a field of the table's struct layout, wherever that is.
pub(crate) fn find_child(layout: &LayoutRef, name: &str) -> Option<LayoutRef> {
    let children = layout.children().ok()?;
    if let Some(index) = layout.child_names().position(|n| n.as_ref() == name) {
        return children.get(index).map(Arc::clone);
    }
    children.iter().find_map(|child| find_child(child, name))
}

/// The first zoned layout at or under `layout`, depth first.
pub(crate) fn find_zoned(layout: &LayoutRef) -> Option<LayoutRef> {
    if layout.is::<Zoned>() {
        return Some(Arc::clone(layout));
    }
    layout.children().ok()?.iter().find_map(find_zoned)
}

/// One part's min and max per zone, unscaled at the part's own scale, `None` where the zone map
/// has no bound (a zone with no values).
type ZoneBounds = (Vec<Option<i128>>, Vec<Option<i128>>);

/// Each part's zone bounds, by field index.
struct Bounds {
    zone_len: u64,
    row_count: u64,
    parts: Vec<ZoneBounds>,
}

type SharedBounds = Shared<BoxFuture<'static, Option<Arc<Bounds>>>>;

struct BarsReader {
    name: Arc<str>,
    inner: LayoutReaderRef,
    zones: Option<PartZones>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    /// Read once per reader, on the first expression this layout can prune.
    bounds: OnceLock<SharedBounds>,
    /// Where pruning is recorded, if the session asks for it (see [`BarsMetrics`]).
    counters: Option<PruneCounters>,
}

impl BarsReader {
    fn bounds(&self) -> SharedBounds {
        self.bounds
            .get_or_init(|| {
                let Some(zones) = &self.zones else {
                    return futures::future::ready(None).boxed().shared();
                };
                let reads: Vec<_> = zones
                    .parts
                    .iter()
                    .map(|(name, zones_layout)| {
                        let n = zones.nzones;
                        zones_layout
                            .new_reader(
                                format!("{}.{name}.zones", self.name).into(),
                                Arc::clone(&self.segment_source),
                                &self.session,
                                &LayoutReaderContext::new(),
                            )
                            .and_then(|reader| {
                                let all = root().bind(zones_layout.dtype())?;
                                reader.projection_evaluation(
                                    &(0..n as u64),
                                    &all,
                                    MaskFuture::new_true(n),
                                )
                            })
                    })
                    .collect();
                let session = self.session.clone();
                let (zone_len, row_count) = (zones.zone_len, zones.row_count);
                async move {
                    let mut parts = Vec::new();
                    for read in reads {
                        let table = read.ok()?.await.ok()?;
                        let mut ctx = session.create_execution_ctx();
                        let table = table.execute::<StructArray>(&mut ctx).ok()?;
                        let mut stat = |want: &str| -> Option<Vec<Option<i128>>> {
                            let idx = table
                                .names()
                                .iter()
                                .position(|n| n.as_ref() == format!("vortex.{want}()"))?;
                            let array = table.unmasked_field(idx).clone();
                            let array =
                                session.arrow().execute_arrow(array, None, &mut ctx).ok()?;
                            let scale = match array.data_type() {
                                DataType::Decimal32(_, s)
                                | DataType::Decimal64(_, s)
                                | DataType::Decimal128(_, s) => *s,
                                _ => return None,
                            };
                            let wide =
                                arrow::compute::cast(&array, &DataType::Decimal128(38, scale))
                                    .ok()?;
                            let wide = wide.as_primitive::<Decimal128Type>();
                            Some(
                                (0..wide.len())
                                    .map(|i| wide.is_valid(i).then(|| wide.value(i)))
                                    .collect(),
                            )
                        };
                        parts.push((stat("min")?, stat("max")?));
                    }
                    Some(Arc::new(Bounds {
                        zone_len,
                        row_count,
                        parts,
                    }))
                }
                .boxed()
                .shared()
            })
            .clone()
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
        self.inner.dtype()
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
        self.inner.register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let predicate = Predicate::parse(expr, true);
        if self.zones.is_none() || matches!(predicate, Predicate::Unknown) {
            return self.inner.pruning_evaluation(row_range, expr, mask);
        }
        // The table's reader is asked only about the zones these bounds keep: its own pruning
        // splits the expression up for every call, which costs more than a split it can't
        // prune (most of them, for a price filter) is worth.
        let bounds = self.bounds();
        let (inner, row_range, expr) = (Arc::clone(&self.inner), row_range.clone(), expr.clone());
        let counters = self.counters.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            let Some(bounds) = bounds.await else {
                return inner.pruning_evaluation(&row_range, &expr, mask)?.await;
            };
            let zone_len = bounds.zone_len;
            let end = row_range.end.min(bounds.row_count);
            let zones = row_range.start / zone_len..end.div_ceil(zone_len);
            let mut keep = Vec::new();
            let mut kept = 0;
            for zone in zones.clone() {
                let zone = zone as usize;
                let decided = predicate.decide(&|i| {
                    let (min, max) = bounds.parts.get(i)?;
                    Some((min[zone]?, max[zone]?))
                });
                if decided == Decided::None {
                    continue;
                }
                kept += 1;
                let zone = zone as u64;
                let start = (zone * zone_len).max(row_range.start) - row_range.start;
                let stop = ((zone + 1) * zone_len).min(end) - row_range.start;
                match keep.last_mut() {
                    Some((_, last)) if *last == start as usize => *last = stop as usize,
                    _ => keep.push((start as usize, stop as usize)),
                }
            }
            if let Some(counters) = counters {
                counters.zones.add(zones.end - zones.start);
                counters.kept.add(kept);
            }
            let pruned = &mask & &Mask::from_slices(mask.len(), keep);
            if pruned.all_false() {
                return Ok(pruned);
            }
            inner.pruning_evaluation(&row_range, &expr, pruned)?.await
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.inner.filter_evaluation(row_range, expr, mask)
    }

    /// Asks the table's reader once the split's mask is known (see the module docs), or right
    /// away when it already is.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        if mask.peek().is_some() {
            return self.inner.projection_evaluation(row_range, expr, mask);
        }
        let (inner, row_range, expr) = (Arc::clone(&self.inner), row_range.clone(), expr.clone());
        Ok(async move {
            let mask = mask.await?;
            let projection =
                inner.projection_evaluation(&row_range, &expr, MaskFuture::ready(mask))?;
            projection.await
        }
        .boxed())
    }
}

/// A decimal expression over the [`OHLC`] struct's parts (by field index) that can be bounded:
/// every leaf has the same scale, so values and bounds are unscaled.
#[derive(Debug)]
enum Term {
    Part(usize),
    Literal(i128),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    /// The first term where the predicate holds, the second where it doesn't.
    Zip(Box<Predicate>, Box<Term>, Box<Term>),
}

impl Term {
    /// The term and its scale; `None` for anything but parts, decimal literals, sums, casts
    /// that keep the scale, and `zip`s of those on a predicate over them.
    fn parse(expr: &BoundExpression) -> Option<(Self, i8)> {
        let DType::Decimal(decimal, _) = expr.dtype() else {
            return None;
        };
        let scale = decimal.scale();
        // A part, `ohlc.<part>` in the table.
        if let Some(name) = expr.as_opt::<GetItem>() {
            let ohlc = expr.child(0);
            let in_table = ohlc.as_opt::<GetItem>()?.as_ref() == OHLC && ohlc.child(0).is_root();
            let index = ohlc.dtype().as_struct_fields_opt()?.find(name)?;
            return in_table.then_some((Term::Part(index), scale));
        }
        if let Some(scalar) = expr.as_opt::<Literal>() {
            let value = scalar.as_decimal().decimal_value()?.cast::<i128>()?;
            return Some((Term::Literal(value), scale));
        }
        if expr.as_opt::<Cast>().is_some() {
            let (term, inner) = Term::parse(expr.child(0))?;
            return (inner == scale).then_some((term, scale));
        }
        let operand = |i: usize| {
            let (term, s) = Term::parse(expr.child(i))?;
            (s == scale).then(|| Box::new(term))
        };
        if expr.as_opt::<Zip>().is_some() {
            // Its children are the value where the mask holds, where it doesn't, and the mask.
            let mask = match Predicate::parse(expr.child(2), false) {
                Predicate::Unknown => return None,
                mask => mask,
            };
            return Some((Term::Zip(Box::new(mask), operand(0)?, operand(1)?), scale));
        }
        match expr.as_opt::<Binary>()? {
            Operator::Add => Some((Term::Add(operand(0)?, operand(1)?), scale)),
            Operator::Sub => Some((Term::Sub(operand(0)?, operand(1)?), scale)),
            _ => None,
        }
    }

    /// The smallest and largest value the term can take, given each part's bounds (`None`
    /// where a part has none).
    fn bounds(&self, part: &impl Fn(usize) -> Option<(i128, i128)>) -> Option<(i128, i128)> {
        match self {
            Term::Part(index) => part(*index),
            Term::Literal(v) => Some((*v, *v)),
            Term::Add(lhs, rhs) => {
                let (a, b) = (lhs.bounds(part)?, rhs.bounds(part)?);
                Some((a.0.saturating_add(b.0), a.1.saturating_add(b.1)))
            }
            Term::Sub(lhs, rhs) => {
                let (a, b) = (lhs.bounds(part)?, rhs.bounds(part)?);
                Some((a.0.saturating_sub(b.1), a.1.saturating_sub(b.0)))
            }
            // Where the bounds decide the mask, the branch it picks; otherwise either.
            Term::Zip(mask, if_true, if_false) => match mask.decide(part) {
                Decided::All => if_true.bounds(part),
                Decided::None => if_false.bounds(part),
                Decided::Some => {
                    let (a, b) = (if_true.bounds(part)?, if_false.bounds(part)?);
                    Some((a.0.min(b.0), a.1.max(b.1)))
                }
            },
        }
    }

    /// The term's value, given each part's.
    #[cfg(test)]
    fn value(&self, part: &impl Fn(usize) -> i128) -> i128 {
        match self {
            Term::Part(index) => part(*index),
            Term::Literal(v) => *v,
            Term::Add(lhs, rhs) => lhs.value(part) + rhs.value(part),
            Term::Sub(lhs, rhs) => lhs.value(part) - rhs.value(part),
            Term::Zip(mask, if_true, if_false) => match mask.holds(part) {
                true => if_true.value(part),
                false => if_false.value(part),
            },
        }
    }
}

/// What a predicate's bounds say about the rows they cover.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Decided {
    None,
    All,
    Some,
}

/// A boolean expression over [`Term`]s; `Unknown` for anything else.
#[derive(Debug)]
enum Predicate {
    Compare(Operator, Term, Term),
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Unknown,
}

impl Predicate {
    /// With `lenient`, an AND keeps whichever side it can bound when the other is unknown,
    /// which only lets more rows through: good enough to prune on, but not to evaluate.
    fn parse(expr: &BoundExpression, lenient: bool) -> Self {
        // Vortex 0.89's scan turns `open >= a AND open < b` into one `between(open, a, b)`: the
        // AND of its two comparisons.
        if let Some(options) = expr.as_opt::<Between>() {
            let lower = match options.lower_strict {
                StrictComparison::Strict => Operator::Gt,
                StrictComparison::NonStrict => Operator::Gte,
            };
            let upper = match options.upper_strict {
                StrictComparison::Strict => Operator::Lt,
                StrictComparison::NonStrict => Operator::Lte,
            };
            let above = Self::compare(lower, expr.child(0), expr.child(1));
            let below = Self::compare(upper, expr.child(0), expr.child(2));
            return Self::and(above, below, lenient);
        }
        let Some(&op) = expr.as_opt::<Binary>() else {
            return Predicate::Unknown;
        };
        let sides = || {
            (
                Self::parse(expr.child(0), lenient),
                Self::parse(expr.child(1), lenient),
            )
        };
        match op {
            Operator::And => {
                let (l, r) = sides();
                Self::and(l, r, lenient)
            }
            Operator::Or => match sides() {
                (Predicate::Unknown, _) | (_, Predicate::Unknown) => Predicate::Unknown,
                (l, r) => Predicate::Or(Box::new(l), Box::new(r)),
            },
            Operator::Eq
            | Operator::NotEq
            | Operator::Gt
            | Operator::Gte
            | Operator::Lt
            | Operator::Lte => Self::compare(op, expr.child(0), expr.child(1)),
            _ => Predicate::Unknown,
        }
    }

    /// `lhs op rhs` over two terms at one scale.
    fn compare(op: Operator, lhs: &BoundExpression, rhs: &BoundExpression) -> Self {
        match (Term::parse(lhs), Term::parse(rhs)) {
            (Some((l, ls)), Some((r, rs))) if ls == rs => Predicate::Compare(op, l, r),
            _ => Predicate::Unknown,
        }
    }

    /// `l AND r`; with `lenient`, whichever side is known when the other isn't.
    fn and(l: Self, r: Self, lenient: bool) -> Self {
        match (l, r) {
            (Predicate::Unknown, p) | (p, Predicate::Unknown) if lenient => p,
            (Predicate::Unknown, _) | (_, Predicate::Unknown) => Predicate::Unknown,
            (l, r) => Predicate::And(Box::new(l), Box::new(r)),
        }
    }

    /// Whether no row, every row or only some rows within the parts' bounds can satisfy the
    /// predicate.
    fn decide(&self, part: &impl Fn(usize) -> Option<(i128, i128)>) -> Decided {
        match self {
            Predicate::Compare(op, lhs, rhs) => {
                let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (lhs.bounds(part), rhs.bounds(part))
                else {
                    return Decided::Some;
                };
                let (all, none) = match op {
                    Operator::Gt => (a_lo > b_hi, a_hi <= b_lo),
                    Operator::Gte => (a_lo >= b_hi, a_hi < b_lo),
                    Operator::Lt => (a_hi < b_lo, a_lo >= b_hi),
                    Operator::Lte => (a_hi <= b_lo, a_lo > b_hi),
                    Operator::Eq => (
                        a_lo == a_hi && b_lo == b_hi && a_lo == b_lo,
                        a_hi < b_lo || b_hi < a_lo,
                    ),
                    Operator::NotEq => (
                        a_hi < b_lo || b_hi < a_lo,
                        a_lo == a_hi && b_lo == b_hi && a_lo == b_lo,
                    ),
                    _ => (false, false),
                };
                match (all, none) {
                    (true, _) => Decided::All,
                    (_, true) => Decided::None,
                    _ => Decided::Some,
                }
            }
            Predicate::And(l, r) => match (l.decide(part), r.decide(part)) {
                (Decided::None, _) | (_, Decided::None) => Decided::None,
                (Decided::All, Decided::All) => Decided::All,
                _ => Decided::Some,
            },
            Predicate::Or(l, r) => match (l.decide(part), r.decide(part)) {
                (Decided::All, _) | (_, Decided::All) => Decided::All,
                (Decided::None, Decided::None) => Decided::None,
                _ => Decided::Some,
            },
            Predicate::Unknown => Decided::Some,
        }
    }

    /// Whether one row, with these part values, satisfies the predicate. Only for a predicate
    /// parsed strictly, with no `Unknown` in it.
    #[cfg(test)]
    fn holds(&self, part: &impl Fn(usize) -> i128) -> bool {
        match self {
            Predicate::Compare(op, lhs, rhs) => {
                let (a, b) = (lhs.value(part), rhs.value(part));
                match op {
                    Operator::Gt => a > b,
                    Operator::Gte => a >= b,
                    Operator::Lt => a < b,
                    Operator::Lte => a <= b,
                    Operator::Eq => a == b,
                    Operator::NotEq => a != b,
                    _ => unreachable!("only comparisons are parsed"),
                }
            }
            Predicate::And(l, r) => l.holds(part) && r.holds(part),
            Predicate::Or(l, r) => l.holds(part) || r.holds(part),
            Predicate::Unknown => unreachable!("only strictly parsed predicates are evaluated"),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::dtype::{DecimalDType, Nullability, StructFields};
    use vortex::expr::{cast, checked_add, get_item, gt, gt_eq, lit, zip_expr, Expression};
    use vortex::scalar::{DecimalValue, Scalar};
    use vortex::scalar_fn::fns::between::BetweenOptions;
    use vortex::scalar_fn::ScalarFnVTableExt;

    use super::*;

    /// A table with an `ohlc` struct of the parts `names`.
    fn table_dtype(names: &'static [&'static str]) -> DType {
        let decimal = DecimalDType::new(16, 4);
        let parts = DType::Struct(
            StructFields::new(
                names.into(),
                vec![DType::Decimal(decimal, Nullability::NonNullable); names.len()],
            ),
            Nullability::NonNullable,
        );
        DType::Struct(
            StructFields::new(
                ["ticker", OHLC].into(),
                vec![DType::Utf8(Nullability::NonNullable), parts],
            ),
            Nullability::NonNullable,
        )
    }

    /// The part `name`, read through the table's `ohlc` struct.
    fn part(name: &str) -> Expression {
        get_item(name, get_item(OHLC, root()))
    }

    /// A literal at the parts' precision.
    fn x(v: i64) -> Expression {
        x_at(v, 16)
    }

    fn x_at(v: i64, precision: u8) -> Expression {
        let decimal = DecimalDType::new(precision, 4);
        lit(Scalar::decimal(
            DecimalValue::I64(v),
            decimal,
            Nullability::NonNullable,
        ))
    }

    /// `between(open, a, b)`, as Vortex 0.89's scan writes `open >= a AND open < b`, parses as the
    /// AND of its two comparisons, strict or not as its options say, and prunes like it.
    #[test]
    fn between_parses_as_two_comparisons() {
        let dtype = table_dtype(&["open", "body"]);
        for (lower_strict, upper_strict) in [
            (StrictComparison::NonStrict, StrictComparison::Strict),
            (StrictComparison::Strict, StrictComparison::NonStrict),
        ] {
            let options = BetweenOptions {
                lower_strict,
                upper_strict,
            };
            let expr: Expression =
                Between.new_expr(options.clone(), [part("open"), x(100), x(200)]);
            let predicate = Predicate::parse(&expr.bind(&dtype).unwrap(), false);
            let Predicate::And(above, below) = &predicate else {
                panic!("{predicate:?}");
            };
            let strict = |op| matches!(op, Operator::Gt | Operator::Lt);
            assert!(
                matches!(**above, Predicate::Compare(op, Term::Part(0), Term::Literal(100)) if strict(op) == (lower_strict == StrictComparison::Strict))
            );
            assert!(
                matches!(**below, Predicate::Compare(op, Term::Part(0), Term::Literal(200)) if strict(op) == (upper_strict == StrictComparison::Strict))
            );
            let bounds = |lo, hi| move |i: usize| (i == 0).then_some((lo, hi));
            assert_eq!(predicate.decide(&bounds(120, 180)), Decided::All);
            assert_eq!(predicate.decide(&bounds(0, 99)), Decided::None);
            assert_eq!(predicate.decide(&bounds(201, 300)), Decided::None);
            assert_eq!(predicate.decide(&bounds(150, 250)), Decided::Some);
        }
    }

    /// `zip(body >= 0, open + body, open) + upper > x`, as high's filter is built, parses as one
    /// comparison over a `zip`, and the zone bounds of body pick its branch.
    #[test]
    fn zip_parses_and_picks_its_branch() {
        let dtype = table_dtype(&["open", "body", "upper"]);
        // High built with `zip`: each operand cast to the precision its place needs, so both of
        // zip's branches have one type.
        let at = |name: &str, precision: u8| {
            let target = DType::Decimal(DecimalDType::new(precision, 4), Nullability::Nullable);
            cast(part(name), target)
        };
        let close = checked_add(at("open", 16), at("body", 16));
        let up = gt_eq(part("body"), x(0));
        let top = zip_expr(up, close, at("open", 17));
        let expr = gt(checked_add(top, at("upper", 17)), x_at(1000, 18));
        let predicate = Predicate::parse(&expr.bind(&dtype).unwrap(), false);
        let Predicate::Compare(Operator::Gt, Term::Add(top, _), Term::Literal(1000)) = &predicate
        else {
            panic!("{predicate:?}");
        };
        assert!(matches!(**top, Term::Zip(..)), "{top:?}");
        // open up to 900 and upper up to 50: only a body of more than 50 can lift high above
        // 1000, and only where the zone's bodies are positive does the close branch count.
        let zone = |body: (i128, i128)| move |i: usize| Some([(800, 900), body, (0, 50)][i]);
        assert_eq!(predicate.decide(&zone((-200, -10))), Decided::None);
        assert_eq!(predicate.decide(&zone((0, 40))), Decided::None);
        assert_eq!(predicate.decide(&zone((0, 100))), Decided::Some);
        assert_eq!(predicate.decide(&zone((-100, 100))), Decided::Some);
    }

    /// For every comparison, and AND / OR of two, over a grid of part bounds: "all" and "none"
    /// must agree with evaluating every pair of values within the bounds. The terms include a
    /// `zip` on the second part's sign, as the prices' filters have.
    #[test]
    fn decide_agrees_with_holds() {
        let ops = [
            Operator::Eq,
            Operator::NotEq,
            Operator::Gt,
            Operator::Gte,
            Operator::Lt,
            Operator::Lte,
        ];
        let plus_two = || Term::Add(Box::new(Term::Part(1)), Box::new(Term::Literal(2)));
        // Part 0 against part 1 plus 2, or against part 0 plus part 1 where part 1 isn't
        // negative and part 0 where it is (the larger of open and close).
        let compare = |op| Predicate::Compare(op, Term::Part(0), plus_two());
        let compare_zip = |op| {
            let mask = Predicate::Compare(Operator::Gte, Term::Part(1), Term::Literal(0));
            let sum = Term::Add(Box::new(Term::Part(0)), Box::new(Term::Part(1)));
            let zip = Term::Zip(Box::new(mask), Box::new(sum), Box::new(Term::Part(0)));
            Predicate::Compare(op, zip, Term::Literal(1))
        };
        let mut predicates: Vec<Predicate> = ops.iter().map(|&op| compare(op)).collect();
        predicates.extend(ops.iter().map(|&op| compare_zip(op)));
        for &a in &ops {
            for &b in &ops {
                predicates.push(Predicate::And(Box::new(compare(a)), Box::new(compare(b))));
                predicates.push(Predicate::Or(
                    Box::new(compare(a)),
                    Box::new(compare_zip(b)),
                ));
            }
        }
        let ranges: Vec<(i128, i128)> = (-3..=3)
            .flat_map(|lo| (lo..=3).map(move |hi| (lo, hi)))
            .collect();
        for predicate in &predicates {
            for &a in &ranges {
                for &b in &ranges {
                    let bounds = [a, b];
                    let decided = predicate.decide(&|i| Some(bounds[i]));
                    let mut seen = (false, false);
                    for x in a.0..=a.1 {
                        for y in b.0..=b.1 {
                            let values = [x, y];
                            match predicate.holds(&|i| values[i]) {
                                true => seen.0 = true,
                                false => seen.1 = true,
                            }
                        }
                    }
                    match decided {
                        Decided::All => assert!(!seen.1, "{predicate:?} on {a:?} {b:?}"),
                        Decided::None => assert!(!seen.0, "{predicate:?} on {a:?} {b:?}"),
                        Decided::Some => {}
                    }
                }
            }
        }
    }
}
