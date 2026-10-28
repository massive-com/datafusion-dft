//! Reading the Vortex file into Arrow batches in the source's types, its plan undone, by one of
//! two [`ReadPath`]s:
//!
//! - [`ReadPath::Restore`], `decimal_parts_bench`'s: Vortex decodes the stored columns, then the
//!   prices are rebuilt from the OHLC parts, block by block, and rescaled columns scaled back up,
//!   in Rust.
//! - [`ReadPath::Expr`]: the scan's projection rebuilds every column as a Vortex expression over
//!   the stored ones, cast to the source's type: what a DataFusion scan does once its
//!   expressions are rewritten for the file.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::{ArrayRef as ArrowArrayRef, Decimal128Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use num_traits::PrimInt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::{DecimalArray, PrimitiveArray, StructArray};
use vortex::array::{ArrayRef, ExecutionCtx, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::Buffer;
use vortex::dtype::{DType, DecimalDType, DecimalType, Nullability};
use vortex::expr::{cast, get_item, pack, root, select, BoundExpression, Expression};
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::metrics::{DefaultMetricsRegistry, MetricValue, MetricsRegistry};
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::ohlc::ohlc_from_parts;
use vortex_ohlc::plan::{
    plan_from_json, stored_name, Plan, Transform, OHLC, OHLC_PARTS, OHLC_PRICES, TRANSFORMS_KEY,
};
use vortex_ohlc::prices::price;

use crate::common::needle::{filter_rows, vortex_filter, Needle};

/// Values per FastLanes block: [`restore`] rebuilds the prices this many at a time.
const BLOCK: usize = 1024;

/// How the plan is undone: see the module docs.
#[derive(Clone, Copy, Debug)]
pub enum ReadPath {
    Restore,
    Expr,
}

impl ReadPath {
    pub const ALL: [ReadPath; 2] = [ReadPath::Restore, ReadPath::Expr];

    pub fn name(self) -> &'static str {
        match self {
            ReadPath::Restore => "vortex restore",
            ReadPath::Expr => "vortex expr",
        }
    }
}

/// Which rows a Vortex read returns.
#[derive(Clone, Copy)]
pub enum Lookup<'a> {
    /// Every row.
    All,
    /// The rows matching a needle, as one filtered scan.
    Filter(&'a Needle),
    /// The rows matching a needle, in two scans: the filter with nothing but the row index
    /// projected, then every column at just those rows. A filtered scan sets up every projected
    /// column for each split before it knows whether the split has a match, and throws that work
    /// away when it doesn't; a row-index-only first pass avoids that.
    TwoPhase(&'a Needle),
}

/// A session with Vortex's defaults on `runtime`, which runs everything on the calling thread,
/// keeping Vortex as single-threaded as the Parquet reader.
pub fn new_session(runtime: &CurrentThreadRuntime) -> VortexSession {
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    session
}

pub fn open_vortex(
    runtime: &CurrentThreadRuntime,
    session: &VortexSession,
    path: &Path,
) -> VortexFile {
    runtime
        .block_on(session.open_options().open_path(path))
        .unwrap()
}

/// The plan the file at `path` records. Opening a file only loads its metadata segments when
/// asked to, which the timed opens aren't.
pub fn read_plan(runtime: &CurrentThreadRuntime, session: &VortexSession, path: &Path) -> Plan {
    let open = session.open_options().include_metadata().open_path(path);
    let file = runtime.block_on(open).unwrap();
    let json = file
        .metadata_segment(TRANSFORMS_KEY)
        .expect("the file records its plan");
    plan_from_json(&serde_json::from_slice(json.as_ref()).unwrap()).expect("a plan of this version")
}

/// What the Vortex engine counted while `lookup` ran once by `path` against the file at
/// `file_path`, opened on a session of its own: the file's I/O. Counters are summed by name;
/// returns the row count too.
#[allow(clippy::too_many_arguments)]
pub fn lookup_stats(
    runtime: &CurrentThreadRuntime,
    file_path: &Path,
    plan: &Plan,
    source: &SchemaRef,
    columns: Option<&[&str]>,
    path: ReadPath,
    lookup: Lookup<'_>,
) -> (usize, BTreeMap<String, u64>) {
    let session = new_session(runtime);
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let open = session
        .open_options()
        .with_metrics_registry(Arc::clone(&registry) as Arc<dyn MetricsRegistry>)
        .open_path(file_path);
    let file = runtime.block_on(open).unwrap();
    let batches = vortex_batches(
        runtime, &session, &file, plan, source, columns, path, lookup,
    );
    let rows = batches.map(|batch| batch.num_rows()).sum();
    let mut counters = BTreeMap::new();
    for metric in registry.snapshot() {
        if let MetricValue::Counter(counter) = metric.value() {
            *counters.entry(metric.name().to_string()).or_default() += counter.value();
        }
    }
    (rows, counters)
}

/// The rows of `file` matching `filter`, ascending: a filtered scan with only the row index
/// projected (the two-phase lookup's first pass).
pub fn matching_rows(
    runtime: &CurrentThreadRuntime,
    session: &VortexSession,
    file: &VortexFile,
    filter: &BoundExpression,
) -> Vec<u64> {
    let mut ctx = session.create_execution_ctx();
    let hits = file
        .scan()
        .unwrap()
        .with_filter(filter.clone())
        .with_projection(row_idx().bind(file.dtype()).unwrap())
        .into_array_iter(runtime)
        .unwrap();
    let mut indices: Vec<u64> = Vec::new();
    for chunk in hits {
        let chunk = chunk.unwrap().execute::<PrimitiveArray>(&mut ctx).unwrap();
        indices.extend_from_slice(chunk.as_slice::<u64>());
    }
    indices
}

/// Read `file` (optionally just `columns`, and just the rows `lookup` picks) by `path`, into
/// Arrow in the types of `source` (the Parquet scan's schema), undoing `plan`.
#[allow(clippy::too_many_arguments)]
pub fn vortex_batches<'a>(
    runtime: &'a CurrentThreadRuntime,
    session: &VortexSession,
    file: &VortexFile,
    plan: &Plan,
    source: &SchemaRef,
    columns: Option<&[&str]>,
    path: ReadPath,
    lookup: Lookup<'_>,
) -> impl Iterator<Item = RecordBatch> + 'a {
    // Terms whose literal the stored type can't hold are checked after decoding.
    let (pushed, after) = match lookup {
        Lookup::All => (None, Vec::new()),
        Lookup::Filter(needle) | Lookup::TwoPhase(needle) => vortex_filter(file, needle, plan),
    };
    assert!(
        after.is_empty() || columns.is_none(),
        "terms checked after decoding need their columns"
    );
    let mut scan = file.scan().unwrap();
    match (lookup, &pushed) {
        (Lookup::TwoPhase(_), Some(filter)) => {
            let indices = matching_rows(runtime, session, file, filter);
            let indices = StrictSortedBuffer::try_new(Buffer::from(indices)).unwrap();
            scan = scan.with_row_indices(indices);
        }
        (_, Some(filter)) => scan = scan.with_filter(filter.clone()),
        (_, None) => {}
    }
    let wanted: Vec<String> = source
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|name| columns.is_none_or(|columns| columns.contains(&name.as_str())))
        .collect();
    let session = session.clone();
    let plan = plan.clone();
    let source = Arc::clone(source);
    let mut ctx = session.create_execution_ctx();
    let decode: Box<dyn FnMut(ArrayRef) -> RecordBatch> = match path {
        ReadPath::Restore => {
            if let Some(columns) = columns {
                scan = scan.with_projection(stored_projection(file, &plan, columns));
            }
            Box::new(move |chunk| restore_chunk(chunk, &plan, &source, &wanted, &session, &mut ctx))
        }
        ReadPath::Expr => {
            scan = scan.with_projection(expr_projection(file, &plan, &source, &wanted));
            let fields: Vec<Field> = wanted
                .iter()
                .map(|name| source.field_with_name(name).unwrap().clone())
                .collect();
            let target = Field::new("", DataType::Struct(fields.into()), false);
            Box::new(move |chunk| {
                let array = session
                    .arrow()
                    .execute_arrow(chunk, Some(&target), &mut ctx)
                    .unwrap();
                RecordBatch::from(array.as_struct().clone())
            })
        }
    };
    scan_chunks(scan, runtime)
        .map(decode)
        .map(move |batch| filter_rows(batch, &after))
}

fn scan_chunks<'a>(
    scan: ScanBuilder<ArrayRef>,
    runtime: &'a CurrentThreadRuntime,
) -> impl Iterator<Item = ArrayRef> + 'a {
    scan.into_array_iter(runtime)
        .unwrap()
        .map(|chunk| chunk.unwrap())
}

// ---------------------------------------------------------------------------------------------
// Restore
// ---------------------------------------------------------------------------------------------

/// The stored columns that hold `columns`, with the [`OHLC`] struct cut down to the parts the
/// prices asked for are rebuilt from: the struct's reader then reads the other parts' segments
/// not at all.
fn stored_projection(file: &VortexFile, plan: &Plan, columns: &[&str]) -> BoundExpression {
    let mut stored: Vec<&str> = Vec::new();
    for &column in columns {
        let name = stored_name(plan, column);
        if !stored.contains(&name) {
            stored.push(name);
        }
    }
    let needed = ohlc_parts_needed(|price| columns.contains(&price));
    let some_parts = stored.contains(&OHLC) && needed.contains(&false);
    let projection = if some_parts {
        let parts: Vec<&str> = OHLC_PARTS
            .into_iter()
            .zip(needed)
            .filter_map(|(p, n)| n.then_some(p))
            .collect();
        let field = |column: &str| match get_item(column, root()) {
            ohlc if column == OHLC => select(parts.clone(), ohlc),
            field => field,
        };
        pack(
            stored.iter().map(|&column| (column, field(column))),
            Nullability::NonNullable,
        )
    } else {
        select(stored, root())
    };
    projection.bind(file.dtype()).unwrap()
}

/// Which of [`OHLC_PARTS`] the prices for which `wanted` holds are rebuilt from: open always;
/// close needs body, high body and the upper wick, low body and the lower one.
fn ohlc_parts_needed(wanted: impl Fn(&str) -> bool) -> [bool; 4] {
    let [_, high, low, close] = OHLC_PRICES.map(wanted);
    [true, close || high || low, high, low]
}

/// One chunk of stored columns as Arrow in the source's types: the prices rebuilt from the
/// [`OHLC`] struct in its place, rescaled columns scaled back up, the rest as Vortex exports
/// them; only the `wanted` columns are kept.
fn restore_chunk(
    chunk: ArrayRef,
    plan: &Plan,
    source: &Schema,
    wanted: &[String],
    session: &VortexSession,
    ctx: &mut ExecutionCtx,
) -> RecordBatch {
    let table = chunk.execute::<StructArray>(ctx).unwrap();
    let is_wanted = |name: &str| wanted.iter().any(|w| w == name);
    let prices: Vec<&'static str> = OHLC_PRICES.into_iter().filter(|p| is_wanted(p)).collect();
    let mut out: Vec<(Field, ArrowArrayRef)> = Vec::with_capacity(wanted.len());
    for (idx, name) in table.names().iter().enumerate() {
        let column = table.unmasked_field(idx).clone();
        if name.as_ref() == OHLC {
            let Some(&Transform::Ohlc(k)) = plan.get(OHLC_PRICES[0]) else {
                panic!("{OHLC} column without an Ohlc plan");
            };
            out.extend(bars_to_arrow(column, k, &prices, source, ctx));
            continue;
        }
        let original = source.field_with_name(name.as_ref()).unwrap();
        let column = match plan.get(name.as_ref()) {
            Some(&Transform::Rescale(k)) => {
                rescaled_to_arrow(column, k, original.name(), original.data_type(), ctx)
            }
            Some(Transform::Ohlc(_)) => unreachable!("prices are read from {OHLC}"),
            None | Some(Transform::Packed) => {
                session.arrow().execute_arrow(column, None, ctx).unwrap()
            }
        };
        let field = Field::new(name.as_ref(), column.data_type().clone(), true);
        out.push((field, column));
    }
    out.retain(|(field, _)| is_wanted(field.name()));
    let (fields, columns): (Vec<Field>, Vec<ArrowArrayRef>) = out.into_iter().unzip();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Decode the OHLC struct straight into the `prices` asked for, each in its source type: the
/// parts they need decoded whole, then put back together a block at a time (see [`restore`]).
fn bars_to_arrow(
    ohlc: ArrayRef,
    k: u8,
    prices: &[&str],
    source: &Schema,
    ctx: &mut ExecutionCtx,
) -> Vec<(Field, ArrowArrayRef)> {
    let parts = ohlc.execute::<StructArray>(ctx).unwrap();
    let len = parts.len();
    let needed = ohlc_parts_needed(|price| prices.contains(&price));
    // The parts come out already multiplied by the prices' rescale (the prices are linear in
    // them), in i64: widening is then all that's left.
    let factor = 10i64.pow(k as u32);
    let mut decoded = OHLC_PARTS.iter().zip(needed).map(|(&want, needed)| {
        needed.then(|| {
            let idx = parts
                .names()
                .iter()
                .position(|n| n.as_ref() == want)
                .unwrap_or_else(|| panic!("{OHLC} has no {want} part"));
            decimal_values(parts.unmasked_field(idx).clone(), factor, OHLC, ctx)
        })
    });
    let decoded: [Option<Vec<i64>>; 4] = std::array::from_fn(|_| decoded.next().unwrap());
    let mut restored = restore(decoded, OHLC_PRICES.map(|p| prices.contains(&p)), len);
    prices
        .iter()
        .map(|&price| {
            let idx = OHLC_PRICES.iter().position(|&p| p == price).unwrap();
            let field = source.field_with_name(price).unwrap();
            let DataType::Decimal128(precision, scale) = *field.data_type() else {
                panic!("{price}: can't restore into {}", field.data_type());
            };
            assert!(
                precision <= 18,
                "{price}: a precision of {precision} may not fit an i64"
            );
            let values = std::mem::take(&mut restored[idx]);
            let array = Decimal128Array::from(values)
                .with_precision_and_scale(precision, scale)
                .unwrap();
            let column = Arc::new(array) as ArrowArrayRef;
            (Field::new(price, column.data_type().clone(), true), column)
        })
        .collect()
}

/// The prices in [`OHLC_PRICES`] order (empty where not `want`ed) from the parts that are there
/// (`None` for those nothing asked for needs), a block at a time, so the loop vectorizes and the
/// block is in cache while it is widened. With only open there, the prices aren't worked out at
/// all: open is its own part.
fn restore(parts: [Option<Vec<i64>>; 4], want: [bool; 4], len: usize) -> [Vec<i128>; 4] {
    let mut out = want.map(|want| Vec::with_capacity(if want { len } else { 0 }));
    let only_open = parts[1..].iter().all(Option::is_none);
    let mut blocks = [[0i64; BLOCK]; 4];
    for start in (0..len).step_by(BLOCK) {
        let n = BLOCK.min(len - start);
        for (part, block) in parts.iter().zip(blocks.iter_mut()) {
            if let Some(part) = part {
                block[..n].copy_from_slice(&part[start..start + n]);
            }
        }
        if !only_open {
            // A part nothing asked for holds whatever the last block left there, and only goes
            // into prices nothing asked for either.
            let [base, body, upper, lower] = &mut blocks;
            for i in 0..BLOCK {
                [base[i], body[i], upper[i], lower[i]] =
                    ohlc_from_parts([base[i], body[i], upper[i], lower[i]]);
            }
        }
        for ((out, block), want) in out.iter_mut().zip(&blocks).zip(want) {
            if want {
                out.extend(block[..n].iter().map(|&v| v as i128));
            }
        }
    }
    out
}

/// Decode a rescaled decimal column straight into its source type (`target`, a `Decimal128`),
/// multiplying by `10^k` on the way. Exporting to Arrow first and rescaling afterwards writes
/// every value twice.
fn rescaled_to_arrow(
    column: ArrayRef,
    k: u8,
    name: &str,
    target: &DataType,
    ctx: &mut ExecutionCtx,
) -> ArrowArrayRef {
    let DataType::Decimal128(precision, scale) = *target else {
        panic!("{name}: can't rescale into {target}");
    };
    let values = decimal_values(column, 10i128.pow(k as u32), name, ctx);
    Arc::new(
        Decimal128Array::from(values)
            .with_precision_and_scale(precision, scale)
            .unwrap(),
    )
}

/// Decode a decimal array and return its unscaled values widened to `W` and multiplied by
/// `factor`, in one pass.
fn decimal_values<W>(column: ArrayRef, factor: W, name: &str, ctx: &mut ExecutionCtx) -> Vec<W>
where
    W: PrimInt + TryFrom<i128> + From<i8> + From<i16> + From<i32> + From<i64>,
{
    fn widen<T: Copy + Into<W>, W: PrimInt>(values: &[T], factor: W) -> Vec<W> {
        values.iter().map(|&v| v.into() * factor).collect()
    }
    let decimal = column.execute::<DecimalArray>(ctx).unwrap();
    match decimal.values_type() {
        DecimalType::I8 => widen(&decimal.buffer::<i8>(), factor),
        DecimalType::I16 => widen(&decimal.buffer::<i16>(), factor),
        DecimalType::I32 => widen(&decimal.buffer::<i32>(), factor),
        DecimalType::I64 => widen(&decimal.buffer::<i64>(), factor),
        DecimalType::I128 => decimal
            .buffer::<i128>()
            .iter()
            .map(|&v| {
                let Ok(v) = W::try_from(v) else {
                    panic!("{name}: a value doesn't fit the scan type");
                };
                v * factor
            })
            .collect(),
        other => panic!("{name}: unexpected {other:?} values"),
    }
}

// ---------------------------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------------------------

/// The `wanted` source columns, in order, each as an expression over the stored columns cast to
/// its source type: a rescaled column cast back to its scale, a price rebuilt from the [`OHLC`]
/// parts (see [`price`]), and the rest as they are.
fn expr_projection(
    file: &VortexFile,
    plan: &Plan,
    source: &Schema,
    wanted: &[String],
) -> BoundExpression {
    let source_type = |name: &str| {
        let field = source.field_with_name(name).unwrap();
        let DataType::Decimal128(precision, scale) = *field.data_type() else {
            panic!("{name} is {}, not Decimal128", field.data_type());
        };
        DType::Decimal(DecimalDType::new(precision, scale), Nullability::Nullable)
    };
    let column = |name: &str| -> Expression {
        match plan.get(name) {
            Some(Transform::Rescale(_)) => cast(get_item(name, root()), source_type(name)),
            Some(Transform::Ohlc(_)) => cast(price(file.dtype(), name).unwrap(), source_type(name)),
            None | Some(Transform::Packed) => get_item(name, root()),
        }
    };
    pack(
        wanted.iter().map(|name| (name.as_str(), column(name))),
        Nullability::NonNullable,
    )
    .bind(file.dtype())
    .unwrap()
}
