//! Reading the files from DataFusion as the source they were written from: a table over them
//! has the source's schema (a Parquet file's, say), and the scan rebuilds each file's planned
//! columns into it.
//!
//! - [`BarsAdapterFactory`] rewrites a scan's expressions from the table's columns to a file's:
//!   a price becomes an [`OhlcPrice`] over the [`OHLC`] struct, a rescaled decimal column a
//!   [`Rescaled`], and everything else is left to DataFusion's default adapter.
//! - [`BarsExpressionConverter`] decides what vortex-datafusion hands Vortex. A comparison of a
//!   price with a literal goes in as [`compare_price`], so Vortex filters on the parts, and one
//!   of a rescaled column with the literal at the column's stored scale. Everything else goes through vortex-datafusion's
//!   default converter. Rebuilding the columns is left to DataFusion ([`OhlcPrice`],
//!   [`Rescaled`]), which is quicker than Vortex's expressions for it.
//!
//! [`vortex_format`] is a format that reads the files this way, given a table with the source's
//! schema and a [`BarsAdapterFactory`].
//!
//! [`ZonePruner`] prunes a Vortex scan before it starts, so the scan doesn't start the zones its
//! filter rules out. A [`VortexExecFactory`] builds scans with it, as Atlas's factory of the same
//! name builds its Vortex scans, and a [`PrunedTable`] has one build its scans.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{Array, Decimal128Array, PrimitiveArray, RecordBatch};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{exec_err, Result, ScalarValue};
use datafusion::logical_expr::{ColumnarValue, Operator as DFOperator};
use datafusion::physical_expr::expressions::{BinaryExpr, CastExpr, Column, Literal};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use vortex::dtype::{DType, DecimalDType, Nullability, StructFields};
use vortex::expr::{and, binary as vortex_binary, get_item, or, root, Expression};
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_datafusion::convert::{DefaultExpressionConverter, ExpressionConverter};
use vortex_datafusion::VortexFormat;

use crate::plan::{OHLC, OHLC_PARTS, OHLC_PRICES};

mod prune;
mod table;
use crate::prices::{compare_price, decimal_literal};
pub use prune::{ZonePruner, DEFAULT_CACHE_BYTES};
pub use table::{reader_factory, PrunedTable, VortexExecFactory};

/// A [`VortexFormat`] on `session` that hands Vortex what [`BarsExpressionConverter`] converts.
pub fn vortex_format(session: VortexSession) -> VortexFormat {
    let converter = BarsExpressionConverter::new(session.clone());
    VortexFormat::new(session).with_expression_converter(Arc::new(converter))
}

// ---------------------------------------------------------------------------------------------
// Rebuilding the prices
// ---------------------------------------------------------------------------------------------

/// One of the four prices, rebuilt from the [`OHLC`] struct (its only child) into the table's
/// type, a `Decimal128` at a scale at least the parts'.
#[derive(Debug, Eq)]
pub struct OhlcPrice {
    /// The price's index in [`OHLC_PRICES`].
    price: usize,
    ohlc: Arc<dyn PhysicalExpr>,
    field: FieldRef,
}

impl OhlcPrice {
    /// The price the table's `field` holds, from `ohlc`; `None` unless `field` is one of
    /// [`OHLC_PRICES`] and a `Decimal128`.
    pub fn new(field: FieldRef, ohlc: Arc<dyn PhysicalExpr>) -> Option<Self> {
        let price = OHLC_PRICES.iter().position(|p| p == field.name())?;
        matches!(field.data_type(), DataType::Decimal128(..)).then_some(Self { price, ohlc, field })
    }

    pub fn name(&self) -> &'static str {
        OHLC_PRICES[self.price]
    }

    /// The parts the price is rebuilt from, in [`OHLC_PARTS`] order: open, then the body for
    /// any price but open, and the wick on its side for high and low.
    fn parts_needed(&self) -> [bool; 4] {
        let [open, high, low, _] = OHLC_PRICES.map(|p| p == self.name());
        [true, !open, high, low]
    }
}

impl PartialEq for OhlcPrice {
    fn eq(&self, other: &Self) -> bool {
        self.price == other.price && self.ohlc.eq(&other.ohlc) && self.field == other.field
    }
}

impl Hash for OhlcPrice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.price.hash(state);
        self.ohlc.hash(state);
        self.field.hash(state);
    }
}

impl fmt::Display for OhlcPrice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name(), self.ohlc)
    }
}

impl PhysicalExpr for OhlcPrice {
    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::clone(&self.field))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let ohlc = self.ohlc.evaluate(batch)?.into_array(batch.num_rows())?;
        let Some(ohlc) = ohlc.as_struct_opt() else {
            return exec_err!("{self}: {OHLC} is {}, not a struct", ohlc.data_type());
        };
        let DataType::Decimal128(precision, scale) = *self.field.data_type() else {
            unreachable!("checked in OhlcPrice::new");
        };
        let mut nulls = ohlc.nulls().cloned();
        let mut parts: [Option<&PrimitiveArray<Decimal128Type>>; 4] = [None; 4];
        let mut parts_scale = None;
        for ((name, part), needed) in OHLC_PARTS.iter().zip(&mut parts).zip(self.parts_needed()) {
            if !needed {
                continue;
            }
            let Some(column) = ohlc.column_by_name(name) else {
                return exec_err!("{self}: {OHLC} has no {name}");
            };
            let Some(values) = column.as_primitive_opt::<Decimal128Type>() else {
                return exec_err!("{self}: {name} is {}", column.data_type());
            };
            nulls = NullBuffer::union(nulls.as_ref(), values.nulls());
            parts_scale = Some(values.scale());
            *part = Some(values);
        }
        let shift = scale - parts_scale.expect("open is always needed");
        if shift < 0 {
            return exec_err!("{self}: the parts' scale is above the price's");
        }
        let factor = 10i64.pow(shift as u32);
        let [open, body, upper, lower] = parts.map(|part| part.map(PrimitiveArray::values));
        let open = open.expect("open is always needed");
        fn needed(part: Option<&ScalarBuffer<i128>>) -> std::slice::Iter<'_, i128> {
            part.expect("the price needs it").iter()
        }
        // The parts have OHLC_PRECISION digits and the price at most 18, so the sums and the
        // rescale fit an i64: a loop of its own for each price, in i64, vectorizes.
        let widen = |price: i64| (price * factor) as i128;
        let rebuilt: Vec<i128> = match self.name() {
            "open" => open.iter().map(|&o| widen(o as i64)).collect(),
            "close" => open
                .iter()
                .zip(needed(body))
                .map(|(&o, &b)| widen(o as i64 + b as i64))
                .collect(),
            "high" => open
                .iter()
                .zip(needed(body))
                .zip(needed(upper))
                .map(|((&o, &b), &u)| {
                    let (o, b, u) = (o as i64, b as i64, u as i64);
                    widen(o + b.max(0) + u)
                })
                .collect(),
            _ => open
                .iter()
                .zip(needed(body))
                .zip(needed(lower))
                .map(|((&o, &b), &l)| {
                    let (o, b, l) = (o as i64, b as i64, l as i64);
                    widen(o + b.min(0) - l)
                })
                .collect(),
        };
        let array = Decimal128Array::new(rebuilt.into(), nulls)
            .with_precision_and_scale(precision, scale)?;
        Ok(ColumnarValue::Array(Arc::new(array)))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.ohlc]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let Some(ohlc) = children.pop().filter(|_| children.is_empty()) else {
            return exec_err!("{self} has exactly one child");
        };
        Ok(Arc::new(Self {
            price: self.price,
            ohlc,
            field: Arc::clone(&self.field),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name())?;
        self.ohlc.fmt_sql(f)?;
        write!(f, ")")
    }
}

/// A decimal column a file stores rescaled (its only child), read back at the table's type, a
/// `Decimal128` at a larger scale: each unscaled value multiplied by `10^k`. A rescale only
/// dropped trailing zeros, so the values fit the table's precision, and unlike DataFusion's cast
/// this doesn't check them again.
#[derive(Debug, Eq)]
pub struct Rescaled {
    column: Arc<dyn PhysicalExpr>,
    field: FieldRef,
}

impl PartialEq for Rescaled {
    fn eq(&self, other: &Self) -> bool {
        self.column.eq(&other.column) && self.field == other.field
    }
}

impl Hash for Rescaled {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.column.hash(state);
        self.field.hash(state);
    }
}

impl fmt::Display for Rescaled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rescaled({}, {})", self.column, self.field.data_type())
    }
}

impl PhysicalExpr for Rescaled {
    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::clone(&self.field))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let column = self.column.evaluate(batch)?.into_array(batch.num_rows())?;
        let Some(values) = column.as_primitive_opt::<Decimal128Type>() else {
            return exec_err!("{self}: the column is {}", column.data_type());
        };
        let DataType::Decimal128(precision, scale) = *self.field.data_type() else {
            unreachable!("checked when adapting");
        };
        let shift = scale - values.scale();
        if shift < 0 {
            return exec_err!("{self}: the stored scale is above the table's");
        }
        let factor = 10i128.pow(shift as u32);
        let array = values
            .unary::<_, Decimal128Type>(|v| v * factor)
            .with_precision_and_scale(precision, scale)?;
        Ok(ColumnarValue::Array(Arc::new(array)))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.column]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let Some(column) = children.pop().filter(|_| children.is_empty()) else {
            return exec_err!("{self} has exactly one child");
        };
        Ok(Arc::new(Self {
            column,
            field: Arc::clone(&self.field),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.column.fmt_sql(f)
    }
}

// ---------------------------------------------------------------------------------------------
// Adapting a table's expressions to a file
// ---------------------------------------------------------------------------------------------

/// Adapts a table with the source's schema to each file: see the module docs.
#[derive(Debug, Default)]
pub struct BarsAdapterFactory;

impl PhysicalExprAdapterFactory for BarsAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        // The decimal columns the file keeps at a smaller scale, and the prices the table has
        // and the file keeps in its OHLC struct.
        let mut columns: HashMap<String, Arc<dyn PhysicalExpr>> = HashMap::new();
        for field in logical_file_schema.fields() {
            let Some((index, stored)) = physical_file_schema.column_with_name(field.name()) else {
                continue;
            };
            if let (DataType::Decimal128(_, scale), DataType::Decimal128(_, stored_scale)) =
                (field.data_type(), stored.data_type())
            {
                if stored_scale < scale {
                    let column = Arc::new(Column::new(field.name(), index));
                    let field = Arc::clone(field);
                    columns.insert(field.name().clone(), Arc::new(Rescaled { column, field }));
                }
            }
        }
        if let Some((index, field)) = physical_file_schema.column_with_name(OHLC) {
            if matches!(field.data_type(), DataType::Struct(_)) {
                let ohlc: Arc<dyn PhysicalExpr> = Arc::new(Column::new(OHLC, index));
                for price in OHLC_PRICES {
                    let Ok(field) = logical_file_schema.field_with_name(price) else {
                        continue;
                    };
                    let field = Arc::new(field.clone());
                    if physical_file_schema.column_with_name(price).is_none() {
                        if let Some(price) = OhlcPrice::new(field, Arc::clone(&ohlc)) {
                            columns.insert(price.name().to_string(), Arc::new(price));
                        }
                    }
                }
            }
        }
        let inner =
            DefaultPhysicalExprAdapterFactory.create(logical_file_schema, physical_file_schema)?;
        Ok(Arc::new(BarsAdapter { columns, inner }))
    }
}

#[derive(Debug)]
struct BarsAdapter {
    /// What each column the file stores another way is read as: an [`OhlcPrice`] or a
    /// [`Rescaled`].
    columns: HashMap<String, Arc<dyn PhysicalExpr>>,
    /// DataFusion's default, for every other column, and for the struct itself: a column the
    /// table doesn't have, which it takes from the file as it is.
    inner: Arc<dyn PhysicalExprAdapter>,
}

impl PhysicalExprAdapter for BarsAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        let expr = if self.columns.is_empty() {
            expr
        } else {
            expr.transform(|expr| {
                let read_as = expr
                    .downcast_ref::<Column>()
                    .and_then(|column| self.columns.get(column.name()));
                Ok(match read_as {
                    Some(read_as) => Transformed::yes(Arc::clone(read_as)),
                    None => Transformed::no(expr),
                })
            })
            .map(|t| t.data)?
        };
        // The default adapter casts the stored column under a `Rescaled` to the table's type
        // too, which hides the column from the converter: the `Rescaled` does that itself.
        self.inner
            .rewrite(expr)?
            .transform(|expr| {
                let Some(rescaled) = expr.downcast_ref::<Rescaled>() else {
                    return Ok(Transformed::no(expr));
                };
                let Some(cast) = rescaled.column.downcast_ref::<CastExpr>() else {
                    return Ok(Transformed::no(expr));
                };
                Ok(Transformed::yes(Arc::new(Rescaled {
                    column: Arc::clone(cast.expr()),
                    field: Arc::clone(&rescaled.field),
                })))
            })
            .map(|t| t.data)
    }
}

// ---------------------------------------------------------------------------------------------
// Handing filters to Vortex
// ---------------------------------------------------------------------------------------------

/// Converts expressions for vortex-datafusion to hand Vortex: see the module docs.
pub struct BarsExpressionConverter {
    inner: DefaultExpressionConverter,
}

impl BarsExpressionConverter {
    pub fn new(session: VortexSession) -> Self {
        Self {
            inner: DefaultExpressionConverter::new(session),
        }
    }

    /// `expr`, which reads an [`OhlcPrice`] or a [`Rescaled`], as a Vortex filter: a comparison
    /// of one with a literal, or an `AND` or `OR` of what can be converted; `None` for anything
    /// else.
    fn convert_stored(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Result<Option<Expression>> {
        let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
            return Ok(None);
        };
        let (left, right) = (binary.left(), binary.right());
        if let Some(combine) = match binary.op() {
            DFOperator::And => Some(and as fn(Expression, Expression) -> Expression),
            DFOperator::Or => Some(or as fn(Expression, Expression) -> Expression),
            _ => None,
        } {
            let (Some(left), Some(right)) = (
                self.try_convert(left, schema)?,
                self.try_convert(right, schema)?,
            ) else {
                return Ok(None);
            };
            return Ok(Some(combine(left, right)));
        }
        let Some(op) = comparison(binary.op()) else {
            return Ok(None);
        };
        // The column on the left.
        let (column, literal, op) = match right.downcast_ref::<Literal>() {
            Some(literal) => (left, literal, op),
            None => match left.downcast_ref::<Literal>() {
                Some(literal) => (right, literal, flipped(op)),
                None => return Ok(None),
            },
        };
        let ScalarValue::Decimal128(Some(unscaled), _, scale) = *literal.value() else {
            return Ok(None);
        };
        // A rescaled column is compared as it is stored, with the literal at its scale.
        if let Some(rescaled) = column.downcast_ref::<Rescaled>() {
            let Some(stored) = rescaled.column.downcast_ref::<Column>() else {
                return Ok(None);
            };
            let DataType::Decimal128(precision, stored_scale) =
                *schema.field_with_name(stored.name())?.data_type()
            else {
                return Ok(None);
            };
            let decimal = DecimalDType::new(precision, stored_scale);
            let Some(literal) = decimal_literal(unscaled, scale, decimal) else {
                return Ok(None);
            };
            let stored = get_item(stored.name(), root());
            return Ok(Some(vortex_binary(op, stored, literal)));
        }
        let Some(price) = column.downcast_ref::<OhlcPrice>() else {
            return Ok(None);
        };
        let Some(ohlc) = price.ohlc.downcast_ref::<Column>() else {
            return Ok(None);
        };
        let Some(table) = ohlc_table(schema.field_with_name(ohlc.name())?) else {
            return Ok(None);
        };
        Ok(compare_price(&table, price.name(), op, unscaled, scale))
    }
}

impl ExpressionConverter for BarsExpressionConverter {
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Result<Option<Expression>> {
        if reads_stored(expr) {
            self.convert_stored(expr, schema)
        } else {
            self.inner.try_convert(expr, schema)
        }
    }
}

/// Whether `expr` reads a column the file stores another way: an [`OhlcPrice`] or a
/// [`Rescaled`].
fn reads_stored(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let mut found = false;
    let _ = expr.apply(|expr| {
        found =
            expr.downcast_ref::<OhlcPrice>().is_some() || expr.downcast_ref::<Rescaled>().is_some();
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    });
    found
}

/// DataFusion's comparison operators as Vortex's.
fn comparison(op: &DFOperator) -> Option<Operator> {
    Some(match op {
        DFOperator::Eq => Operator::Eq,
        DFOperator::NotEq => Operator::NotEq,
        DFOperator::Lt => Operator::Lt,
        DFOperator::LtEq => Operator::Lte,
        DFOperator::Gt => Operator::Gt,
        DFOperator::GtEq => Operator::Gte,
        _ => return None,
    })
}

/// `op` with its operands swapped: `x op y` is `y flipped(op) x`.
fn flipped(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::Lte => Operator::Gte,
        Operator::Gt => Operator::Lt,
        Operator::Gte => Operator::Lte,
        op => op,
    }
}

/// A table holding just the [`OHLC`] struct `field` (its Arrow field in the file's schema), as
/// [`compare_price`] takes it; `None` unless it is a struct of decimals.
fn ohlc_table(field: &Field) -> Option<DType> {
    let DataType::Struct(parts) = field.data_type() else {
        return None;
    };
    let dtypes = parts
        .iter()
        .map(|part| match part.data_type() {
            DataType::Decimal128(precision, scale) => Some(DType::Decimal(
                DecimalDType::new(*precision, *scale),
                part.is_nullable().into(),
            )),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let names: Vec<&str> = parts.iter().map(|part| part.name().as_str()).collect();
    let ohlc = DType::Struct(
        StructFields::new(names.into(), dtypes),
        field.is_nullable().into(),
    );
    Some(DType::Struct(
        StructFields::new([OHLC].into(), vec![ohlc]),
        Nullability::NonNullable,
    ))
}
