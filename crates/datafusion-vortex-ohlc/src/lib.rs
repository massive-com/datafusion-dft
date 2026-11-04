// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Reading the files from DataFusion as the table they were written from. The bars layout gives a
//! file the source's prices ([`vortex_ohlc::bars`]); what's left here is the rescaled decimal
//! columns, which a file keeps at a smaller scale until the layout takes them over too. A table
//! may declare the source's own schema (a Parquet file's, say): the scan then reads each rescaled
//! column back at the source's scale.
//!
//! - [`BarsAdapterFactory`] rewrites a scan's expressions from the table's columns to a file's:
//!   a rescaled decimal column becomes a [`Rescaled`], and everything else is left to
//!   DataFusion's default adapter.
//! - [`BarsExpressionConverter`] decides what vortex-datafusion hands Vortex: a comparison of a
//!   rescaled column with a literal goes in with the literal at the column's stored scale, and
//!   everything else through vortex-datafusion's default converter.
//!
//! [`vortex_format_factory`] makes formats that read the files this way, given a table with a
//! [`BarsAdapterFactory`]. [`source_table`] makes that table from a listing table over the files.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, FieldRef, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{exec_err, Result, ScalarValue};
use datafusion::logical_expr::{ColumnarValue, Operator as DFOperator};
use datafusion::physical_expr::expressions::{BinaryExpr, CastExpr, Column, Literal};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use vortex::dtype::{DecimalDType, Nullability};
use vortex::expr::{and, binary as vortex_binary, get_item, lit, or, root, Expression};
use vortex::scalar::{DecimalValue, Scalar};
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_datafusion::convert::{DefaultExpressionConverter, ExpressionConverter};
use vortex_datafusion::VortexFormatFactory;

mod source;
pub use source::source_table;

/// A [`VortexFormatFactory`] on `session`, with vortex-ohlc registered on it
/// ([`vortex_ohlc::register`]), whose formats hand Vortex what [`BarsExpressionConverter`]
/// converts; their options are the DataFusion session's `vortex` ones, and
/// [`default`](datafusion::datasource::file_format::FileFormatFactory::default)'s Vortex's.
pub fn vortex_format_factory(session: VortexSession) -> VortexFormatFactory {
    vortex_ohlc::register(&session);
    let converter = BarsExpressionConverter::new(session.clone());
    VortexFormatFactory::new_with_session(session).with_expression_converter(Arc::new(converter))
}

// ---------------------------------------------------------------------------------------------
// Reading rescaled columns
// ---------------------------------------------------------------------------------------------

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
        // The decimal columns the file keeps at a smaller scale.
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
        let inner =
            DefaultPhysicalExprAdapterFactory.create(logical_file_schema, physical_file_schema)?;
        Ok(Arc::new(BarsAdapter { columns, inner }))
    }
}

#[derive(Debug)]
struct BarsAdapter {
    /// The columns the file keeps rescaled, each as its [`Rescaled`].
    columns: HashMap<String, Arc<dyn PhysicalExpr>>,
    /// DataFusion's default, for every other column.
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

    /// `expr`, which reads a [`Rescaled`], as a Vortex filter: a comparison of one with a
    /// literal, or an `AND` or `OR` of what can be converted; `None` for anything else.
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
                Some(literal) => match op.swap() {
                    Some(op) => (right, literal, op),
                    None => return Ok(None),
                },
                None => return Ok(None),
            },
        };
        let ScalarValue::Decimal128(Some(unscaled), _, scale) = *literal.value() else {
            return Ok(None);
        };
        // A rescaled column is compared as it is stored, with the literal at its scale.
        let Some(rescaled) = column.downcast_ref::<Rescaled>() else {
            return Ok(None);
        };
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
        Ok(Some(vortex_binary(op, stored, literal)))
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

/// Whether `expr` reads a column the file keeps rescaled.
fn reads_stored(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let mut found = false;
    let _ = expr.apply(|expr| {
        found = expr.downcast_ref::<Rescaled>().is_some();
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

/// The decimal `unscaled / 10^scale` as a literal of type `decimal`, or `None` if that type
/// can't hold it exactly.
fn decimal_literal(unscaled: i128, scale: i8, decimal: DecimalDType) -> Option<Expression> {
    let shift = decimal.scale() as i32 - scale as i32;
    let factor = 10i128.checked_pow(shift.unsigned_abs())?;
    let value = if shift >= 0 {
        unscaled.checked_mul(factor)?
    } else {
        (unscaled % factor == 0).then(|| unscaled / factor)?
    };
    let value = DecimalValue::I128(value);
    value
        .fits_in_precision(decimal)
        .then(|| lit(Scalar::decimal(value, decimal, Nullability::NonNullable)))
}
