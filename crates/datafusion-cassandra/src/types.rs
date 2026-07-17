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

//! Mapping between CQL types/values and Arrow types/arrays.
//!
//! Scalar CQL types map to their natural Arrow equivalents. Lists, sets and
//! vectors map to Arrow lists. Types without a natural Arrow equivalent
//! (map, tuple, UDT, duration, decimal, varint, uuid, inet) map to `Utf8`
//! with a readable rendering, so every Cassandra table remains queryable.

use std::sync::Arc;

use datafusion::arrow::{
    array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array, Int16Array,
        Int32Array, Int64Array, Int8Array, ListArray, RecordBatch, RecordBatchOptions, StringArray,
        Time64NanosecondArray, TimestampMillisecondArray,
    },
    buffer::{NullBuffer, OffsetBuffer},
    datatypes::{DataType, Field, SchemaRef, TimeUnit},
};
use datafusion::common::{internal_err, Result};
use scylla::frame::response::result::{CollectionType, ColumnType, NativeType};
use scylla::value::{CqlValue, Row};

/// Days between the CQL `date` epoch (-2^31 days before 1970-01-01) and the
/// Arrow `Date32` epoch (1970-01-01)
const CQL_DATE_EPOCH_OFFSET: i64 = 1 << 31;

/// Maps a CQL column type to the Arrow type used to represent it
pub(crate) fn cql_type_to_arrow(typ: &ColumnType) -> DataType {
    match typ {
        ColumnType::Native(native) => native_type_to_arrow(native),
        ColumnType::Collection {
            typ: CollectionType::List(inner) | CollectionType::Set(inner),
            ..
        } => list_of(inner),
        ColumnType::Vector { typ, .. } => list_of(typ),
        // Maps, tuples, UDTs and anything the driver adds later render as text
        _ => DataType::Utf8,
    }
}

fn list_of(inner: &ColumnType) -> DataType {
    DataType::List(Arc::new(Field::new("item", cql_type_to_arrow(inner), true)))
}

fn native_type_to_arrow(native: &NativeType) -> DataType {
    match native {
        NativeType::Ascii | NativeType::Text => DataType::Utf8,
        NativeType::Boolean => DataType::Boolean,
        NativeType::Blob => DataType::Binary,
        NativeType::Counter | NativeType::BigInt => DataType::Int64,
        NativeType::Date => DataType::Date32,
        NativeType::Double => DataType::Float64,
        NativeType::Float => DataType::Float32,
        NativeType::Int => DataType::Int32,
        NativeType::SmallInt => DataType::Int16,
        NativeType::TinyInt => DataType::Int8,
        NativeType::Timestamp => {
            DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("+00:00")))
        }
        NativeType::Time => DataType::Time64(TimeUnit::Nanosecond),
        // Uuid, Timeuuid, Inet, Decimal, Varint, Duration and future types
        _ => DataType::Utf8,
    }
}

/// Converts a chunk of CQL rows into a [`RecordBatch`] matching `schema`.
/// The rows' columns must be in the same order as the schema fields (the
/// generated CQL `SELECT` lists columns explicitly, guaranteeing this).
pub(crate) fn rows_to_batch(schema: &SchemaRef, rows: Vec<Row>) -> Result<RecordBatch> {
    let num_fields = schema.fields().len();
    let num_rows = rows.len();
    // A projection can be empty (e.g. `SELECT count(*)`): the CQL query still
    // selects one column but the batch carries only a row count
    if num_fields == 0 {
        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
        return Ok(RecordBatch::try_new_with_options(
            Arc::clone(schema),
            vec![],
            &options,
        )?);
    }
    let mut columns: Vec<Vec<Option<CqlValue>>> = (0..num_fields)
        .map(|_| Vec::with_capacity(num_rows))
        .collect();
    for row in rows {
        if row.columns.len() != num_fields {
            return internal_err!(
                "Cassandra row has {} columns but schema has {num_fields}",
                row.columns.len()
            );
        }
        for (i, value) in row.columns.into_iter().enumerate() {
            columns[i].push(value);
        }
    }
    let arrays = schema
        .fields()
        .iter()
        .zip(columns)
        .map(|(field, values)| build_column(field.data_type(), values))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(Arc::clone(schema), arrays)?)
}

/// Builds an Arrow array of `data_type` from CQL values. Values of an
/// unexpected variant (including the legacy `Empty`) become nulls rather
/// than errors, matching how other integrations tolerate loosely typed rows.
fn build_column(data_type: &DataType, values: Vec<Option<CqlValue>>) -> Result<ArrayRef> {
    match data_type {
        DataType::Boolean => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Boolean(b)) => Some(*b),
                    _ => None,
                })
                .collect::<BooleanArray>(),
        )),
        DataType::Int8 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::TinyInt(i)) => Some(*i),
                    _ => None,
                })
                .collect::<Int8Array>(),
        )),
        DataType::Int16 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::SmallInt(i)) => Some(*i),
                    _ => None,
                })
                .collect::<Int16Array>(),
        )),
        DataType::Int32 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Int(i)) => Some(*i),
                    _ => None,
                })
                .collect::<Int32Array>(),
        )),
        DataType::Int64 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::BigInt(i)) => Some(*i),
                    Some(CqlValue::Counter(c)) => Some(c.0),
                    _ => None,
                })
                .collect::<Int64Array>(),
        )),
        DataType::Float32 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Float(f)) => Some(*f),
                    _ => None,
                })
                .collect::<Float32Array>(),
        )),
        DataType::Float64 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Double(f)) => Some(*f),
                    _ => None,
                })
                .collect::<Float64Array>(),
        )),
        DataType::Date32 => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Date(d)) => Some((d.0 as i64 - CQL_DATE_EPOCH_OFFSET) as i32),
                    _ => None,
                })
                .collect::<Date32Array>(),
        )),
        DataType::Time64(TimeUnit::Nanosecond) => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Time(t)) => Some(t.0),
                    _ => None,
                })
                .collect::<Time64NanosecondArray>(),
        )),
        DataType::Timestamp(TimeUnit::Millisecond, tz) => Ok(Arc::new(
            values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Timestamp(t)) => Some(t.0),
                    _ => None,
                })
                .collect::<TimestampMillisecondArray>()
                .with_timezone_opt(tz.clone()),
        )),
        DataType::Binary => {
            let bytes: Vec<Option<&[u8]>> = values
                .iter()
                .map(|v| match v {
                    Some(CqlValue::Blob(b)) => Some(b.as_slice()),
                    _ => None,
                })
                .collect();
            Ok(Arc::new(BinaryArray::from_iter(bytes)))
        }
        DataType::Utf8 => Ok(Arc::new(
            values
                .iter()
                .map(|v| v.as_ref().map(render_string))
                .collect::<StringArray>(),
        )),
        DataType::List(field) => {
            let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
            offsets.push(0);
            let mut validity: Vec<bool> = Vec::with_capacity(values.len());
            let mut child_values: Vec<Option<CqlValue>> = Vec::new();
            let mut current: i32 = 0;
            for value in values {
                match value {
                    Some(
                        CqlValue::List(items) | CqlValue::Set(items) | CqlValue::Vector(items),
                    ) => {
                        current += items.len() as i32;
                        child_values.extend(items.into_iter().map(Some));
                        validity.push(true);
                    }
                    _ => validity.push(false),
                }
                offsets.push(current);
            }
            let child = build_column(field.data_type(), child_values)?;
            Ok(Arc::new(ListArray::new(
                Arc::clone(field),
                OffsetBuffer::new(offsets.into()),
                child,
                Some(NullBuffer::from(validity)),
            )))
        }
        other => internal_err!("Unsupported Arrow type for Cassandra column: {other}"),
    }
}

/// Renders any CQL value as a readable string, used for types mapped to `Utf8`
fn render_string(value: &CqlValue) -> String {
    match value {
        CqlValue::Ascii(s) | CqlValue::Text(s) => s.clone(),
        CqlValue::Boolean(b) => b.to_string(),
        CqlValue::TinyInt(i) => i.to_string(),
        CqlValue::SmallInt(i) => i.to_string(),
        CqlValue::Int(i) => i.to_string(),
        CqlValue::BigInt(i) => i.to_string(),
        CqlValue::Counter(c) => c.0.to_string(),
        CqlValue::Float(f) => f.to_string(),
        CqlValue::Double(f) => f.to_string(),
        CqlValue::Blob(b) => format!("0x{}", hex_encode(b)),
        CqlValue::Decimal(d) => {
            let (bytes, scale) = d.as_signed_be_bytes_slice_and_exponent();
            render_decimal(bytes, scale)
        }
        CqlValue::Varint(v) => match i128_from_signed_be_bytes(v.as_signed_bytes_be_slice()) {
            Some(i) => i.to_string(),
            None => format!("0x{}", hex_encode(v.as_signed_bytes_be_slice())),
        },
        CqlValue::Date(d) => (d.0 as i64 - CQL_DATE_EPOCH_OFFSET).to_string(),
        CqlValue::Time(t) => t.0.to_string(),
        CqlValue::Timestamp(t) => t.0.to_string(),
        CqlValue::Duration(d) => format!("{}mo{}d{}ns", d.months, d.days, d.nanoseconds),
        CqlValue::Inet(ip) => ip.to_string(),
        CqlValue::Uuid(u) => u.to_string(),
        CqlValue::Timeuuid(u) => u.to_string(),
        CqlValue::List(items) | CqlValue::Set(items) | CqlValue::Vector(items) => {
            let rendered: Vec<String> = items.iter().map(render_string).collect();
            format!("[{}]", rendered.join(", "))
        }
        CqlValue::Map(entries) => {
            let rendered: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", render_string(k), render_string(v)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
        CqlValue::Tuple(items) => {
            let rendered: Vec<String> = items
                .iter()
                .map(|v| match v {
                    Some(v) => render_string(v),
                    None => "null".to_string(),
                })
                .collect();
            format!("({})", rendered.join(", "))
        }
        CqlValue::UserDefinedType { fields, .. } => {
            let rendered: Vec<String> = fields
                .iter()
                .map(|(name, v)| match v {
                    Some(v) => format!("{name}: {}", render_string(v)),
                    None => format!("{name}: null"),
                })
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
        CqlValue::Empty => String::new(),
        other => format!("{other:?}"),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Interprets a big-endian signed integer of up to 16 bytes
fn i128_from_signed_be_bytes(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() {
        return Some(0);
    }
    if bytes.len() > 16 {
        return None;
    }
    let sign = if bytes[0] & 0x80 != 0 { 0xff } else { 0x00 };
    let mut buf = [sign; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(buf))
}

/// Renders a CQL decimal (unscaled big-endian integer + scale) as a decimal
/// string, e.g. bytes for 12345 with scale 2 -> "123.45"
fn render_decimal(bytes: &[u8], scale: i32) -> String {
    let Some(unscaled) = i128_from_signed_be_bytes(bytes) else {
        return format!("0x{}e-{scale}", hex_encode(bytes));
    };
    if scale <= 0 {
        let zeros = "0".repeat(scale.unsigned_abs() as usize);
        return format!("{unscaled}{zeros}");
    }
    let negative = unscaled < 0;
    let digits = unscaled.unsigned_abs().to_string();
    let scale = scale as usize;
    let (int_part, frac_part) = if digits.len() > scale {
        let (i, f) = digits.split_at(digits.len() - scale);
        (i.to_string(), f.to_string())
    } else {
        ("0".to_string(), format!("{digits:0>scale$}"))
    };
    format!("{}{int_part}.{frac_part}", if negative { "-" } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::Schema;
    use scylla::value::{CqlDate, CqlDecimal, CqlTimestamp};

    #[test]
    fn test_native_type_mapping() {
        assert_eq!(
            cql_type_to_arrow(&ColumnType::Native(NativeType::Text)),
            DataType::Utf8
        );
        assert_eq!(
            cql_type_to_arrow(&ColumnType::Native(NativeType::BigInt)),
            DataType::Int64
        );
        assert_eq!(
            cql_type_to_arrow(&ColumnType::Native(NativeType::Uuid)),
            DataType::Utf8
        );
        assert_eq!(
            cql_type_to_arrow(&ColumnType::Native(NativeType::Timestamp)),
            DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("+00:00")))
        );
    }

    #[test]
    fn test_list_type_mapping() {
        let typ = ColumnType::Collection {
            frozen: false,
            typ: CollectionType::List(Box::new(ColumnType::Native(NativeType::Int))),
        };
        assert_eq!(
            cql_type_to_arrow(&typ),
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
        );
    }

    #[test]
    fn test_map_type_renders_as_utf8() {
        let typ = ColumnType::Collection {
            frozen: false,
            typ: CollectionType::Map(
                Box::new(ColumnType::Native(NativeType::Text)),
                Box::new(ColumnType::Native(NativeType::Int)),
            ),
        };
        assert_eq!(cql_type_to_arrow(&typ), DataType::Utf8);
    }

    #[test]
    fn test_rows_to_batch_scalars() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let rows = vec![
            Row {
                columns: vec![
                    Some(CqlValue::Int(1)),
                    Some(CqlValue::Text("alice".to_string())),
                ],
            },
            Row {
                columns: vec![Some(CqlValue::Int(2)), None],
            },
        ];
        let batch = rows_to_batch(&schema, rows).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice");
        assert!(names.is_null(1));
    }

    #[test]
    fn test_rows_to_batch_empty_projection() {
        let schema = Arc::new(Schema::empty());
        let rows = vec![
            Row {
                columns: vec![Some(CqlValue::Int(1))],
            },
            Row {
                columns: vec![Some(CqlValue::Int(2))],
            },
        ];
        // Rows may still carry the placeholder column; only the count is used
        let batch = rows_to_batch(&schema, rows).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 0);
    }

    #[test]
    fn test_build_list_column() {
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let values = vec![
            Some(CqlValue::List(vec![CqlValue::Int(1), CqlValue::Int(2)])),
            None,
            Some(CqlValue::Set(vec![CqlValue::Int(3)])),
        ];
        let array = build_column(&DataType::List(field), values).unwrap();
        let list = array.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list.len(), 3);
        assert!(list.is_null(1));
        assert_eq!(list.value(0).len(), 2);
        assert_eq!(list.value(2).len(), 1);
    }

    #[test]
    fn test_timestamp_and_date_conversion() {
        let values = vec![Some(CqlValue::Timestamp(CqlTimestamp(1_000)))];
        let array = build_column(
            &DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("+00:00"))),
            values,
        )
        .unwrap();
        assert_eq!(array.len(), 1);

        // 2^31 days offset means the CQL value for 1970-01-01 is 2^31
        let values = vec![Some(CqlValue::Date(CqlDate(1 << 31)))];
        let array = build_column(&DataType::Date32, values).unwrap();
        let dates = array.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(dates.value(0), 0);
    }

    #[test]
    fn test_render_decimal() {
        let decimal = CqlDecimal::from_signed_be_bytes_and_exponent(vec![0x30, 0x39], 2);
        let (bytes, scale) = decimal.as_signed_be_bytes_slice_and_exponent();
        assert_eq!(render_decimal(bytes, scale), "123.45");
        assert_eq!(render_decimal(&[0x05], 3), "0.005");
        assert_eq!(render_decimal(&[0x05], -2), "500");
        assert_eq!(render_decimal(&[0xFB], 1), "-0.5");
    }

    #[test]
    fn test_render_string_collections() {
        let map = CqlValue::Map(vec![(CqlValue::Text("a".to_string()), CqlValue::Int(1))]);
        assert_eq!(render_string(&map), "{a: 1}");
        let tuple = CqlValue::Tuple(vec![Some(CqlValue::Int(1)), None]);
        assert_eq!(render_string(&tuple), "(1, null)");
    }
}
