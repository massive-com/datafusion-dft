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

//! CQL `SELECT` generation: identifier/literal quoting and conversion of
//! DataFusion filter expressions into CQL predicates.
//!
//! Cassandra only allows efficient `WHERE` clauses that constrain the full
//! partition key, so filter pushdown is limited to equality predicates that
//! together cover every partition key column. Everything else is left for
//! DataFusion to evaluate.

use std::collections::HashMap;

use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;
use scylla::frame::response::result::{ColumnType, NativeType};

/// Builds a CQL `SELECT` with an explicit column list (never `*`, so row
/// column order always matches the projected Arrow schema)
pub(crate) fn build_select(
    keyspace: &str,
    table: &str,
    columns: &[String],
    predicates: &[String],
    limit: Option<usize>,
) -> String {
    let column_list = columns
        .iter()
        .map(|c| quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    let mut query = format!(
        "SELECT {column_list} FROM {}.{}",
        quote_ident(keyspace),
        quote_ident(table)
    );
    if !predicates.is_empty() {
        query.push_str(" WHERE ");
        query.push_str(&predicates.join(" AND "));
    }
    if let Some(limit) = limit {
        query.push_str(&format!(" LIMIT {limit}"));
    }
    query
}

/// Quotes a CQL identifier, preserving case and escaping embedded quotes
pub(crate) fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

fn quote_string(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Converts the filters that are equality predicates on partition key
/// columns into CQL predicates, in partition key order. Returns `None`
/// unless every partition key column is constrained, because Cassandra
/// rejects partially constrained partition keys (without `ALLOW FILTERING`).
pub(crate) fn partition_key_predicates(
    filters: &[&Expr],
    partition_key: &[String],
    column_types: &HashMap<String, ColumnType<'static>>,
) -> Option<Vec<String>> {
    let mut by_column: HashMap<&str, String> = HashMap::new();
    for filter in filters {
        if let Some((column, predicate)) = filter_to_predicate(filter, partition_key, column_types)
        {
            // A second predicate on the same column stays with DataFusion
            by_column.entry(column).or_insert(predicate);
        }
    }
    partition_key
        .iter()
        .map(|column| by_column.get(column.as_str()).cloned())
        .collect()
}

/// Converts `col = literal` (or `literal = col`) on a partition key column
/// into a CQL predicate, returning the column name and the predicate
pub(crate) fn filter_to_predicate<'a>(
    filter: &'a Expr,
    partition_key: &[String],
    column_types: &HashMap<String, ColumnType<'static>>,
) -> Option<(&'a str, String)> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = filter else {
        return None;
    };
    if *op != Operator::Eq {
        return None;
    }
    let (column, literal) = match (left.as_ref(), right.as_ref()) {
        (Expr::Column(c), Expr::Literal(v, _)) => (c, v),
        (Expr::Literal(v, _), Expr::Column(c)) => (c, v),
        _ => return None,
    };
    if !partition_key.iter().any(|pk| pk == &column.name) {
        return None;
    }
    let ColumnType::Native(native) = column_types.get(&column.name)? else {
        return None;
    };
    let rendered = literal_to_cql(literal, native)?;
    Some((
        column.name.as_str(),
        format!("{} = {rendered}", quote_ident(&column.name)),
    ))
}

/// Renders a DataFusion literal as a CQL literal, only for combinations
/// where the CQL text form unambiguously matches the column type
fn literal_to_cql(value: &ScalarValue, column_type: &NativeType) -> Option<String> {
    match (column_type, value) {
        (
            NativeType::Ascii | NativeType::Text,
            ScalarValue::Utf8(Some(s))
            | ScalarValue::LargeUtf8(Some(s))
            | ScalarValue::Utf8View(Some(s)),
        ) => Some(quote_string(s)),
        (NativeType::Boolean, ScalarValue::Boolean(Some(b))) => Some(b.to_string()),
        (NativeType::TinyInt | NativeType::SmallInt | NativeType::Int | NativeType::BigInt, _) => {
            integer_literal(value)
        }
        _ => None,
    }
}

fn integer_literal(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Int8(Some(i)) => Some(i.to_string()),
        ScalarValue::Int16(Some(i)) => Some(i.to_string()),
        ScalarValue::Int32(Some(i)) => Some(i.to_string()),
        ScalarValue::Int64(Some(i)) => Some(i.to_string()),
        ScalarValue::UInt8(Some(i)) => Some(i.to_string()),
        ScalarValue::UInt16(Some(i)) => Some(i.to_string()),
        ScalarValue::UInt32(Some(i)) => Some(i.to_string()),
        ScalarValue::UInt64(Some(i)) => Some(i.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::{col, lit};

    fn text_pk_types() -> HashMap<String, ColumnType<'static>> {
        HashMap::from([
            ("id".to_string(), ColumnType::Native(NativeType::Int)),
            ("name".to_string(), ColumnType::Native(NativeType::Text)),
        ])
    }

    #[test]
    fn test_build_select_basic() {
        let query = build_select(
            "ks",
            "users",
            &["id".to_string(), "name".to_string()],
            &[],
            None,
        );
        assert_eq!(query, r#"SELECT "id", "name" FROM "ks"."users""#);
    }

    #[test]
    fn test_build_select_with_predicates_and_limit() {
        let query = build_select(
            "ks",
            "users",
            &["id".to_string()],
            &[r#""id" = 1"#.to_string()],
            Some(10),
        );
        assert_eq!(
            query,
            r#"SELECT "id" FROM "ks"."users" WHERE "id" = 1 LIMIT 10"#
        );
    }

    #[test]
    fn test_quote_ident_escapes() {
        assert_eq!(quote_ident(r#"we"ird"#), r#""we""ird""#);
    }

    #[test]
    fn test_string_literal_escapes() {
        assert_eq!(quote_string("o'brien"), "'o''brien'");
    }

    #[test]
    fn test_filter_to_predicate_eq() {
        let pk = vec!["id".to_string()];
        let types = text_pk_types();
        let filter = col("id").eq(lit(5i32));
        let (column, predicate) = filter_to_predicate(&filter, &pk, &types).unwrap();
        assert_eq!(column, "id");
        assert_eq!(predicate, r#""id" = 5"#);
    }

    #[test]
    fn test_filter_to_predicate_reversed_and_string() {
        let pk = vec!["name".to_string()];
        let types = text_pk_types();
        let filter = lit("o'brien").eq(col("name"));
        let (_, predicate) = filter_to_predicate(&filter, &pk, &types).unwrap();
        assert_eq!(predicate, r#""name" = 'o''brien'"#);
    }

    #[test]
    fn test_filter_to_predicate_rejects_non_pk_and_non_eq() {
        let pk = vec!["id".to_string()];
        let types = text_pk_types();
        assert!(filter_to_predicate(&col("other").eq(lit(1)), &pk, &types).is_none());
        assert!(filter_to_predicate(&col("id").gt(lit(1)), &pk, &types).is_none());
    }

    #[test]
    fn test_partition_key_predicates_full_coverage() {
        let pk = vec!["id".to_string(), "name".to_string()];
        let types = text_pk_types();
        let f1 = col("id").eq(lit(1i32));
        let f2 = col("name").eq(lit("alice"));
        let predicates = partition_key_predicates(&[&f1, &f2], &pk, &types).unwrap();
        assert_eq!(predicates, vec![r#""id" = 1"#, r#""name" = 'alice'"#]);
    }

    #[test]
    fn test_partition_key_predicates_partial_coverage() {
        let pk = vec!["id".to_string(), "name".to_string()];
        let types = text_pk_types();
        let f1 = col("id").eq(lit(1i32));
        assert!(partition_key_predicates(&[&f1], &pk, &types).is_none());
    }
}
