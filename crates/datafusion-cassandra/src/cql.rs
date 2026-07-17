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
//! Cassandra only executes `WHERE` clauses (without `ALLOW FILTERING`) that
//! constrain the full partition key with equality, optionally followed by a
//! prefix of the clustering key: equality on leading clustering columns and
//! at most one range (lower/upper bound) on the last constrained one.
//! [`plan_pushdown`] translates exactly that shape; everything else is left
//! for DataFusion to evaluate after the scan.

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

/// The role a predicate plays in a CQL `WHERE` clause
#[derive(Clone, Copy, Debug, PartialEq)]
enum PredicateKind {
    Eq,
    /// `>` or `>=`
    LowerBound,
    /// `<` or `<=`
    UpperBound,
}

/// A filter translated to a CQL predicate on a single column
struct Candidate<'a> {
    column: &'a str,
    kind: PredicateKind,
    cql: String,
}

/// The CQL-pushable subset of a scan's filters
pub(crate) struct Pushdown {
    /// CQL predicates, in partition key then clustering key order
    pub predicates: Vec<String>,
    /// For each input filter, whether it is fully covered by `predicates`
    pub pushed: Vec<bool>,
}

/// Computes the largest CQL-executable `WHERE` clause from the filters:
/// equality on every partition key column, then equality on a prefix of the
/// clustering key, then at most one lower and one upper bound on the next
/// clustering column. Returns no predicates unless the full partition key is
/// covered, because Cassandra rejects partially constrained partition keys.
pub(crate) fn plan_pushdown(
    filters: &[&Expr],
    partition_key: &[String],
    clustering_key: &[String],
    column_types: &HashMap<String, ColumnType<'static>>,
) -> Pushdown {
    let candidates: Vec<Option<Candidate>> = filters
        .iter()
        .map(|f| filter_to_candidate(f, column_types))
        .collect();
    let mut pushed = vec![false; filters.len()];
    let mut predicates = Vec::new();

    // Find the first unused candidate on `column` of kind `kind`
    let find = |pushed: &[bool], column: &str, kind: PredicateKind| -> Option<usize> {
        candidates.iter().enumerate().position(|(i, c)| {
            !pushed[i]
                && c.as_ref()
                    .is_some_and(|c| c.column == column && c.kind == kind)
        })
    };

    // Every partition key column must have an equality predicate
    for pk_column in partition_key {
        match find(&pushed, pk_column, PredicateKind::Eq) {
            Some(i) => {
                predicates.push(candidates[i].as_ref().unwrap().cql.clone());
                pushed[i] = true;
            }
            None => {
                return Pushdown {
                    predicates: Vec::new(),
                    pushed: vec![false; filters.len()],
                }
            }
        }
    }

    // Clustering key prefix: equality continues the walk, a range (or no
    // constraint) ends it
    for ck_column in clustering_key {
        if let Some(i) = find(&pushed, ck_column, PredicateKind::Eq) {
            predicates.push(candidates[i].as_ref().unwrap().cql.clone());
            pushed[i] = true;
            continue;
        }
        if let Some(i) = find(&pushed, ck_column, PredicateKind::LowerBound) {
            predicates.push(candidates[i].as_ref().unwrap().cql.clone());
            pushed[i] = true;
        }
        if let Some(i) = find(&pushed, ck_column, PredicateKind::UpperBound) {
            predicates.push(candidates[i].as_ref().unwrap().cql.clone());
            pushed[i] = true;
        }
        break;
    }

    Pushdown { predicates, pushed }
}

/// Converts `col OP literal` (or `literal OP col`) into a CQL predicate
/// candidate, for operators and literal types CQL can execute
fn filter_to_candidate<'a>(
    filter: &'a Expr,
    column_types: &HashMap<String, ColumnType<'static>>,
) -> Option<Candidate<'a>> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = filter else {
        return None;
    };
    let (column, literal, op) = match (left.as_ref(), right.as_ref()) {
        (Expr::Column(c), Expr::Literal(v, _)) => (c, v, *op),
        // `literal OP column` is equivalent to `column swapped-OP literal`
        (Expr::Literal(v, _), Expr::Column(c)) => (c, v, op.swap()?),
        _ => return None,
    };
    let (kind, cql_op) = match op {
        Operator::Eq => (PredicateKind::Eq, "="),
        Operator::Gt => (PredicateKind::LowerBound, ">"),
        Operator::GtEq => (PredicateKind::LowerBound, ">="),
        Operator::Lt => (PredicateKind::UpperBound, "<"),
        Operator::LtEq => (PredicateKind::UpperBound, "<="),
        _ => return None,
    };
    let ColumnType::Native(native) = column_types.get(&column.name)? else {
        return None;
    };
    let rendered = literal_to_cql(literal, native)?;
    Some(Candidate {
        column: column.name.as_str(),
        kind,
        cql: format!("{} {cql_op} {rendered}", quote_ident(&column.name)),
    })
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
        // CQL accepts integer literals (milliseconds since epoch) for timestamps
        (NativeType::Timestamp, ScalarValue::TimestampMillisecond(Some(ms), _)) => {
            Some(ms.to_string())
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

    fn minute_aggs_types() -> HashMap<String, ColumnType<'static>> {
        HashMap::from([
            ("ticker".to_string(), ColumnType::Native(NativeType::Text)),
            (
                "timestamp_ms".to_string(),
                ColumnType::Native(NativeType::BigInt),
            ),
            ("seq".to_string(), ColumnType::Native(NativeType::Int)),
            ("close".to_string(), ColumnType::Native(NativeType::Double)),
        ])
    }

    fn pk() -> Vec<String> {
        vec!["ticker".to_string()]
    }

    fn ck() -> Vec<String> {
        vec!["timestamp_ms".to_string()]
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
    fn test_partition_key_eq_pushdown() {
        let f = col("ticker").eq(lit("C:EUR-QAR"));
        let plan = plan_pushdown(&[&f], &pk(), &ck(), &minute_aggs_types());
        assert_eq!(plan.predicates, vec![r#""ticker" = 'C:EUR-QAR'"#]);
        assert_eq!(plan.pushed, vec![true]);
    }

    #[test]
    fn test_reversed_literal_eq() {
        let f = lit("o'brien").eq(col("ticker"));
        let plan = plan_pushdown(&[&f], &pk(), &ck(), &minute_aggs_types());
        assert_eq!(plan.predicates, vec![r#""ticker" = 'o''brien'"#]);
    }

    #[test]
    fn test_clustering_range_pushdown() {
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = col("timestamp_ms").gt_eq(lit(1_000i64));
        let f3 = col("timestamp_ms").lt(lit(2_000i64));
        let plan = plan_pushdown(&[&f1, &f2, &f3], &pk(), &ck(), &minute_aggs_types());
        assert_eq!(
            plan.predicates,
            vec![
                r#""ticker" = 'AAPL'"#,
                r#""timestamp_ms" >= 1000"#,
                r#""timestamp_ms" < 2000"#,
            ]
        );
        assert_eq!(plan.pushed, vec![true, true, true]);
    }

    #[test]
    fn test_reversed_range_swaps_operator() {
        // 1000 <= timestamp_ms is a lower bound on the column
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = lit(1_000i64).lt_eq(col("timestamp_ms"));
        let plan = plan_pushdown(&[&f1, &f2], &pk(), &ck(), &minute_aggs_types());
        assert_eq!(
            plan.predicates,
            vec![r#""ticker" = 'AAPL'"#, r#""timestamp_ms" >= 1000"#]
        );
    }

    #[test]
    fn test_range_without_partition_key_not_pushed() {
        // Cassandra rejects clustering restrictions without the partition key
        let f = col("timestamp_ms").gt(lit(1_000i64));
        let plan = plan_pushdown(&[&f], &pk(), &ck(), &minute_aggs_types());
        assert!(plan.predicates.is_empty());
        assert_eq!(plan.pushed, vec![false]);
    }

    #[test]
    fn test_regular_column_filter_not_pushed() {
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = col("close").gt(lit(10.0f64));
        let plan = plan_pushdown(&[&f1, &f2], &pk(), &ck(), &minute_aggs_types());
        assert_eq!(plan.predicates, vec![r#""ticker" = 'AAPL'"#]);
        assert_eq!(plan.pushed, vec![true, false]);
    }

    #[test]
    fn test_clustering_prefix_gap_stops_walk() {
        // Range on the second clustering column without constraining the
        // first cannot be pushed
        let two_ck = vec!["timestamp_ms".to_string(), "seq".to_string()];
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = col("seq").gt(lit(5i32));
        let plan = plan_pushdown(&[&f1, &f2], &pk(), &two_ck, &minute_aggs_types());
        assert_eq!(plan.predicates, vec![r#""ticker" = 'AAPL'"#]);
        assert_eq!(plan.pushed, vec![true, false]);
    }

    #[test]
    fn test_clustering_eq_continues_walk() {
        let two_ck = vec!["timestamp_ms".to_string(), "seq".to_string()];
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = col("timestamp_ms").eq(lit(1_000i64));
        let f3 = col("seq").gt(lit(5i32));
        let plan = plan_pushdown(&[&f1, &f2, &f3], &pk(), &two_ck, &minute_aggs_types());
        assert_eq!(
            plan.predicates,
            vec![
                r#""ticker" = 'AAPL'"#,
                r#""timestamp_ms" = 1000"#,
                r#""seq" > 5"#,
            ]
        );
        assert_eq!(plan.pushed, vec![true, true, true]);
    }

    #[test]
    fn test_range_stops_walk_before_next_column() {
        // After a range on timestamp_ms, seq cannot be constrained in CQL
        let two_ck = vec!["timestamp_ms".to_string(), "seq".to_string()];
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = col("timestamp_ms").gt(lit(1_000i64));
        let f3 = col("seq").eq(lit(5i32));
        let plan = plan_pushdown(&[&f1, &f2, &f3], &pk(), &two_ck, &minute_aggs_types());
        assert_eq!(
            plan.predicates,
            vec![r#""ticker" = 'AAPL'"#, r#""timestamp_ms" > 1000"#]
        );
        assert_eq!(plan.pushed, vec![true, true, false]);
    }

    #[test]
    fn test_duplicate_bounds_only_one_pushed() {
        // Two lower bounds on the same column: CQL rejects duplicates, so the
        // second stays with DataFusion
        let f1 = col("ticker").eq(lit("AAPL"));
        let f2 = col("timestamp_ms").gt(lit(1_000i64));
        let f3 = col("timestamp_ms").gt(lit(500i64));
        let plan = plan_pushdown(&[&f1, &f2, &f3], &pk(), &ck(), &minute_aggs_types());
        assert_eq!(
            plan.predicates,
            vec![r#""ticker" = 'AAPL'"#, r#""timestamp_ms" > 1000"#]
        );
        assert_eq!(plan.pushed, vec![true, true, false]);
    }

    #[test]
    fn test_partial_partition_key_not_pushed() {
        let two_pk = vec!["ticker".to_string(), "seq".to_string()];
        let f1 = col("ticker").eq(lit("AAPL"));
        let plan = plan_pushdown(&[&f1], &two_pk, &ck(), &minute_aggs_types());
        assert!(plan.predicates.is_empty());
        assert_eq!(plan.pushed, vec![false]);
    }
}
