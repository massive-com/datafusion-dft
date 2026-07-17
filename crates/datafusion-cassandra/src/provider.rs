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

//! [`CassandraTable`]: a [`TableProvider`] backed by a Cassandra table or
//! materialized view

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::catalog::{Session as DfSession, TableProvider};
use datafusion::common::{plan_err, project_schema, Result};
use datafusion::datasource::TableType;
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;
use scylla::client::session::Session;
use scylla::cluster::metadata::{ColumnKind, Table};
use scylla::frame::response::result::ColumnType;

use crate::cql::{build_select, plan_pushdown};
use crate::exec::CassandraExec;
use crate::metadata::table_metadata;
use crate::types::cql_type_to_arrow;

/// [`TableProvider`] backed by a Cassandra table. The Arrow schema is derived
/// from the driver's cached `system_schema` metadata at construction time.
#[derive(Debug)]
pub struct CassandraTable {
    session: Arc<Session>,
    keyspace: String,
    name: String,
    /// Column names in schema order (partition key, clustering key, then
    /// regular columns alphabetically)
    columns: Vec<String>,
    schema: SchemaRef,
    partition_key: Vec<String>,
    clustering_key: Vec<String>,
    column_types: HashMap<String, ColumnType<'static>>,
}

impl CassandraTable {
    /// Builds a provider for `keyspace.table` from the session's cached
    /// schema metadata. Fails if the table (or materialized view) is unknown.
    pub fn try_new(session: Arc<Session>, keyspace: &str, table: &str) -> Result<Self> {
        let Some(meta) = table_metadata(&session, keyspace, table) else {
            return plan_err!("Cassandra table '{keyspace}.{table}' not found");
        };
        let columns = ordered_columns(&meta);
        let fields: Vec<Field> = columns
            .iter()
            .map(|name| {
                let column = &meta.columns[name];
                // Partition and clustering key values can never be null
                let nullable = matches!(column.kind, ColumnKind::Regular | ColumnKind::Static);
                Field::new(name, cql_type_to_arrow(&column.typ), nullable)
            })
            .collect();
        let column_types = meta
            .columns
            .iter()
            .map(|(name, column)| (name.clone(), column.typ.clone()))
            .collect();
        Ok(Self {
            session,
            keyspace: keyspace.to_string(),
            name: table.to_string(),
            columns,
            schema: Arc::new(Schema::new(fields)),
            partition_key: meta.partition_key,
            clustering_key: meta.clustering_key,
            column_types,
        })
    }
}

/// Deterministic column order: partition key, clustering key, then the
/// remaining columns alphabetically (the driver stores columns in a HashMap)
fn ordered_columns(meta: &Table) -> Vec<String> {
    let mut ordered: Vec<String> = meta
        .partition_key
        .iter()
        .chain(meta.clustering_key.iter())
        .cloned()
        .collect();
    let mut regular: Vec<String> = meta
        .columns
        .keys()
        .filter(|name| !ordered.contains(name))
        .cloned()
        .collect();
    regular.sort();
    ordered.extend(regular);
    ordered
}

#[async_trait]
impl TableProvider for CassandraTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        let plan = plan_pushdown(
            filters,
            &self.partition_key,
            &self.clustering_key,
            &self.column_types,
        );
        Ok(plan
            .pushed
            .into_iter()
            .map(|pushed| {
                if pushed {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn DfSession,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let projected_schema = project_schema(&self.schema, projection)?;
        let mut select_columns: Vec<String> = projected_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        // An empty projection (e.g. COUNT(*)) still needs a column to select;
        // the resulting values are discarded and only the row count is used
        if select_columns.is_empty() {
            select_columns.push(self.columns[0].clone());
        }
        let filter_refs: Vec<&Expr> = filters.iter().collect();
        let predicates = plan_pushdown(
            &filter_refs,
            &self.partition_key,
            &self.clustering_key,
            &self.column_types,
        )
        .predicates;
        let cql = build_select(
            &self.keyspace,
            &self.name,
            &select_columns,
            &predicates,
            limit,
        );
        Ok(Arc::new(CassandraExec::new(
            Arc::clone(&self.session),
            cql,
            projected_schema,
        )))
    }
}
