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

//! A listing table over the files as a table of the source they were written from.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::listing::{ListingTable, ListingTableConfig, PartitionedFile};
use datafusion::logical_expr::Expr;
use futures::StreamExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_datafusion::reader::VortexReaderFactory;
use vortex_ohlc::write::file_source_schema;

use crate::BarsAdapterFactory;

/// `listing`, a listing table over Vortex files, as a table of the source they were written from,
/// if the first of them records a plan (see [`file_source_schema`]): with the source's schema
/// (columns the table already has as the source has them are kept), and every file read through a
/// [`BarsAdapterFactory`]. Its definition, constraints and column defaults are kept, and its files'
/// statistics are kept in `state`'s cache, as DataFusion's `CREATE EXTERNAL TABLE` keeps them.
/// The first file is opened on `session`, through `readers`. `None` if the table has no files, or
/// the first records no plan that fits the table.
pub async fn source_table(
    listing: &ListingTable,
    state: &dyn Session,
    readers: &dyn VortexReaderFactory,
    session: &VortexSession,
) -> Result<Option<ListingTable>> {
    let Some(url) = listing.table_paths().first() else {
        return Ok(None);
    };
    let store = state.runtime_env().object_store(url.object_store())?;
    let extension = &listing.options().file_extension;
    let first = url
        .list_all_files(state, store.as_ref(), extension)
        .await?
        .next()
        .await
        .transpose()?;
    let Some(first) = first else {
        return Ok(None);
    };
    let file = PartitionedFile::from(first);
    let reader = readers.create_reader(&file, session)?;
    let file = session
        .open_options()
        .include_metadata()
        .with_file_size(file.object_meta.size)
        .open_read(reader)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    // The table's schema has the partition columns after the files'.
    let table = listing.schema();
    let partitions = listing.options().table_partition_cols.len();
    let files = &table.fields()[..table.fields().len() - partitions];
    let stored = Schema::new_with_metadata(files.to_vec(), table.metadata().clone());
    let Some(source) = file_source_schema(&file, &stored) else {
        return Ok(None);
    };
    let column_defaults: HashMap<String, Expr> = table
        .fields()
        .iter()
        .filter_map(|field| {
            let default = listing.get_column_default(field.name())?;
            Some((field.name().clone(), default.clone()))
        })
        .collect();
    let config = ListingTableConfig::new_with_multi_paths(listing.table_paths().clone())
        .with_listing_options(listing.options().clone())
        .with_schema(Arc::new(source))
        .with_expr_adapter_factory(Arc::new(BarsAdapterFactory));
    let cache = state.runtime_env().cache_manager.get_file_statistic_cache();
    Ok(Some(
        ListingTable::try_new(config)?
            .with_cache(cache)
            .with_definition(listing.get_table_definition().map(str::to_string))
            .with_constraints(listing.constraints().cloned().unwrap_or_default())
            .with_column_defaults(column_defaults),
    ))
}
