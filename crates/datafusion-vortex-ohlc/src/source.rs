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

//! A listing table over the files as a table of bars, with the prices in place of the struct
//! that stores them.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result;
use datafusion::datasource::listing::{ListingTable, ListingTableConfig};
use datafusion::logical_expr::Expr;
use vortex_ohlc::ohlc::{Part, Price};
use vortex_ohlc::plan::OHLC;

use crate::BarsAdapterFactory;

/// `listing`, a listing table over Vortex files, as a table of bars: an [`OHLC`] struct in its
/// schema becomes the four prices, at the parts' scale, and every file is read through a
/// [`BarsAdapterFactory`], so the prices are rebuilt from a file's struct. Its definition,
/// constraints and column defaults are kept, and its files' statistics are kept in `state`'s
/// cache, as DataFusion's `CREATE EXTERNAL TABLE` keeps them.
pub fn source_table(listing: &ListingTable, state: &dyn Session) -> Result<ListingTable> {
    // The table's schema has the partition columns after the files'.
    let table = listing.schema();
    let partitions = listing.options().table_partition_cols.len();
    let files = &table.fields()[..table.fields().len() - partitions];
    let schema = Schema::new_with_metadata(with_prices(files), table.metadata().clone());
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
        .with_schema(Arc::new(schema))
        .with_expr_adapter_factory(Arc::new(BarsAdapterFactory));
    let cache = state.runtime_env().cache_manager.get_file_statistic_cache();
    Ok(ListingTable::try_new(config)?
        .with_cache(cache)
        .with_definition(listing.get_table_definition().map(str::to_string))
        .with_constraints(listing.constraints().cloned().unwrap_or_default())
        .with_column_defaults(column_defaults))
}

/// `fields`, with an [`OHLC`] struct of decimal parts replaced by the four prices: each a
/// `Decimal128` at the parts' scale, two digits wider than them, which holds a sum of three.
fn with_prices(fields: &[FieldRef]) -> Vec<Field> {
    let mut out = Vec::with_capacity(fields.len() + Price::ALL.len() - 1);
    for field in fields {
        let parts = match field.data_type() {
            DataType::Struct(parts) if field.name() == OHLC => parts,
            _ => {
                out.push(field.as_ref().clone());
                continue;
            }
        };
        let names = parts.iter().map(|part| part.name().as_str());
        match parts.first().map(|open| open.data_type()) {
            Some(&DataType::Decimal128(precision, scale))
                if names.eq(Part::ALL.map(Part::name)) =>
            {
                let price = DataType::Decimal128(precision + 2, scale);
                for name in Price::ALL.map(Price::name) {
                    out.push(Field::new(name, price.clone(), field.is_nullable()));
                }
            }
            _ => out.push(field.as_ref().clone()),
        }
    }
    out
}
