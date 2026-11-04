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

//! Vortex scans in DataFusion that skip the zones their filter can't match before they start:
//!
//! - [`ZonePruner`] gives each file of a scan the zones a filter can match, from the file's zone
//!   maps, read once and kept.
//! - A [`VortexExecFactory`] builds scans with it, and a [`PrunedTable`] has one build its scans.
//!   [`PrunedTable::over`] puts it all together for a listing table over Vortex files.
//! - [`PrunedTableFactory`] and [`enable_url_table`] make a DataFusion session's Vortex tables
//!   pruned ones, those created with `CREATE EXTERNAL TABLE` and those queried by path.
//!
//! Files written by vortex-ohlc have their prices under the bars layout (`vortex_ohlc::bars`),
//! which stores them as parts: the zone maps are found under the layout, and the prices pruned on
//! the bounds of the parts.

mod prune;
mod table;
mod tables;

pub use prune::{ZonePruner, DEFAULT_CACHE_BYTES};
pub use table::{reader_factory, PrunedTable, VortexExecFactory};
pub use tables::{enable_url_table, PrunedTableFactory};
