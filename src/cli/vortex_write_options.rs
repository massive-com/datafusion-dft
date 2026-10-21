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

//! Vortex output writer options for `--output` with a `.vortex` suffix.
//!
//! The layout strategy is built here instead of using
//! `vortex_file::WriteStrategyBuilder`. That builder can only override a field with a complete
//! leaf strategy, which drops zone maps. This module composes the same pipeline as
//! `WriteStrategyBuilder::build` for every column and changes only the compressor.
//!
//! Keep the pipeline in [`build_leaf`] in sync with `vortex_file::WriteStrategyBuilder::build`
//! when the pinned Vortex version changes.

use color_eyre::eyre::{eyre, Result};
use datafusion::arrow::datatypes::{DataType, Schema};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::OnceLock;
use vortex::array::session::ArraySessionExt;
use vortex::array::{ArrayId, ArrayRef, Canonical, ExecutionCtx};
use vortex::compressor::{
    BtrBlocksCompressor, BtrBlocksCompressorBuilder, CompressionSessionExt, Scheme,
};
use vortex::dtype::{FieldName, FieldPath};
use vortex::editions::{
    ComponentKind, Edition, EditionDeclaration, EditionFamily, EditionId, EditionMember,
    EditionSessionExt,
};
use vortex::error::VortexResult;
use vortex::layout::layouts::buffered::BufferedStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::collect::CollectStrategy;
use vortex::layout::layouts::compressed::{CompressingStrategy, CompressorPlugin};
use vortex::layout::layouts::dict::writer::{DictLayoutOptions, DictStrategy};
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::list::writer::ListLayoutStrategy;
use vortex::layout::layouts::list::List;
use vortex::layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex::layout::layouts::table::TableStrategy;
use vortex::layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
use vortex::layout::{LayoutStrategy, VTable};
use vortex::VortexSessionDefault;
use vortex_btrblocks::schemes::{binary, float, integer, string};
use vortex_btrblocks::{
    ArrayAndStats, CascadingCompressor, CompressorContext, GenerateStatsOptions, SchemeExt,
    SchemeId,
};
use vortex_compressor::scheme::{
    AncestorExclusion, CompressionEstimate, DescendantExclusion, EstimateVerdict,
};
use vortex_session::VortexSession;

/// Default row block size. Matches `vortex_file::WriteStrategyBuilder`.
const DEFAULT_ROW_BLOCK_SIZE: usize = 8192;
/// Default target size for coalesced data blocks. Matches `vortex_file::WriteStrategyBuilder`.
const DEFAULT_DATA_BLOCK_TARGET_BYTES: u64 = 1 << 20;
/// Buffer size between compression and layout emission. Matches `vortex_file::WriteStrategyBuilder`.
const BUFFER_SIZE: u64 = 2 * (1 << 20);

/// dft's edition family, for Vortex components dft writes that no Vortex edition lists yet.
static DFT_EDITION_FAMILY: EditionFamily = EditionFamily {
    name: "dft",
    origin: "datafusion-dft",
    doc: "Vortex components that datafusion-dft writes on request and that no Vortex edition \
lists yet. Files that use them need a Vortex reader that decodes those components.",
};

/// The edition `blocked_encodings` enables.
const BLOCKED_ENCODINGS_EDITION: EditionId = EditionId::new("dft", 2026, 10, 0);

/// Vortex's blocked FoR (a reference per chunk of values) and blocked bit-packing (a bit width
/// per 1024 values), new in Vortex 0.89 and in no Vortex edition.
static BLOCKED_ENCODINGS_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: BLOCKED_ENCODINGS_EDITION,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"fastlanes.for.v2"),
        EditionMember::array(&"fastlanes.bitpacked.v2"),
    ],
};

/// Parsed `--format-option` values for a `.vortex` output file.
#[derive(Debug)]
pub struct VortexWriteConfig {
    row_block_size: usize,
    data_block_target_bytes: Option<u64>,
    compact: bool,
    statistics: bool,
    blocked_encodings: bool,
    columns: Vec<ColumnOverride>,
}

#[derive(Debug)]
struct ColumnOverride {
    name: String,
    encoding: Option<String>,
    dictionary: Option<bool>,
    statistics: Option<bool>,
}

impl Default for VortexWriteConfig {
    fn default() -> Self {
        Self {
            row_block_size: DEFAULT_ROW_BLOCK_SIZE,
            data_block_target_bytes: Some(DEFAULT_DATA_BLOCK_TARGET_BYTES),
            compact: false,
            statistics: true,
            blocked_encodings: false,
            columns: Vec::new(),
        }
    }
}

impl VortexWriteConfig {
    /// Parse `--format-option KEY=VALUE` pairs. The optional `format.` prefix is accepted.
    pub fn parse(options: &[(String, String)]) -> Result<Self> {
        let mut config = Self::default();
        for (key, value) in options {
            let key = key.strip_prefix("format.").unwrap_or(key);
            match key {
                "row_block_size" => {
                    let size: usize = value.parse().map_err(|_| {
                        eyre!("Invalid row_block_size '{value}': expected a positive integer")
                    })?;
                    if size == 0 {
                        return Err(eyre!(
                            "Invalid row_block_size '0': expected a positive integer"
                        ));
                    }
                    config.row_block_size = size;
                }
                "data_block_target_bytes" => {
                    config.data_block_target_bytes = match value.to_ascii_lowercase().as_str() {
                        "none" | "off" | "0" => None,
                        _ => Some(value.parse().map_err(|_| {
                            eyre!(
                                "Invalid data_block_target_bytes '{value}': expected a byte count or 'none'"
                            )
                        })?),
                    };
                }
                "compressor" => {
                    config.compact = match value.to_ascii_lowercase().as_str() {
                        "default" => false,
                        "compact" => true,
                        other => {
                            return Err(eyre!(
                                "Unknown vortex compressor '{other}': expected 'default' or 'compact'"
                            ));
                        }
                    };
                }
                "statistics_enabled" => {
                    config.statistics = parse_bool(value)?;
                }
                "blocked_encodings" => {
                    config.blocked_encodings = parse_bool(value)?;
                }
                _ => {
                    if let Some(column) = key.strip_prefix("encoding::") {
                        if column.is_empty() {
                            return Err(eyre!(
                                "encoding requires a column name, for example encoding::id=bitpacking"
                            ));
                        }
                        config.column(column).encoding = Some(value.clone());
                    } else if let Some(column) = key.strip_prefix("dictionary_enabled::") {
                        if column.is_empty() {
                            return Err(eyre!(
                                "dictionary_enabled requires a column name, for example dictionary_enabled::id=false"
                            ));
                        }
                        let enabled = parse_bool(value)?;
                        config.column(column).dictionary = Some(enabled);
                    } else if let Some(column) = key.strip_prefix("statistics_enabled::") {
                        if column.is_empty() {
                            return Err(eyre!(
                                "statistics_enabled requires a column name, for example statistics_enabled::id=false"
                            ));
                        }
                        let enabled = parse_bool(value)?;
                        config.column(column).statistics = Some(enabled);
                    } else {
                        return Err(eyre!("Unknown vortex output format option '{key}'"));
                    }
                }
            }
        }
        Ok(config)
    }

    /// The session to write with: Vortex's defaults, plus [`BLOCKED_ENCODINGS_EDITION`] when
    /// `blocked_encodings` is set. The strategy and the writer both take what they may write from
    /// the session's enabled editions.
    pub fn session(&self) -> Result<VortexSession> {
        let session = VortexSession::default();
        if self.blocked_encodings {
            session.editions().declare_family(&DFT_EDITION_FAMILY)?;
            session.register_edition(&BLOCKED_ENCODINGS_DECLARATION)?;
            session.enable_edition(BLOCKED_ENCODINGS_EDITION)?;
        }
        Ok(session)
    }

    fn column(&mut self, name: &str) -> &mut ColumnOverride {
        if let Some(index) = self.columns.iter().position(|column| column.name == name) {
            return &mut self.columns[index];
        }
        self.columns.push(ColumnOverride {
            name: name.to_string(),
            encoding: None,
            dictionary: None,
            statistics: None,
        });
        self.columns.last_mut().expect("a column was just pushed")
    }

    /// Build the layout strategy. Every column uses the same pipeline; only the compressor and
    /// the zone-map layer differ per column, so overrides keep zone maps unless disabled.
    pub fn build_strategy(
        &self,
        schema: &Schema,
        session: &VortexSession,
    ) -> Result<Arc<dyn LayoutStrategy>> {
        let base = if self.compact {
            BtrBlocksCompressorBuilder::from_session(session).with_compact()
        } else {
            BtrBlocksCompressorBuilder::from_session(session)
        };
        let stats_compressor: Arc<dyn CompressorPlugin> = Arc::new(base.clone().build());
        let adaptive_data_compressor = base
            .clone()
            .exclude_schemes([integer::IntDictScheme.id()])
            .build();
        let default_data_compressor: Arc<dyn CompressorPlugin> =
            Arc::new(adaptive_data_compressor.clone());

        let row_block_size = NonZeroUsize::new(self.row_block_size)
            .ok_or_else(|| eyre!("row_block_size must be greater than zero"))?;

        let default_leaf = build_leaf(
            Arc::clone(&default_data_compressor),
            Arc::clone(&stats_compressor),
            row_block_size,
            self.data_block_target_bytes,
            true,
            self.statistics,
        );

        let writable_ids = writable_array_ids(session);
        let writable = |id: &ArrayId| writable_ids.contains(id);

        // The builder each level of an override chain starts from: the column's adaptive schemes,
        // with the compact preset for a forced Zstd or Pco, which the default preset excludes.
        let data_builder = base.clone().exclude_schemes([integer::IntDictScheme.id()]);
        let level_builder = |scheme: &'static dyn Scheme| {
            if !self.compact && default_preset_excludes(scheme) {
                data_builder.clone().with_compact()
            } else {
                data_builder.clone()
            }
        };

        let mut field_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>> = HashMap::new();
        for column in &self.columns {
            let (path, data_type) = resolve_field(schema, &column.name)?;
            if let Some(other) = field_writers.keys().find(|other| path.overlap(other)) {
                return Err(eyre!(
                    "Vortex format options for '{}' and '{other}' overlap; set options on one of them",
                    column.name
                ));
            }
            let kind = scheme_kind(&data_type).ok_or_else(|| {
                eyre!(
                    "Column '{}' has type {} and cannot use a vortex encoding override",
                    column.name,
                    data_type
                )
            })?;
            let data_compressor: Arc<dyn CompressorPlugin> = match &column.encoding {
                Some(raw) => {
                    let mut chain = Vec::new();
                    for (index, step) in raw.split('/').enumerate() {
                        let scheme = if index == 0 {
                            resolve_scheme(step, kind, &column.name)?
                        } else {
                            resolve_scheme_any(step, &column.name)?
                        };
                        check_writable(scheme, &column.name, &writable)?;
                        chain.push(scheme.refine(&writable));
                    }
                    Arc::new(override_compressor(
                        &chain,
                        &level_builder,
                        &adaptive_data_compressor,
                    ))
                }
                None => Arc::clone(&default_data_compressor),
            };
            let leaf = build_leaf(
                data_compressor,
                Arc::clone(&stats_compressor),
                row_block_size,
                self.data_block_target_bytes,
                column.dictionary.unwrap_or(true),
                column.statistics.unwrap_or(self.statistics),
            );
            field_writers.insert(path, leaf);
        }

        let validity: Arc<dyn LayoutStrategy> = Arc::new(CollectStrategy::new(
            CompressingStrategy::new(flat_strategy(), Arc::clone(&stats_compressor)),
        ));

        let mut table = TableStrategy::new(validity, Arc::clone(&default_leaf))
            .with_field_writers(field_writers);

        // Matches `vortex_file::WriteStrategyBuilder`: list fields use the list layout only when
        // an enabled edition includes it.
        if session
            .enabled_component_ids(ComponentKind::Layout)
            .contains(&List.id())
        {
            let stats_compressor = Arc::clone(&stats_compressor);
            table = table.with_list_layout_factory(
                move |list_layout: ListLayoutStrategy| -> Arc<dyn LayoutStrategy> {
                    let zoned = ZonedStrategy::new(
                        list_layout,
                        CompressingStrategy::new(flat_strategy(), Arc::clone(&stats_compressor)),
                        ZonedLayoutOptions {
                            block_size: row_block_size,
                            ..Default::default()
                        },
                    );
                    Arc::new(RepartitionStrategy::new(
                        zoned,
                        RepartitionWriterOptions {
                            block_size_minimum: 0,
                            block_len_multiple: row_block_size.get(),
                            block_size_target: None,
                            canonicalize: false,
                        },
                    ))
                },
            );
        }

        Ok(Arc::new(table))
    }
}

fn flat_strategy() -> Arc<dyn LayoutStrategy> {
    Arc::new(FlatLayoutStrategy::default())
}

/// The serialized array IDs a write with `session` accepts: registered in the session and
/// permitted by its enabled editions. `VortexWriteOptions::write` rejects any other ID while
/// serializing, after the query has run and part of the file is written.
fn writable_array_ids(session: &VortexSession) -> HashSet<ArrayId> {
    let editions: HashSet<ArrayId> = session
        .enabled_component_ids(ComponentKind::Array)
        .into_iter()
        .collect();
    session.arrays().registry().read(|registry| {
        registry
            .keys()
            .copied()
            .filter(|id| editions.contains(id))
            .collect()
    })
}

/// Fail before writing when a forced scheme can only produce encodings the writer rejects.
///
/// This is the check `BtrBlocksCompressorBuilder::build` uses to drop such a scheme, which would
/// otherwise turn the override into a silent no-op.
fn check_writable(
    scheme: &'static dyn Scheme,
    column: &str,
    writable: &dyn Fn(&ArrayId) -> bool,
) -> Result<()> {
    let rejected: Vec<String> = scheme
        .refine(writable)
        .produced_encodings()
        .iter()
        .filter(|id| !writable(id))
        .map(|id| id.to_string())
        .collect();
    if rejected.is_empty() {
        return Ok(());
    }
    Err(eyre!(
        "Encoding '{}' for column '{column}' writes {}, which the enabled Vortex editions do not permit",
        scheme.scheme_name(),
        rejected.join(", ")
    ))
}

/// Build a compressor that forces `chain` down a column.
///
/// `chain[0]` encodes the top of every chunk, and each later scheme encodes the direct children of
/// the one before it that it applies to (`byte_parts/for` forces FoR on the integer part of decimal
/// byte parts). Everything else, including children a scheme does not apply to and the children of
/// the last scheme, is compressed adaptively. A forced scheme is used even where its own estimate
/// would skip the data, unless it fails or does not shrink it, in which case the data is compressed
/// adaptively instead.
///
/// Each level is the adaptive compressor with its scheme replaced by a [`ForcedScheme`], so the
/// schemes it does not force still compete as usual.
fn override_compressor(
    chain: &[&'static dyn Scheme],
    level_builder: &dyn Fn(&'static dyn Scheme) -> BtrBlocksCompressorBuilder,
    adaptive: &BtrBlocksCompressor,
) -> BtrBlocksCompressor {
    let mut children = adaptive.0.clone();
    let mut compressor = None;
    for (index, &scheme) in chain.iter().enumerate().rev() {
        // `CascadingCompressor` takes `&'static` schemes, so one is leaked per chain step.
        let forced: &'static ForcedScheme = Box::leak(Box::new(ForcedScheme {
            inner: scheme,
            parent: index.checked_sub(1).map(|parent| chain[parent].id()),
            children,
            fallback: adaptive.0.clone(),
        }));
        let level = level_builder(scheme)
            .exclude_schemes([scheme.id()])
            .with_new_scheme(forced)
            .build();
        children = level.0.clone();
        compressor = Some(level);
    }
    compressor.expect("an override chain has at least one scheme")
}

/// Whether `BtrBlocksCompressorBuilder::from_session`'s default preset leaves `scheme` out.
fn default_preset_excludes(scheme: &dyn Scheme) -> bool {
    [
        string::ZstdScheme.id(),
        binary::ZstdScheme.id(),
        binary::ZstdBuffersScheme.id(),
        integer::PcoScheme.id(),
        float::PcoScheme.id(),
    ]
    .contains(&scheme.id())
}

/// `inner`, forced on the direct children of `parent` (or on the top of a column, for `None`):
/// there it is always chosen, and its children are compressed with `children`. Anywhere else it
/// is `inner`, competing as usual.
#[derive(Debug)]
struct ForcedScheme {
    inner: &'static dyn Scheme,
    parent: Option<SchemeId>,
    children: CascadingCompressor,
    fallback: CascadingCompressor,
}

impl ForcedScheme {
    fn is_forced(&self, compress_ctx: &CompressorContext) -> bool {
        compress_ctx.cascade_history().last().map(|(id, _)| *id) == self.parent
    }
}

impl Scheme for ForcedScheme {
    fn scheme_name(&self) -> &'static str {
        self.inner.scheme_name()
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        self.inner.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        self.inner.produced_encodings()
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        self.inner.stats_options()
    }

    fn num_children(&self) -> usize {
        self.inner.num_children()
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        self.inner.descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        self.inner.ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if self.is_forced(&compress_ctx) {
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }
        self.inner
            .expected_compression_ratio(data, compress_ctx, exec_ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if !self.is_forced(&compress_ctx) {
            return self
                .inner
                .compress(compressor, data, compress_ctx, exec_ctx);
        }
        let before = data.array().nbytes();
        match self
            .inner
            .compress(&self.children, data, compress_ctx, exec_ctx)
        {
            Ok(compressed) if compressed.nbytes() < before => Ok(compressed),
            Ok(_) => self.fallback.compress(data.array(), exec_ctx),
            Err(err) => {
                log::debug!(
                    "Forced vortex encoding {} failed, compressing adaptively: {err}",
                    self.inner.scheme_name()
                );
                self.fallback.compress(data.array(), exec_ctx)
            }
        }
    }
}

/// Compose the leaf pipeline used for a single column.
///
/// This mirrors the leaf of `vortex_file::WriteStrategyBuilder::build`, including the
/// [`ZonedStrategy`] that writes zone maps. When `zone_maps` is false, the zone-map layer is
/// skipped. The outer repartition exists only to give the zone-map layer fixed-size row blocks,
/// so it is skipped too.
fn build_leaf(
    data_compressor: Arc<dyn CompressorPlugin>,
    stats_compressor: Arc<dyn CompressorPlugin>,
    row_block_size: NonZeroUsize,
    data_block_target_bytes: Option<u64>,
    dictionary: bool,
    zone_maps: bool,
) -> Arc<dyn LayoutStrategy> {
    let flat = flat_strategy();
    let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat));
    let buffered = BufferedStrategy::new(chunked, BUFFER_SIZE);
    let compressing = CompressingStrategy::new(buffered, data_compressor);
    let coalescing = RepartitionStrategy::new(
        compressing,
        RepartitionWriterOptions {
            block_size_minimum: data_block_target_bytes.unwrap_or(0),
            block_len_multiple: row_block_size.get(),
            block_size_target: data_block_target_bytes,
            canonicalize: true,
        },
    );
    let compress_then_flat =
        CompressingStrategy::new(Arc::clone(&flat), Arc::clone(&stats_compressor));

    let data: Arc<dyn LayoutStrategy> = if dictionary {
        Arc::new(DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            DictLayoutOptions::default(),
            Arc::clone(&stats_compressor),
        ))
    } else {
        Arc::new(coalescing)
    };

    if !zone_maps {
        return data;
    }

    let stats = ZonedStrategy::new(
        data,
        compress_then_flat,
        ZonedLayoutOptions {
            block_size: row_block_size,
            ..Default::default()
        },
    );

    Arc::new(RepartitionStrategy::new(
        stats,
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: row_block_size.get(),
            block_size_target: None,
            canonicalize: false,
        },
    ))
}

struct SchemeEntry {
    name: &'static str,
    kind: &'static str,
    short: &'static str,
    scheme: &'static dyn Scheme,
}

fn scheme_entries() -> &'static [SchemeEntry] {
    static ENTRIES: OnceLock<Vec<SchemeEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        let mut entries = Vec::new();
        for scheme in VortexSession::default().compression().schemes() {
            push_scheme_entry(&mut entries, *scheme);
        }
        entries
    })
}

fn push_scheme_entry(entries: &mut Vec<SchemeEntry>, scheme: &'static dyn Scheme) {
    let name = scheme.scheme_name();
    let Some(rest) = name.strip_prefix("vortex.") else {
        return;
    };
    let Some((kind, short)) = rest.split_once('.') else {
        return;
    };
    entries.push(SchemeEntry {
        name,
        kind,
        short,
        scheme,
    });
}

/// Resolve a user supplied encoding name against the column kind.
///
/// Accepts the full name (`vortex.int.bitpacking`), a kind-qualified short name
/// (`int.bitpacking`), or a short name (`bitpacking`) when the column kind disambiguates it.
fn resolve_scheme(raw: &str, kind: &str, column: &str) -> Result<&'static dyn Scheme> {
    let entries = scheme_entries();

    if let Some(entry) = entries.iter().find(|entry| entry.name == raw) {
        if entry.kind != kind {
            return Err(eyre!(
                "Encoding '{raw}' is for {} columns, but column '{column}' is {kind}",
                entry.kind
            ));
        }
        return Ok(entry.scheme);
    }

    if let Some((prefix, short)) = raw.split_once('.') {
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.kind == prefix && entry.short == short)
        {
            if entry.kind != kind {
                return Err(eyre!(
                    "Encoding '{raw}' is for {} columns, but column '{column}' is {kind}",
                    entry.kind
                ));
            }
            return Ok(entry.scheme);
        }
    }

    if let Some(entry) = entries
        .iter()
        .find(|entry| entry.kind == kind && entry.short == raw)
    {
        return Ok(entry.scheme);
    }

    let mut available: Vec<&str> = entries
        .iter()
        .filter(|entry| entry.kind == kind)
        .map(|entry| entry.short)
        .collect();
    available.sort_unstable();
    available.dedup();
    Err(eyre!(
        "Unknown vortex encoding '{raw}' for {kind} column '{column}'; available encodings: {}",
        available.join(", ")
    ))
}

/// Resolve a column option's name to its field path and type: a top-level column, or failing that
/// a dot-separated path through struct fields (`ohlc.open`).
fn resolve_field(schema: &Schema, name: &str) -> Result<(FieldPath, DataType)> {
    if let Ok(field) = schema.field_with_name(name) {
        return Ok((
            FieldPath::from_name(name.to_string()),
            field.data_type().clone(),
        ));
    }
    let unknown = || eyre!("Unknown column '{name}' in vortex format option");
    let mut parts = name.split('.');
    let first = parts.next().ok_or_else(unknown)?;
    let mut data_type = schema
        .field_with_name(first)
        .map_err(|_| unknown())?
        .data_type()
        .clone();
    let mut path = FieldPath::from_name(first.to_string());
    for part in parts {
        let DataType::Struct(fields) = &data_type else {
            return Err(unknown());
        };
        let field = fields
            .iter()
            .find(|field| field.name() == part)
            .ok_or_else(unknown)?;
        data_type = field.data_type().clone();
        path = path.push(FieldName::from(part));
    }
    Ok((path, data_type))
}

/// Resolve an encoding below the top of an override chain, where the type of the array it applies
/// to is not known up front: a full or kind-qualified name, or a short name only one kind has.
fn resolve_scheme_any(raw: &str, column: &str) -> Result<&'static dyn Scheme> {
    let entries = scheme_entries();
    if let Some(entry) = entries.iter().find(|entry| entry.name == raw) {
        return Ok(entry.scheme);
    }
    if let Some((prefix, short)) = raw.split_once('.') {
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.kind == prefix && entry.short == short)
        {
            return Ok(entry.scheme);
        }
    }
    let matching: Vec<&SchemeEntry> = entries.iter().filter(|entry| entry.short == raw).collect();
    match matching.as_slice() {
        [entry] => Ok(entry.scheme),
        [] => Err(eyre!(
            "Unknown vortex encoding '{raw}' in the override for column '{column}'"
        )),
        _ => {
            let qualified: Vec<String> = matching
                .iter()
                .map(|entry| format!("{}.{}", entry.kind, entry.short))
                .collect();
            Err(eyre!(
                "Vortex encoding '{raw}' in the override for column '{column}' is ambiguous below the top of the column; use one of: {}",
                qualified.join(", ")
            ))
        }
    }
}

/// Map an Arrow type to the Vortex scheme kind prefix.
fn scheme_kind(dtype: &DataType) -> Option<&'static str> {
    match dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Some("int"),
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Some("float"),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Some("string"),
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => Some("binary"),
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => Some("decimal"),
        DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(_) => Some("ext"),
        _ => None,
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(eyre!("Expected a boolean, got '{other}'")),
    }
}

/// Print the available `.vortex` output format options.
pub fn print_options() {
    println!("Available vortex output format options:\n");
    println!("{:<44} {:<18} DESCRIPTION", "OPTION", "DEFAULT");
    println!(
        "{:<44} {:<18} Row block size for repartitioning and zone maps",
        "row_block_size", DEFAULT_ROW_BLOCK_SIZE
    );
    println!(
        "{:<44} {:<18} Target uncompressed bytes per data block; 'none' disables coalescing",
        "data_block_target_bytes", DEFAULT_DATA_BLOCK_TARGET_BYTES
    );
    println!(
        "{:<44} {:<18} 'default' or 'compact' (adds zstd and pco schemes)",
        "compressor", "default"
    );
    println!(
        "{:<44} {:<18} Force a compression scheme; COLUMN may be a.b, VALUE may be a/b",
        "encoding::COLUMN", "<adaptive>"
    );
    println!(
        "{:<44} {:<18} Enable dictionary encoding for a top-level column",
        "dictionary_enabled::COLUMN", "true"
    );
    println!(
        "{:<44} {:<18} Write zone maps (statistics) for pruning",
        "statistics_enabled", "true"
    );
    println!(
        "{:<44} {:<18} Allow blocked FoR and bit-packing (needs a Vortex 0.89+ reader)",
        "blocked_encodings", "false"
    );
    println!(
        "{:<44} {:<18} Write zone maps for a top-level column",
        "statistics_enabled::COLUMN", "true"
    );

    println!("\nPass an option with: --format-option KEY=VALUE");
    println!(
        "Use KEY::COLUMN=VALUE for column-specific keys (for example, encoding::id=bitpacking)."
    );

    // List only the encodings the default editions let the writer use.
    let writable_ids = writable_array_ids(&VortexSession::default());
    let writable = |id: &ArrayId| writable_ids.contains(id);
    let entries: Vec<&SchemeEntry> = scheme_entries()
        .iter()
        .filter(|entry| check_writable(entry.scheme, "", &writable).is_ok())
        .collect();

    println!("\nAvailable encodings by column type:");
    let mut kinds: Vec<&str> = entries.iter().map(|entry| entry.kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    for kind in kinds {
        let mut shorts: Vec<&str> = entries
            .iter()
            .filter(|entry| entry.kind == kind)
            .map(|entry| entry.short)
            .collect();
        shorts.sort_unstable();
        shorts.dedup();
        println!("  {kind}: {}", shorts.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{
        ArrayRef as ArrowArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray,
        StructArray,
    };
    use datafusion::arrow::datatypes::Field;
    use futures::StreamExt;
    use vortex::arrow::ArrowSessionExt;
    use vortex::file::OpenOptionsSessionExt;
    use vortex_file::VortexWriteOptions;

    const ROWS: i64 = 65_536;

    fn options(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    /// Write `column` as column `c` with an override to `encoding`, then return the chunks of `c`
    /// read back from the file.
    async fn written_chunks(column: ArrowArrayRef, encoding: &str) -> Vec<ArrayRef> {
        written_chunks_with(column, encoding, &[]).await
    }

    /// [`written_chunks`], with `extra` format options.
    async fn written_chunks_with(
        column: ArrowArrayRef,
        encoding: &str,
        extra: &[(&str, &str)],
    ) -> Vec<ArrayRef> {
        let mut pairs = vec![
            ("encoding::c", encoding),
            ("dictionary_enabled::c", "false"),
        ];
        pairs.extend_from_slice(extra);
        written_field_chunks(single_column_batch(column), &pairs, &["c"]).await
    }

    fn single_column_batch(column: ArrowArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c",
            column.data_type().clone(),
            false,
        )]));
        RecordBatch::try_new(schema, vec![column]).unwrap()
    }

    /// Write `batch` with the format options `pairs`, then return the chunks of the field at
    /// `path` read back from the file.
    async fn written_field_chunks(
        batch: RecordBatch,
        pairs: &[(&str, &str)],
        path: &[&str],
    ) -> Vec<ArrayRef> {
        let schema = batch.schema();
        let config = VortexWriteConfig::parse(&options(pairs)).unwrap();
        let session = config.session().unwrap();
        let strategy = config.build_strategy(&schema, &session).unwrap();

        let array = session
            .arrow()
            .from_arrow_record_batch(batch, &schema)
            .unwrap();
        let mut buffer = Vec::new();
        VortexWriteOptions::new(session.clone())
            .with_strategy(strategy)
            .write(&mut buffer, array.to_array_stream())
            .await
            .unwrap();

        let file = session.open_options().open_buffer(buffer).unwrap();
        let mut stream = file.scan().unwrap().into_array_stream().unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            let mut array = chunk.unwrap();
            for name in path {
                array = child(&array, name);
            }
            chunks.push(array);
        }
        chunks
    }

    fn child(array: &ArrayRef, name: &str) -> ArrayRef {
        array
            .named_children()
            .into_iter()
            .find(|(child, _)| child == name)
            .map(|(_, child)| child)
            .unwrap_or_else(|| panic!("{} has no child {name}", array.encoding_id()))
    }

    /// Whether `array` is encoded with `family`. Arrays read back report their in-memory encoding
    /// ID, which is the newest wire format's ID (for example `fastlanes.bitpacked.v2`) even when
    /// the file stores an older one. The write itself only succeeds when every serialized ID is in
    /// the enabled editions.
    fn is_family(array: &ArrayRef, family: &str) -> bool {
        let id = array.encoding_id().to_string();
        id == family || id.starts_with(&format!("{family}."))
    }

    /// Assert that `encoding` encodes the top of every chunk, and return the chunks.
    async fn assert_written_with(
        column: ArrowArrayRef,
        encoding: &str,
        family: &str,
    ) -> Vec<ArrayRef> {
        let chunks = written_chunks(column, encoding).await;
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                is_family(chunk, family),
                "expected {family} for encoding '{encoding}', got {}",
                chunk.encoding_id()
            );
        }
        chunks
    }

    #[tokio::test]
    async fn bitpacking_override_is_written() {
        // Increasing values need a different bit width per block, which bitpacking's v2 wire
        // format stores. v2 is outside the default editions, so the override must fall back to
        // v1 rather than fail while writing.
        let values = Int64Array::from_iter_values(0..ROWS);
        assert_written_with(Arc::new(values), "bitpacking", "fastlanes.bitpacked").await;
    }

    #[tokio::test]
    async fn for_override_is_written() {
        // Drifting values favour a reference per chunk, which only FoR's v2 wire format stores.
        let values = Int64Array::from_iter_values((0..ROWS).map(|i| 1_000_000_000 + i));
        assert_written_with(Arc::new(values), "for", "fastlanes.for").await;
    }

    #[tokio::test]
    async fn byte_parts_override_compresses_its_children() {
        let values =
            Decimal128Array::from_iter_values((0..ROWS).map(|i| ((i * 7919) % 100_000) as i128))
                .with_precision_and_scale(12, 2)
                .unwrap();
        let chunks =
            assert_written_with(Arc::new(values), "byte_parts", "vortex.decimal_byte_parts").await;
        // The integer part is left to the adaptive compressor rather than stored as is.
        for chunk in &chunks {
            for child in chunk.children() {
                assert!(
                    !child.is_canonical(),
                    "uncompressed {} under {}",
                    child.encoding_id(),
                    chunk.encoding_id()
                );
            }
        }
    }

    #[tokio::test]
    async fn fsst_override_is_written() {
        let values = StringArray::from_iter_values(
            (0..ROWS).map(|i| format!("https://example.com/item/{}", (i * 7919) % 5000)),
        );
        assert_written_with(Arc::new(values), "fsst", "vortex.fsst").await;
    }

    #[tokio::test]
    async fn forced_override_ignores_the_scheme_estimate() {
        // With a minimum of 1, FoR narrows nothing that bit-packing doesn't, so its own estimate
        // skips these values. Forced, it encodes them anyway.
        let values = Int64Array::from_iter_values((0..ROWS).map(|i| 1 + (i * 7919) % 1000));
        assert_written_with(Arc::new(values), "for", "fastlanes.for").await;
    }

    #[tokio::test]
    async fn forced_override_that_fails_is_compressed_adaptively() {
        // These values are not a sequence, so the sequence encoding cannot represent them.
        let values = Int64Array::from_iter_values((0..ROWS).map(|i| (i * 7919) % 1000));
        let chunks = written_chunks(Arc::new(values), "sequence").await;
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                !chunk.is_canonical(),
                "uncompressed {}",
                chunk.encoding_id()
            );
            assert!(!is_family(chunk, "vortex.sequence"));
        }
    }

    #[tokio::test]
    async fn override_applies_to_a_struct_field() {
        let field = |name: &str| Arc::new(Field::new(name, DataType::Int64, false));
        let a: ArrowArrayRef = Arc::new(Int64Array::from_iter_values(0..ROWS));
        let b: ArrowArrayRef = Arc::new(Int64Array::from_iter_values(0..ROWS));
        let s: ArrowArrayRef = Arc::new(StructArray::from(vec![(field("a"), a), (field("b"), b)]));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            s.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![s]).unwrap();
        let pairs = [
            ("encoding::s.a", "bitpacking"),
            ("dictionary_enabled::s.a", "false"),
        ];

        // The sequence 0, 1, 2, ... is left as a sequence unless the field is forced otherwise.
        let a = written_field_chunks(batch.clone(), &pairs, &["s", "a"]).await;
        assert!(a
            .iter()
            .all(|chunk| is_family(chunk, "fastlanes.bitpacked")));
        let b = written_field_chunks(batch, &pairs, &["s", "b"]).await;
        assert!(b
            .iter()
            .all(|chunk| !is_family(chunk, "fastlanes.bitpacked")));
    }

    /// Sorted values in a few drifting levels, like prices of a few tickers: the integer part of
    /// their decimal byte parts suits a FoR reference per block, which FoR's estimate skips.
    fn sorted_levels() -> ArrowArrayRef {
        let values = (0..ROWS).map(|i| {
            let level = 100_000 + (i / 4096) * 250_000;
            (level + (i * 7919) % 50) as i128
        });
        Arc::new(
            Decimal128Array::from_iter_values(values)
                .with_precision_and_scale(16, 4)
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn override_chain_forces_a_child_encoding() {
        let msp = |chunks: &[ArrayRef]| -> Vec<ArrayRef> {
            chunks
                .iter()
                .map(|chunk| chunk.children()[0].clone())
                .collect()
        };
        let adaptive =
            assert_written_with(sorted_levels(), "byte_parts", "vortex.decimal_byte_parts").await;
        assert!(msp(&adaptive)
            .iter()
            .all(|part| !is_family(part, "fastlanes.for")));

        let chained = assert_written_with(
            sorted_levels(),
            "byte_parts/for",
            "vortex.decimal_byte_parts",
        )
        .await;
        assert!(msp(&chained)
            .iter()
            .all(|part| is_family(part, "fastlanes.for")));
    }

    #[test]
    fn override_chain_step_must_be_unambiguous() {
        let schema = Schema::new(vec![Field::new("c", DataType::Decimal128(16, 4), false)]);
        let Err(err) = VortexWriteConfig::parse(&options(&[("encoding::c", "byte_parts/dict")]))
            .unwrap()
            .build_strategy(&schema, &VortexSession::default())
        else {
            panic!("expected an ambiguous chain step to be rejected");
        };
        assert!(err.to_string().contains("int.dict"), "{err}");
    }

    #[test]
    fn unknown_struct_field_is_rejected() {
        let inner = Field::new("a", DataType::Int64, false);
        let schema = Schema::new(vec![Field::new(
            "s",
            DataType::Struct(vec![inner].into()),
            false,
        )]);
        let Err(err) = VortexWriteConfig::parse(&options(&[("encoding::s.missing", "bitpacking")]))
            .unwrap()
            .build_strategy(&schema, &VortexSession::default())
        else {
            panic!("expected an unknown struct field to be rejected");
        };
        assert!(
            err.to_string().contains("Unknown column 's.missing'"),
            "{err}"
        );
    }

    /// Whether any chunk has a child named `name`.
    fn has_child(chunks: &[ArrayRef], name: &str) -> bool {
        chunks.iter().any(|chunk| {
            chunk
                .named_children()
                .iter()
                .any(|(child, _)| child == name)
        })
    }

    #[tokio::test]
    async fn blocked_encodings_write_blocked_bit_packing() {
        // Increasing values need a different bit width per block. Only blocked bit-packing stores
        // one, with the offsets of its blocks as a child.
        let values = || Arc::new(Int64Array::from_iter_values(0..ROWS));
        let default = written_chunks(values(), "bitpacking").await;
        assert!(!has_child(&default, "block_offsets"));

        let blocked =
            written_chunks_with(values(), "bitpacking", &[("blocked_encodings", "true")]).await;
        assert!(has_child(&blocked, "block_offsets"));
    }

    #[tokio::test]
    async fn blocked_encodings_write_blocked_for() {
        // Drifting values favour a reference per chunk, which only blocked FoR stores.
        let values = || {
            Arc::new(Int64Array::from_iter_values(
                (0..ROWS).map(|i| 1_000_000_000 + i),
            ))
        };
        let default = written_chunks(values(), "for").await;
        assert!(!has_child(&default, "blocked_references"));

        let blocked = written_chunks_with(values(), "for", &[("blocked_encodings", "true")]).await;
        assert!(has_child(&blocked, "blocked_references"));
    }

    #[test]
    fn override_outside_enabled_editions_is_rejected() {
        // Delta is registered but not in the default editions, so the writer would reject it.
        let schema = Schema::new(vec![Field::new("c", DataType::Int64, false)]);
        let Err(err) = VortexWriteConfig::parse(&options(&[("encoding::c", "delta")]))
            .unwrap()
            .build_strategy(&schema, &VortexSession::default())
        else {
            panic!("expected the delta override to be rejected");
        };
        assert!(err.to_string().contains("fastlanes.delta"), "{err}");
    }

    #[test]
    fn check_writable_rejects_encodings_not_permitted() {
        let fsst = resolve_scheme("fsst", "string", "c").unwrap();
        let produced = fsst.produced_encodings();
        let err = check_writable(fsst, "c", &|id: &ArrayId| !produced.contains(id)).unwrap_err();
        assert!(err.to_string().contains("vortex.fsst"), "{err}");
        assert!(check_writable(fsst, "c", &|_: &ArrayId| true).is_ok());
    }
}
