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

//! [`DftSessionStateBuilder`] for configuring DataFusion [`SessionState`]

use color_eyre::{eyre, Result};
use datafusion::catalog::MemoryCatalogProviderList;
use datafusion::catalog::{CatalogProvider, TableProviderFactory};
use datafusion::config::ConfigExtension;
use datafusion::datasource::file_format::FileFormatFactory;
use datafusion::execution::context::SessionState;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::prelude::SessionConfig;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::ExecutionConfig;

use super::{enabled_extensions, Extension};

/// Builds a DataFusion [`SessionState`] with any necessary configuration
///
/// Wraps a [`SessionStateBuilder`] with DataFusion's default features, adding the
/// [`ExecutionConfig`] settings and `&mut` methods that let each [`Extension`] add to
/// those defaults rather than replace them.
#[derive(Debug)]
pub struct DftSessionStateBuilder {
    execution_config: Option<ExecutionConfig>,
    inner: SessionStateBuilder,
}

impl Default for DftSessionStateBuilder {
    fn default() -> Self {
        Self::new(None, SessionConfig::default().with_information_schema(true))
    }
}

impl DftSessionStateBuilder {
    fn new(execution_config: Option<ExecutionConfig>, session_config: SessionConfig) -> Self {
        Self {
            execution_config,
            inner: SessionStateBuilder::new()
                .with_default_features()
                .with_config(session_config),
        }
    }

    /// Create a new builder
    pub fn try_new(config: Option<ExecutionConfig>) -> Result<Self> {
        let session_config = if let Some(cfg) = config.clone().unwrap_or_default().datafusion {
            SessionConfig::from_string_hash_map(&cfg)?.with_information_schema(true)
        } else {
            SessionConfig::default().with_information_schema(true)
        };
        Ok(Self::new(config, session_config))
    }

    /// Add a table factory to the list of factories on this builder
    pub fn add_table_factory(&mut self, name: &str, factory: Arc<dyn TableProviderFactory>) {
        self.inner
            .table_factories()
            .get_or_insert_with(HashMap::new)
            .insert(name.to_string(), factory);
    }

    /// Add a file format factory to the list of file format factories on this builder
    pub fn add_file_format_factory(&mut self, factory: Arc<dyn FileFormatFactory>) {
        self.inner
            .file_formats()
            .get_or_insert_with(Vec::new)
            .push(factory);
    }

    /// Register a configuration extension, so its options can be changed with
    /// `SET <prefix>.<option>` and from the `datafusion` table of the execution config.
    pub fn add_config_extension<T: ConfigExtension>(
        &mut self,
        extension: T,
    ) -> datafusion::common::Result<()> {
        let options = self
            .inner
            .config()
            .get_or_insert_with(SessionConfig::default)
            .options_mut();
        options.extensions.insert(extension);

        // `SessionConfig::from_string_hash_map` only reads built-in options, so this extension's
        // keys are applied here, once its namespace exists.
        let prefix = format!("{}.", T::PREFIX);
        if let Some(settings) = self
            .execution_config
            .as_ref()
            .and_then(|config| config.datafusion.as_ref())
        {
            for (key, value) in settings.iter().filter(|(key, _)| key.starts_with(&prefix)) {
                options.set(key, value)?;
            }
        }
        Ok(())
    }

    /// Keep `extension` on the session, for whatever the session runs to get with
    /// `SessionConfig::get_extension`
    pub fn add_session_extension<T: Send + Sync + 'static>(&mut self, extension: Arc<T>) {
        self.inner
            .config()
            .get_or_insert_with(SessionConfig::default)
            .set_extension(extension);
    }

    /// Add a catalog provider to the list of providers on this builder
    pub fn add_catalog_provider(&mut self, name: &str, factory: Arc<dyn CatalogProvider>) {
        self.inner
            .catalog_list()
            .get_or_insert_with(|| Arc::new(MemoryCatalogProviderList::new()))
            .register_catalog(name.to_string(), factory);
    }

    /// Add a physical optimizer rule that runs after DataFusion's default rules
    pub fn add_physical_optimizer_rule(
        &mut self,
        rule: Arc<dyn PhysicalOptimizerRule + Send + Sync>,
    ) {
        self.inner
            .physical_optimizer_rules()
            .get_or_insert_with(Vec::new)
            .push(rule);
    }

    /// Return the current [`RuntimeEnv`], creating a default if it doesn't exist
    pub fn runtime_env(&mut self) -> &RuntimeEnv {
        self.inner.runtime_env().get_or_insert_with(Arc::default)
    }

    pub async fn register_extension(
        &mut self,
        config: ExecutionConfig,
        extension: Arc<dyn Extension>,
    ) -> color_eyre::Result<()> {
        extension
            .register(config, self)
            .await
            .map_err(|e| eyre::eyre!("Failed to register extension: {e}"))
    }

    pub async fn with_extensions(mut self) -> color_eyre::Result<Self> {
        let extensions = enabled_extensions();

        for extension in extensions {
            let execution_config = self.execution_config.clone();
            self.register_extension(execution_config.unwrap_or_default(), extension)
                .await?;
        }

        Ok(self)
    }

    /// Build the [`SessionState`] from the specified configuration
    pub fn build(self) -> datafusion::common::Result<SessionState> {
        Ok(self.inner.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::catalog::MemoryCatalogProvider;
    use datafusion::datasource::provider::DefaultTableFactory;

    #[test]
    fn added_table_factories_keep_the_defaults() {
        let mut builder = DftSessionStateBuilder::try_new(None).unwrap();
        builder.add_table_factory("EXTRA", Arc::new(DefaultTableFactory::new()));
        let state = builder.build().unwrap();

        let factories = state.table_factories();
        assert!(factories.contains_key("EXTRA"));
        assert!(factories.contains_key("CSV"));
    }

    #[test]
    fn added_catalogs_keep_the_default_catalog() {
        let mut builder = DftSessionStateBuilder::try_new(None).unwrap();
        builder.add_catalog_provider("extra", Arc::new(MemoryCatalogProvider::new()));
        let state = builder.build().unwrap();

        let catalogs = state.catalog_list().catalog_names();
        assert!(catalogs.contains(&"extra".to_string()));
        assert!(catalogs.contains(&"datafusion".to_string()));
    }

    #[test]
    fn default_file_formats_are_registered() {
        let state = DftSessionStateBuilder::try_new(None)
            .unwrap()
            .build()
            .unwrap();

        for ext in ["parquet", "csv", "json", "arrow"] {
            assert!(
                state.get_file_format_factory(ext).is_some(),
                "missing file format: {ext}"
            );
        }
    }
}
