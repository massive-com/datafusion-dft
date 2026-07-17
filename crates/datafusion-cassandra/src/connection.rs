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

//! Building a Cassandra [`Session`] from connection options

use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result};
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use scylla::errors::TranslationError;
use scylla::policies::address_translator::{AddressTranslator, UntranslatedPeer};

/// Options for connecting to a Cassandra cluster
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CassandraConnectionOptions {
    /// Contact points, each as `host` or `host:port` (default port 9042)
    pub contact_points: Vec<String>,
    /// Username for `PasswordAuthenticator` clusters
    pub user: Option<String>,
    /// Password for `PasswordAuthenticator` clusters
    pub password: Option<String>,
    /// Per-node connection timeout in milliseconds (driver default if unset)
    pub connection_timeout_ms: Option<u64>,
    /// Route every node connection through the first contact point instead of
    /// the addresses the nodes advertise. Needed when the cluster is reached
    /// through a tunnel (`kubectl port-forward`, SSH tunnel, NAT, ...) and the
    /// advertised addresses are not reachable from this machine.
    pub single_endpoint: bool,
}

/// Translates every node address to one fixed endpoint, for clusters reached
/// through a tunnel where advertised node addresses are unreachable
#[derive(Debug)]
struct FixedAddressTranslator {
    address: SocketAddr,
}

#[async_trait]
impl AddressTranslator for FixedAddressTranslator {
    async fn translate_address(
        &self,
        _untranslated_peer: &UntranslatedPeer,
    ) -> std::result::Result<SocketAddr, TranslationError> {
        Ok(self.address)
    }
}

/// Resolves a `host` or `host:port` contact point to a socket address,
/// defaulting to port 9042
fn resolve_contact_point(contact_point: &str) -> Result<SocketAddr> {
    let with_port = if contact_point.contains(':') {
        contact_point.to_string()
    } else {
        format!("{contact_point}:9042")
    };
    with_port
        .to_socket_addrs()
        .map_err(|e| {
            DataFusionError::Configuration(format!(
                "Failed to resolve Cassandra contact point '{contact_point}': {e}"
            ))
        })?
        .next()
        .ok_or_else(|| {
            DataFusionError::Configuration(format!(
                "Cassandra contact point '{contact_point}' resolved to no addresses"
            ))
        })
}

/// Connects to a Cassandra cluster and fetches its schema metadata
pub async fn connect(options: &CassandraConnectionOptions) -> Result<Session> {
    if options.contact_points.is_empty() {
        return Err(DataFusionError::Configuration(
            "Cassandra connection requires at least one contact point".to_string(),
        ));
    }
    let mut builder = SessionBuilder::new().known_nodes(&options.contact_points);
    if let (Some(user), Some(password)) = (&options.user, &options.password) {
        builder = builder.user(user, password);
    }
    if let Some(timeout_ms) = options.connection_timeout_ms {
        builder = builder.connection_timeout(Duration::from_millis(timeout_ms));
    }
    if options.single_endpoint {
        let address = resolve_contact_point(&options.contact_points[0])?;
        builder = builder.address_translator(Arc::new(FixedAddressTranslator { address }));
    }
    builder
        .build()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_contact_point_with_port() {
        let addr = resolve_contact_point("127.0.0.1:9043").unwrap();
        assert_eq!(addr.port(), 9043);
    }

    #[test]
    fn test_resolve_contact_point_default_port() {
        let addr = resolve_contact_point("127.0.0.1").unwrap();
        assert_eq!(addr.port(), 9042);
    }

    #[test]
    fn test_resolve_contact_point_invalid() {
        assert!(resolve_contact_point("no-such-host.invalid:9042").is_err());
    }
}
