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

//! [`CassandraExec`]: execution plan that streams a CQL `SELECT` as record
//! batches, using the driver's transparent paging

use std::sync::Arc;

use datafusion::arrow::{array::RecordBatch, datatypes::SchemaRef};
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    stream::RecordBatchReceiverStream,
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::StreamExt;
use scylla::client::session::Session;
use scylla::value::Row;
use tokio::sync::mpsc::Sender;

use crate::types::rows_to_batch;

/// Number of rows accumulated before a batch is emitted
const BATCH_SIZE: usize = 1024;

/// Execution plan that runs a CQL `SELECT` on execute
#[derive(Debug)]
pub(crate) struct CassandraExec {
    session: Arc<Session>,
    cql: String,
    /// Schema of the produced batches (after projection)
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl CassandraExec {
    pub(crate) fn new(session: Arc<Session>, cql: String, projected_schema: SchemaRef) -> Self {
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projected_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            session,
            cql,
            projected_schema,
            cache,
        }
    }
}

impl DisplayAs for CassandraExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CassandraExec: cql={}", self.cql)
    }
}

impl ExecutionPlan for CassandraExec {
    fn name(&self) -> &str {
        "CassandraExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // This is a leaf node and has no children
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            internal_err!("Children cannot be replaced in {self:?}")
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("CassandraExec has a single partition, got {partition}");
        }
        let mut builder = RecordBatchReceiverStream::builder(Arc::clone(&self.projected_schema), 2);
        let tx = builder.tx();
        let session = Arc::clone(&self.session);
        let cql = self.cql.clone();
        let schema = Arc::clone(&self.projected_schema);
        builder.spawn(async move { stream_query(session, cql, schema, tx).await });
        Ok(builder.build())
    }
}

fn to_external_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Runs the query with paging and sends batches of `BATCH_SIZE` rows until
/// the results end or the consumer drops the stream
async fn stream_query(
    session: Arc<Session>,
    cql: String,
    schema: SchemaRef,
    tx: Sender<Result<RecordBatch>>,
) -> Result<()> {
    let pager = session
        .query_iter(cql.as_str(), &[])
        .await
        .map_err(to_external_err)?;
    let mut rows_stream = pager.rows_stream::<Row>().map_err(to_external_err)?;
    let mut rows: Vec<Row> = Vec::with_capacity(BATCH_SIZE);
    while let Some(row) = rows_stream.next().await {
        rows.push(row.map_err(to_external_err)?);
        if rows.len() >= BATCH_SIZE {
            let batch = rows_to_batch(&schema, std::mem::take(&mut rows))?;
            if tx.send(Ok(batch)).await.is_err() {
                // Consumer dropped the stream
                return Ok(());
            }
        }
    }
    if !rows.is_empty() {
        let batch = rows_to_batch(&schema, rows)?;
        // If the consumer already dropped the stream the send is a no-op
        let _ = tx.send(Ok(batch)).await;
    }
    Ok(())
}
