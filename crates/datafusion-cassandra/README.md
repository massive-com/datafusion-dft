# datafusion-cassandra

A [DataFusion](https://datafusion.apache.org/) `TableProvider` for querying
[Apache Cassandra](https://cassandra.apache.org/) (and ScyllaDB) tables with
SQL, built on the [scylla](https://crates.io/crates/scylla) CQL driver.

## Usage

```rust,ignore
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_cassandra::{connect, CassandraConnectionOptions, CassandraTable};

let session = Arc::new(
    connect(&CassandraConnectionOptions {
        contact_points: vec!["localhost:9042".to_string()],
        user: Some("cassandra".to_string()),
        password: Some("cassandra".to_string()),
        ..Default::default()
    })
    .await?,
);

let ctx = SessionContext::new();
let table = CassandraTable::try_new(Arc::clone(&session), "my_keyspace", "users")?;
ctx.register_table("users", Arc::new(table))?;

let df = ctx.sql("SELECT id, name FROM users LIMIT 10").await?;
df.show().await?;
```

Keyspace and table discovery helpers are also provided:

```rust,ignore
use datafusion_cassandra::{keyspace_table_names, user_keyspace_names};

let keyspaces = user_keyspace_names(&session); // excludes system keyspaces
let tables = keyspace_table_names(&session, "my_keyspace"); // includes materialized views
```

## Type mapping

| CQL | Arrow |
|---|---|
| `ascii`, `text`, `varchar` | `Utf8` |
| `tinyint`, `smallint`, `int`, `bigint`, `counter` | `Int8` / `Int16` / `Int32` / `Int64` |
| `float`, `double` | `Float32` / `Float64` |
| `boolean` | `Boolean` |
| `blob` | `Binary` |
| `timestamp` | `Timestamp(Millisecond, UTC)` |
| `date` | `Date32` |
| `time` | `Time64(Nanosecond)` |
| `list<T>`, `set<T>`, `vector<T, N>` | `List<T>` |
| `uuid`, `timeuuid`, `inet`, `decimal`, `varint`, `duration`, `map`, `tuple`, UDT | `Utf8` (readable rendering) |

## Query pushdown

- **Projection**: only the selected columns are fetched (the generated CQL
  lists columns explicitly).
- **Limit**: `LIMIT` is pushed into the CQL query.
- **Filters**: equality predicates that together cover the *full* partition
  key are pushed down (that is the only `WHERE` shape Cassandra executes
  without `ALLOW FILTERING`). All other predicates are evaluated by
  DataFusion after the scan.

Results are streamed with the driver's transparent paging, so large tables do
not need to fit in memory on the server side.
