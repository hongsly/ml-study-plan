# Day 18 Quick Reference: Airflow & Feature Store Transformations

**Study Date**: 2025-11-14 (Week 3, Day 4)
**Topics**: Apache Airflow fundamentals (DAGs, idempotency, executors, catchup), Feature Store transformation patterns (streaming vs batch)
**Knowledge Check Score**: 95.5% (A+)

---

## Apache Airflow Fundamentals

### DAG (Directed Acyclic Graph)

**Definition**: A collection of tasks with dependencies, no cycles allowed

**Task Dependencies**:
```python
# Method 1: Bitshift operators
task1 >> task2 >> task3  # Sequential
task1 >> [task2, task3]  # Parallel (both depend on task1)

# Method 2: set_downstream/set_upstream
task1.set_downstream(task2)
```

**Key Properties**:
- **Acyclic**: No circular dependencies
- **Directed**: Clear upstream → downstream flow
- **Schedule**: Defined by `schedule_interval` (cron expression or timedelta)

---

### Idempotency in Airflow

**Definition**: A task produces identical results when run multiple times with the same inputs

**Why Critical**:
1. **Retries**: Tasks may fail and automatically retry
2. **Backfills**: `catchup=True` reruns tasks for historical dates
3. **Manual reruns**: Operators may clear and rerun tasks to fix issues

**Best Practices** (from Airflow docs):
- ❌ **Don't use**: `datetime.now()` inside tasks (non-deterministic)
- ✅ **Do use**: Templated date variables (`{{ data_interval_start }}`, `{{ data_interval_end }}`)
- ✅ **Use UPSERT** instead of INSERT to prevent duplicates
- ✅ **Partition by date** using Airflow's logical date
- ✅ **Read from fixed data sources**, not "latest available"

**Example**:
```python
# ❌ Non-idempotent (duplicates on retry)
sql = "INSERT INTO table VALUES (...)"

# ✅ Idempotent (same result every time)
sql = """
DELETE FROM table WHERE date = '{{ data_interval_start.strftime('%Y-%m-%d') }}'
INSERT INTO table SELECT ... WHERE date = '{{ data_interval_start.strftime('%Y-%m-%d') }}'
"""

# ✅ Also idempotent (UPSERT pattern)
sql = """
MERGE INTO table USING source ON (table.id = source.id AND table.date = '{{ ds }}')
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...
"""
```

**Philosophy**: Treat tasks like database transactions - complete all work or fail entirely, never produce partial results.

---

### Template Variables (Modern Airflow 2.2+)

**Preferred variables** (have real-world semantics):
- `{{ data_interval_start }}` - Start of data interval (datetime object)
- `{{ data_interval_end }}` - End of data interval (datetime object)

**Legacy variables** (still supported):
- `{{ ds }}` - Logical date as `YYYY-MM-DD`
- `{{ execution_date }}` - Legacy name for logical_date
- `{{ ts }}` - ISO format timestamp

**Usage**:
```python
# In PythonOperator
def process_data(**context):
    start = context['data_interval_start']
    print(f"Processing data from {start}")

# In SQL (Jinja templating)
sql = """
SELECT * FROM events
WHERE event_time >= '{{ data_interval_start }}'
  AND event_time < '{{ data_interval_end }}'
"""
```

---

### Catchup and Backfills

**`catchup` parameter**:
- `catchup=True` (default): Run all missed DAG runs between `start_date` and now
- `catchup=False`: Only run the latest DAG run

**When to use each**:
- **catchup=True**: Need to process all data window-by-window (e.g., daily ETL for historical dates)
- **catchup=False**: Only latest execution matters (e.g., real-time dashboard update)

**Key dates**:
- `start_date`: When the DAG is established (first possible run)
- `execution_date` (now `logical_date`): Logical start of the data interval for a DAG run
- Actual run time: When the scheduler actually executes the task

**Example**:
```python
# Scenario: DAG with start_date=2025-11-01, schedule_interval=daily
# Today is 2025-11-14, DAG just created

# catchup=True → Runs 13 DAG runs (2025-11-01 through 2025-11-13)
# catchup=False → Runs only 1 DAG run (2025-11-13)
```

---

### Airflow Executors

| Executor | Architecture | Use Case | Pros | Cons |
|----------|--------------|----------|------|------|
| **LocalExecutor** | Tasks run on same machine as scheduler | Development, simple/fast tasks | Simple setup, no extra infrastructure | Limited scalability, competes with scheduler for resources |
| **CeleryExecutor** | Tasks sent to pool of always-on workers | Production batch/intensive tasks | Efficient, workers always ready, good for steady load | Costly if idle frequently, tasks compete for resources |
| **KubernetesExecutor** | Each task runs in isolated pod | Sporadic tasks with variable resource needs | Independent resources per task, cost-effective, fault isolation | Higher latency (pod startup ~10-30s), more complex setup |

**Decision framework**:
- **Fast/simple tasks** → LocalExecutor
- **Steady batch workload** → CeleryExecutor
- **Variable/sporadic workload** → KubernetesExecutor
- **GPU/special resources** → KubernetesExecutor (can request specific resources per task)

---

## Feature Store Transformations

### Batch vs Streaming Comparison

| Aspect | Batch | Streaming |
|--------|-------|-----------|
| **Freshness** | Hours to days | Seconds to minutes |
| **Data Volume** | Large (90 days, full scans) | Small (last 5 min, incremental) |
| **Complexity** | Can handle complex aggregations | Limited by streaming constraints |
| **Cost** | Lower (scheduled compute) | Higher (always running) |
| **Latency Sensitivity** | Low | High |
| **Tools** | Airflow + Spark, dbt | Flink, Spark Structured Streaming, Kafka Streams |

---

### When to Use Batch Transformations

**Use cases**:
- Features that don't change frequently (demographics, historical stats)
- Complex aggregations requiring full dataset scans
- Long time windows (30-day avg, 90-day stats)
- Features computed daily/hourly

**Examples**:
- "User's average purchase amount in last 90 days"
- "Product popularity rank (updated daily)"
- "User's lifetime value score"
- "Merchant's historical transaction patterns"

**Architecture**:
```
Raw Data (S3/Warehouse)
  ↓ (Airflow scheduled job, daily/hourly)
Spark/dbt transformation
  ↓
Offline Store (Snowflake/BigQuery)
  ↓ (Materialization job)
Online Store (Redis/DynamoDB)
```

**Tools**:
- **Orchestration**: Airflow
- **Compute**: Spark, dbt, Pandas (small data)
- **Storage**: S3 → Snowflake/BigQuery → Redis/DynamoDB

---

### When to Use Streaming Transformations

**Use cases**:
- Real-time fraud detection (transaction patterns)
- Session-based features (clicks in last 10 min)
- Event-driven features (user just added to cart)
- Real-time recommendation updates

**Examples**:
- "Number of transactions in last 5 minutes" (fraud detection)
- "Items in cart right now" (e-commerce)
- "User's last 10 clicks" (recommendation)
- "Real-time velocity check" (fintech)

**Architecture**:
```
Event Stream (Kafka)
  ↓ (Real-time processing)
Flink/Spark Streaming transformation
  ↓ (Direct write)
Online Store (Redis/DynamoDB)
```

**Tools**:
- **Streaming platform**: Kafka, Kinesis, Pulsar
- **Compute**: Flink, Spark Structured Streaming, Kafka Streams
- **Storage**: Write directly to Online Store (bypass offline store)

**Trade-offs**:
- ✅ Low latency (seconds)
- ✅ Always up-to-date
- ❌ Higher cost (always running)
- ❌ More complex (stateful stream processing, windowing)
- ❌ Limited aggregation complexity

---

### Hybrid Approach (Most Common in Practice)

**Pattern**: Combine batch and streaming for different feature types

**Example - E-commerce recommendation**:
- **Batch features** (updated daily):
  - "User's 30-day purchase history"
  - "Product popularity rank"
  - "User's category preferences (aggregated weekly)"
- **Streaming features** (real-time):
  - "Items in cart right now"
  - "Last 5 page views"
  - "Current session length"
- **Serving**: Feature store serves both types together at inference time

**Benefits**:
- Cost-effective: Use batch for most features (95%+), streaming only when needed
- Flexibility: Can add real-time features without recomputing batch features
- Performance: Best of both worlds

---

## Feast Architecture Recap (Day 17 + 18)

### Components

1. **Offline Store**: Historical data for training (Snowflake, BigQuery, S3)
   - Large volume, high latency (seconds to minutes)
   - Used for: Training data generation, batch scoring

2. **Online Store**: Low-latency serving (Redis, DynamoDB, Cassandra)
   - Small volume, low latency (< 10ms)
   - Used for: Real-time inference API

3. **Registry**: Feature metadata (definitions, schemas, lineage)

4. **Feature Server**: Optional REST API for non-Python clients

### Materialization Process

**Definition**: Moving features from Offline Store → Online Store

**Why needed**: Offline store too slow for real-time inference (seconds vs milliseconds)

**How it works**:
```
1. Define feature view (schema, sources, transformations)
2. Feast CLI: `feast materialize-incremental`
3. Reads latest features from Offline Store
4. Writes to Online Store (with TTL for freshness)
5. Now available for low-latency serving
```

**Frequency**: Typically hourly or daily (batch features)

**Limitation**: Feast doesn't natively handle streaming - for real-time features, write directly to Online Store via Flink/Kafka Streams

---

## Interview Q&A

### Q: "Explain idempotency in Airflow. Why is it important?"

**A**: "Idempotency means a task produces identical results when run multiple times with the same inputs. This is critical in Airflow because:

1. **Retries**: Tasks may fail and automatically retry
2. **Backfills**: When `catchup=True` or manual backfills, tasks rerun for historical dates
3. **Manual reruns**: Operators may clear and rerun tasks to fix issues

**How to achieve it**:
- Use Airflow's templated date variables (`data_interval_start`) instead of `datetime.now()`
- Use UPSERT or DELETE+INSERT instead of INSERT
- Partition data by date using logical date
- Treat tasks like database transactions - complete all work or fail entirely

**Example**: Instead of `INSERT INTO table VALUES (...)`, use:
```sql
DELETE FROM table WHERE date = '{{ data_interval_start }}'
INSERT INTO table SELECT ... WHERE date = '{{ data_interval_start }}'
```
This ensures rerunning the task produces the same result."

---

### Q: "When would you use streaming vs batch feature transformations?"

**A**: "It depends on data freshness requirements and cost trade-offs:

**Batch transformations** (via Airflow + Spark):
- Use when features update daily/hourly (e.g., 'user's 30-day purchase count')
- Long time windows requiring large aggregations
- More cost-effective - compute runs on schedule, not continuously
- Good for complex calculations requiring full dataset scans

**Streaming transformations** (via Flink/Kafka Streams):
- Use when you need real-time features (e.g., 'clicks in last 5 minutes' for fraud detection)
- Short time windows with small data volumes
- Higher cost - always-on infrastructure
- Critical for event-driven ML (fraud detection, real-time recommendations)

**In practice**, most systems use both:
- Batch for slowly-changing features (demographics, historical stats) - 90%+ of features
- Streaming for session-based or event-driven features - ~5-10% of features
- Feature store serves both types together at inference time

Example: E-commerce recommendation uses batch for 'user's 30-day purchase history' (computed nightly) and streaming for 'items in cart right now' (real-time)."

---

### Q: "Compare Airflow executors. When would you use KubernetesExecutor?"

**A**: "Airflow has three main executors:

**LocalExecutor**:
- Tasks run on same machine as scheduler
- Best for: Development, simple/fast tasks
- Limitation: Limited scalability, competes with scheduler for resources

**CeleryExecutor**:
- Tasks sent to pool of always-on workers
- Best for: Production batch workloads with steady traffic
- Trade-off: Efficient when utilized, but costly if workers idle frequently

**KubernetesExecutor**:
- Each task runs in isolated Kubernetes pod
- Best for: Sporadic workloads with variable resource needs
- Benefits: Independent resources per task, cost-effective (scale to zero), fault isolation
- Trade-off: Higher latency (~10-30s pod startup)

I'd use **KubernetesExecutor** when:
- Tasks have variable resource needs (some need GPUs, some need lots of memory)
- Workload is sporadic (don't want to pay for idle workers)
- Tasks need isolation (failure in one task shouldn't affect others)
- Need to dynamically scale based on load"

---

### Q: "What is catchup in Airflow? When would you set it to True vs False?"

**A**: "Catchup controls whether Airflow runs missed DAG runs for historical dates.

**catchup=True** (default):
- Runs all missed DAG runs between `start_date` and now
- Use when: Need to process all data window-by-window
- Example: Daily ETL pipeline that must process every day's data for consistency

**catchup=False**:
- Only runs the latest DAG run
- Use when: Only the latest execution matters
- Example: Real-time dashboard update, model re-training (only need latest)

**Key dates**:
- `start_date`: First possible DAG run (when DAG is established)
- `execution_date` (now `logical_date`): Logical start of data interval for each run

**Example scenario**: DAG with `start_date=2025-11-01`, `schedule_interval=daily`, created on 2025-11-14
- `catchup=True` → Runs 13 DAG runs (2025-11-01 through 2025-11-13)
- `catchup=False` → Runs only 1 DAG run (2025-11-13)"

---

## Common Pitfalls

### Airflow
1. **Using `now()` in tasks**: Non-deterministic, breaks idempotency
   - Fix: Use `{{ data_interval_start }}` or `{{ ds }}`

2. **Forgetting to set `catchup=False`**: DAG created late runs all historical dates
   - Fix: Explicitly set `catchup=False` if only latest run matters

3. **Large task parallelism without right executor**: LocalExecutor can't handle 100s of parallel tasks
   - Fix: Use CeleryExecutor or KubernetesExecutor for high parallelism

4. **Tasks producing partial results**: Non-transactional writes can leave incomplete data
   - Fix: Write to temp location, then atomic move/rename

### Feature Store Transformations
1. **Using streaming for everything**: Expensive and complex, most features don't need real-time
   - Fix: Default to batch, only use streaming when freshness requirement < 5-10 min

2. **Not using point-in-time correctness**: Training features leak future data
   - Fix: Always query features "as of" a specific timestamp

3. **Mixing batch/streaming without clear boundaries**: Complexity explosion
   - Fix: Clear separation - batch for historical, streaming for real-time

---

## Resources Studied

### Airflow
- Airflow Official Docs: https://airflow.apache.org/docs/apache-airflow/stable/
  - Core Concepts: Tasks, DAG Runs, Executors
  - Best Practices: Idempotency section
  - Templates Reference: data_interval_start/end

### Feature Stores
- Made With ML - Feature Store Guide: https://madewithml.com/courses/mlops/feature-store/
  - Streaming vs Batch comparison
  - Architecture patterns

- Feast Official Docs: https://docs.feast.dev/
  - Components overview
  - Materialization process
  - Limitations (no native streaming)

---

## Next Steps (Day 19)

**Topics**:
- Docker basics (30 min): Dockerfile, images, containers, multi-stage builds, GPU support
- Kubernetes basics (1 hour): Pods, deployments, services, resource management, autoscaling

**Goal**: Complete ML Infrastructure week - bring readiness from ~25% → 65-70%

**Target**: Can explain containerization and orchestration at interview level
