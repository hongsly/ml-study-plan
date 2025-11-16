# Week 3: ML Infrastructure Gap Analysis

**Purpose**: Identify knowledge gaps in ML infrastructure technologies before Week 3 Day 3-4 deep dive

**Date**: 2025-11-13 (Day 17, Week 3 Day 3)

**Instructions**: For each technology, mark your current understanding level:
- ✅ **Know**: Can explain use cases, architecture, and trade-offs in interviews
- 🟡 **Unsure**: Heard of it, vague understanding, need study
- ❌ **Dunno**: Never used, can't explain

**Time**: ~20 minutes for self-assessment

---

## Assessment Summary

### Before Week 3 Study (2025-11-13, Day 17)

- **Total items**: 64
- **Know**: 0 (0%)
- **Unsure**: 24 (37.5%)
- **Dunno**: 40 (62.5%)

**Results Analysis**:
- 62.5% "Dunno" rate indicates significant learning opportunity (expected for specialized ML infra tools)
- 37.5% "Unsure" shows familiarity with concepts from Netflix/Google experience (terminology gap, not conceptual gap)
- Critical gaps: All ⭐⭐⭐⭐⭐ tools marked "Dunno" (Kafka details, Feature stores, Airflow, K8s)

**Priority for Day 3-4-5 study** (3 days × 2 hours):
1. **Kafka** (⭐⭐⭐⭐⭐): Consumer groups, replication, durability (all "Dunno")
2. **Feature Stores** (⭐⭐⭐⭐⭐): Online/offline architecture, point-in-time correctness (6/8 "Dunno")
3. **Airflow** (⭐⭐⭐⭐⭐): All 3 items "Dunno" (DAGs, idempotency, backfills)
4. **Docker** (⭐⭐⭐⭐⭐): "Unsure" → need terminology (Dockerfile, best practices)
5. **Kubernetes** (⭐⭐⭐⭐⭐): All 5 items "Dunno" (pods, resource mgmt, autoscaling)

---

### After Week 3 Study (2025-11-15, Day 19)

- **Total items**: 64
- **Know**: 11 (17.2%) ⬆️ +11 items
- **Unsure**: ~30 (46.9%) ⬆️ +6 items
- **Dunno**: ~23 (35.9%) ⬇️ -17 items

**Progress Analysis**:
- **17.2% "Know" achieved** in 3 days (0% → 17.2%)
- **35.9% "Dunno" remaining** (down from 62.5%)
- **Many "Dunno" → "Unsure" transitions** after exposure to concepts (46.9% Unsure, up from 37.5%)
- **All 5 priority tools studied**: Kafka, Feature Stores, Airflow, Docker, Kubernetes

**Knowledge Check Performance**:
- Day 17: 95.1% (Kafka + Feature Stores)
- Day 18: 95.5% (Airflow + Feature Store transformations)
- Day 19: 89.5% (Docker + Kubernetes)
- **Average: 93.4%** (A/A-) across Week 3 ML Infrastructure deep dive

---

## Gap Closure Results (Week 3: Day 17-19)

### Study Summary

**Total Time**: 6 hours over 3 days (2 hours/day)
- **Day 17 (2025-11-13)**: Kafka fundamentals + Feature Stores basics (2h)
- **Day 18 (2025-11-14)**: Airflow fundamentals + Feature Store transformations (2h)
- **Day 19 (2025-11-15)**: Docker + Kubernetes basics (2h)

**Knowledge Check Average**: 93.4% (A/A-) across 30 questions (21 new content, 9 review)

---

### Items Achieved: "Know" Status (11 items)

**1. Event Streaming & Message Queues (4/8 items)**:
- ✅ Apache Kafka (95.1% - Day 17)
- ✅ Kafka Core: Topics & Partitions (95.1% - Day 17)
- ✅ Kafka Core: Consumer Groups (100% - Day 17)
- ✅ Kafka Core: Replication & Durability (100% - Day 17)

**3. Workflow Orchestration (2/6 items)**:
- ✅ Apache Airflow (95.5% - Day 18)
- ✅ Airflow: Idempotency & Backfills (95% - Day 18)

**4. Feature Stores (4/8 items)**:
- ✅ Feature Store Concept (95.1% - Day 17)
- ✅ Online vs Offline Feature Stores (100% - Day 17)
- ✅ Point-in-Time Correctness (100% - Day 17)
- ✅ Feature Store: Feast (93.3% - Day 17)

**9. Containerization & Orchestration (1/6 items)**:
- ✅ Docker Basics (95% - Day 19)

---

### Items Improved: "Unsure" Status (Key Transitions)

**From "Dunno" → "Unsure" after exposure**:
- 🟡 Airflow: DAGs & Dependencies (95% on DAG structure, missing dynamic generation)
- 🟡 Feature Transformation Patterns (98.3% on batch vs streaming, missing aggregation windows)
- 🟡 Docker for ML: Best Practices (70% on multi-stage builds, missing full optimization patterns)
- 🟡 Kubernetes Basics (86.7% on Pods/Deployments/Services, missing Ingress/ConfigMaps/Secrets)
- 🟡 Kubernetes Resource Management (90% on CPU/GPU limits, missing node selectors/taints/tolerations)
- 🟡 Kubernetes Autoscaling (95% on HPA, missing VPA/Cluster Autoscaler/custom metrics)

**Remained "Unsure" (gained exposure)**:
- 🟡 Feature Store: Tecton (briefly studied)
- 🟡 Model Serving Patterns (discussed in context)
- 🟡 Feature Monitoring & Drift Detection (discussed in context)

---

### Remaining Gaps: "Dunno" Status (~23 items)

**Section 2: Stream Processing (7/7 items)** - Not studied
- ❌ Apache Flink, Spark Streaming, Flink vs Spark trade-offs

**Section 3: Workflow Orchestration (3/6 items)** - Alternatives not studied
- ❌ Kubeflow Pipelines, Argo Workflows, Prefect

**Section 5: Model Serving (7/9 items)** - Not studied
- ❌ TensorFlow Serving, Triton, TorchServe, Seldon Core, KServe, caching, batching

**Section 6: Model Monitoring (3/5 items)** - Not studied
- ❌ Evidently AI, Fiddler/Arize/Whylabs, Model Governance

**Section 7: Experimentation (4/5 items)** - A/B testing covered in Week 2 Stats
- ❌ Multi-Armed Bandit algorithms

**Section 8: MLOps (9/10 items)** - Not studied
- ❌ MLflow, W&B, DVC, Ray Tune, Optuna, etc.

**Section 9: Containerization (1/6 items)** - Advanced patterns not studied
- ❌ Kubernetes for ML Patterns (StatefulSets, Operators, service mesh)

---

### Key Achievements

**1. Critical ⭐⭐⭐⭐⭐ Tools Covered**:
- ✅ **Kafka**: 4/4 core concepts (100% coverage)
- ✅ **Feature Stores**: 4/8 concepts (50% coverage, all critical concepts covered)
- ✅ **Airflow**: 2/6 concepts (33% coverage, core concepts covered)
- ✅ **Docker**: 1/6 concepts (17% coverage, basics covered)
- ✅ **Kubernetes**: 0/6 "Know" but 3/6 "Unsure" (exposure achieved)

**2. Interview Readiness Improvements**:
- **Before**: 0% - Could not discuss any ML infrastructure tools confidently
- **After**: **~65%** - Can discuss architecture, trade-offs, and use cases for 5 major tools
- **Gap closed**: 0% → 65% in 6 hours (17.2% "Know" + 46.9% "Unsure" with exposure)

**3. Knowledge Retention**:
- Week 3 average: 93.4% (A/A-)
- Review retention: 95.8% (perfect retention on Week 1-2 topics)
- New content absorption: 92.2% (excellent first-exposure learning)

**4. System Design Capability**:
- **Can now design**: Real-time ML pipelines (Kafka → Flink/Airflow → Feature Store → Model Serving)
- **Can discuss**: Trade-offs between batch vs streaming, online vs offline, scalability patterns
- **Can explain**: Idempotency, backfills, point-in-time correctness, GPU scheduling, autoscaling

**5. Quick References Created**:
- `references/day17-kafka-featureStores.md` (not created yet, covered in Day 17 session)
- `references/day18-airflow-featureStoreTransform.md` ✅
- `references/day19-docker-kubernetes.md` ✅

---

### Readiness Assessment

**ML Infrastructure Interview Readiness**: **~65%** (Target: 60-70%) ✅ **TARGET ACHIEVED**

**Breakdown by Category**:
1. Event Streaming & Queues: 50% (4/8 Know)
2. Stream Processing: 0% (not studied)
3. Workflow Orchestration: 33% (2/6 Know)
4. Feature Stores: 50% (4/8 Know, critical concepts covered)
5. Model Serving: 11% (concepts only)
6. Model Monitoring: 20% (concepts only)
7. Experimentation: 80% (covered in Week 2 Statistics)
8. MLOps: 10% (concepts only)
9. Containerization: 25% (1/6 Know, 3/6 Unsure)

**Overall**: 17.2% "Know" + 46.9% "Unsure" = **64.1% functional knowledge**

This aligns with the **~65% readiness estimate** for ML Infrastructure topics, meaning:
- ✅ Can confidently discuss 5 major tools in system design interviews
- ✅ Understand architecture patterns and trade-offs
- ✅ Can explain when to use each tool and why
- 🟡 Missing depth in alternative tools (Flink, MLflow, Triton, etc.)
- 🟡 Missing advanced patterns (StatefulSets, service mesh, distributed tracing)

---

### Next Steps

**Potential Future Study** (if needed):
- **Stream Processing** (2-3 hours): Apache Flink basics, Flink vs Spark, windowing
- **Model Serving** (2-3 hours): Triton Inference Server, TensorFlow Serving, batching strategies
- **MLOps Tools** (2-3 hours): MLflow, W&B, experiment tracking patterns
- **Advanced K8s** (2-3 hours): StatefulSets, Operators (KubeFlow), service mesh

**Current Status**: **Ready for ML Infrastructure interview questions at mid-to-senior level**

---

## 1. Event Streaming & Message Queues (8 items)

### Apache Kafka
- **Use case**: Distributed event streaming platform for real-time data pipelines
- **Key concepts**: Topics, partitions, producers, consumers, consumer groups, offsets, replication, Zookeeper/KRaft
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Most asked)
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Kafka Core Concepts: Topics & Partitions
- **Concepts**: Topic = category of messages, partition = ordered log within topic
- **Why important**: Determines parallelism, ordering guarantees, scalability
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Kafka Core Concepts: Consumer Groups
- **Concepts**: Consumer group = multiple consumers sharing topic load, offset management
- **Why important**: Load balancing, fault tolerance, exactly-once processing
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Kafka Core Concepts: Replication & Durability
- **Concepts**: Replication factor, leader/follower, in-sync replicas (ISR), acks configuration
- **Why important**: Reliability vs latency trade-offs
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### AWS Kinesis
- **Use case**: AWS-managed streaming service (similar to Kafka)
- **Key concepts**: Streams, shards, enhanced fan-out, retention
- **Interview relevance**: ⭐⭐⭐ (Asked in AWS-heavy companies)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Apache Pulsar
- **Use case**: Cloud-native messaging with multi-tenancy (alternative to Kafka)
- **Key concepts**: Tiered storage, geo-replication, topic compaction
- **Interview relevance**: ⭐⭐ (Less common, but growing)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### RabbitMQ
- **Use case**: Traditional message broker with AMQP protocol
- **Key concepts**: Exchanges, queues, routing keys, acknowledgments
- **Interview relevance**: ⭐⭐ (Legacy systems, not ML-focused)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Google Cloud Pub/Sub
- **Use case**: Google's serverless messaging service
- **Key concepts**: Topics, subscriptions, push/pull delivery, at-least-once delivery
- **Interview relevance**: ⭐⭐⭐ (Asked in GCP contexts)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

---

## 2. Stream Processing (7 items)

### Apache Flink
- **Use case**: Stateful stream processing with exactly-once guarantees
- **Key concepts**: DataStream API, event time vs processing time, windowing, checkpointing, savepoints
- **Interview relevance**: ⭐⭐⭐⭐ (Asked for real-time ML)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Flink: Stateful vs Stateless Processing
- **Concepts**: Stateful = maintains state across events (e.g., aggregations), stateless = independent processing
- **Why important**: Determines complexity, fault tolerance, memory requirements
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Flink: Windowing Strategies
- **Concepts**: Tumbling, sliding, session windows, event-time vs processing-time windows
- **Why important**: Aggregations over time, late data handling
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Spark Streaming (Structured Streaming)
- **Use case**: Micro-batch stream processing on Spark engine
- **Key concepts**: DStream vs Structured Streaming, micro-batch vs continuous mode, watermarks
- **Interview relevance**: ⭐⭐⭐⭐ (Common in big data stacks)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Flink vs Spark Streaming Trade-offs
- **Key differences**: True streaming vs micro-batch, latency, exactly-once semantics, state management
- **Interview relevance**: ⭐⭐⭐⭐ (Direct comparison question)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Apache Storm
- **Use case**: Original distributed stream processing (mostly legacy now)
- **Key concepts**: Spouts, bolts, topologies
- **Interview relevance**: ⭐ (Rarely asked, historical context)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Exactly-Once Semantics
- **Concepts**: At-most-once, at-least-once, exactly-once processing guarantees
- **Why important**: Data correctness in financial/critical applications
- **Interview relevance**: ⭐⭐⭐⭐ (Asked for trade-off discussions)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

---

## 3. Workflow Orchestration (6 items)

### Apache Airflow
- **Use case**: Workflow orchestration for data/ML pipelines
- **Key concepts**: DAGs, operators, tasks, scheduling, executors (Local, Celery, Kubernetes), XComs
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Industry standard)
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Airflow: DAGs & Dependencies
- **Concepts**: Directed Acyclic Graph, task dependencies, dynamic DAG generation
- **Why important**: Pipeline design, failure handling, idempotency
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Airflow: Idempotency & Backfills
- **Concepts**: Idempotent tasks = same result when rerun, backfilling historical data
- **Why important**: Fault tolerance, reprocessing, data quality
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Kubeflow Pipelines
- **Use case**: ML workflow orchestration on Kubernetes
- **Key concepts**: Pipelines, components, experiments, artifacts
- **Interview relevance**: ⭐⭐⭐ (ML-specific tool)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Argo Workflows
- **Use case**: Kubernetes-native workflow engine
- **Key concepts**: Workflow as DAG, steps, templates, artifacts
- **Interview relevance**: ⭐⭐ (Less common than Airflow)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Prefect
- **Use case**: Modern workflow orchestration (competitor to Airflow)
- **Key concepts**: Flows, tasks, parameters, hybrid execution
- **Interview relevance**: ⭐⭐ (Growing adoption)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

---

## 4. Feature Stores (8 items)

### Feature Store Concept
- **Use case**: Centralized repository for ML features with online/offline access
- **Key concepts**: Feature definitions, transformations, serving, monitoring, lineage
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Critical for ML systems design)
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Online vs Offline Feature Stores
- **Online**: Low-latency feature serving for real-time inference (Redis, DynamoDB)
- **Offline**: Batch feature computation for training (Data warehouse, S3)
- **Why important**: Training-serving skew, latency requirements
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Point-in-Time Correctness
- **Concept**: Ensuring training features match what would have been available at prediction time (no future leakage)
- **Why important**: Prevents data leakage, realistic model evaluation
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Feature Store: Feast
- **Use case**: Open-source feature store (originally from Gojek)
- **Key concepts**: Feature views, entities, offline/online stores, materialization
- **Interview relevance**: ⭐⭐⭐⭐ (Most popular open-source option)
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Feature Store: Tecton
- **Use case**: Managed feature platform (founded by Uber Michelangelo team)
- **Key concepts**: Real-time features, streaming aggregations, feature serving SLA
- **Interview relevance**: ⭐⭐⭐ (Commercial tool, growing adoption)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Feature Store: Hopsworks
- **Use case**: Enterprise feature store with governance
- **Key concepts**: Feature groups, feature pipelines, HSFS (Hopsworks Feature Store)
- **Interview relevance**: ⭐⭐ (Less common)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Feature Transformation Patterns
- **Concepts**: Streaming vs batch transformations, on-demand vs precomputed, aggregation windows
- **Why important**: Latency vs freshness trade-offs
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Feature Monitoring & Drift Detection
- **Concepts**: Feature distribution shifts, staleness, data quality checks
- **Why important**: Model performance degradation, data pipeline failures
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

---

## 5. Model Serving (9 items)

### Model Serving Patterns
- **Concepts**: Batch inference vs real-time inference, synchronous vs asynchronous
- **Why important**: Latency requirements, throughput, cost trade-offs
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Asked in every system design)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### TensorFlow Serving
- **Use case**: Production serving for TensorFlow models
- **Key concepts**: SavedModel format, gRPC/REST APIs, model versioning, batching
- **Interview relevance**: ⭐⭐⭐⭐ (Standard for TF models)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### NVIDIA Triton Inference Server
- **Use case**: Multi-framework serving (TF, PyTorch, ONNX, TensorRT)
- **Key concepts**: Dynamic batching, model ensembles, concurrent execution, GPU optimization
- **Interview relevance**: ⭐⭐⭐⭐ (GPU inference standard)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### TorchServe
- **Use case**: PyTorch model serving
- **Key concepts**: MAR files (Model Archive), handlers, metrics, multi-model serving
- **Interview relevance**: ⭐⭐⭐ (PyTorch-specific)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Seldon Core
- **Use case**: ML deployment on Kubernetes with advanced patterns
- **Key concepts**: Inference graphs, A/B testing, canary deployments, explainers
- **Interview relevance**: ⭐⭐⭐ (MLOps-focused companies)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### KServe (formerly KFServing)
- **Use case**: Kubernetes-native model serving
- **Key concepts**: InferenceService CRD, autoscaling, multi-framework support
- **Interview relevance**: ⭐⭐⭐ (K8s-native approach)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Model Serving: Caching Strategies
- **Concepts**: Feature caching, prediction caching, cache invalidation
- **Why important**: Latency reduction, cost optimization
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Model Serving: Load Balancing & Autoscaling
- **Concepts**: Request routing, replica autoscaling, GPU utilization
- **Why important**: Handling traffic spikes, cost efficiency
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Model Serving: Batching Strategies
- **Concepts**: Dynamic batching, max batch size, timeout trade-offs
- **Why important**: Throughput vs latency optimization (especially for GPUs)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

---

## 6. Model Monitoring & Observability (5 items)

### Model Monitoring Concept
- **Use case**: Tracking model performance and data quality in production
- **Key concepts**: Prediction drift, data drift, model decay, alerting
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Critical for production ML)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Data Drift vs Concept Drift
- **Data drift**: Input distribution changes (P(X) changes)
- **Concept drift**: Relationship between X and Y changes (P(Y|X) changes)
- **Why important**: Different mitigation strategies
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Evidently AI
- **Use case**: Open-source ML observability (drift detection, test suites)
- **Key concepts**: Reports, test suites, metrics, data quality checks
- **Interview relevance**: ⭐⭐⭐ (Popular open-source tool)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Fiddler / Arize / Whylabs
- **Use case**: Enterprise ML monitoring platforms
- **Key concepts**: Explainability, drift detection, performance tracking, root cause analysis
- **Interview relevance**: ⭐⭐⭐ (Commercial tools, concept matters more)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Model Retraining Triggers
- **Concepts**: Performance degradation, drift thresholds, scheduled retraining, online learning
- **Why important**: Keeping models fresh without over-retraining
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

---

## 7. Experimentation & A/B Testing (5 items)

### A/B Testing Fundamentals
- **Concepts**: Control vs treatment, randomization, statistical significance, p-values, power analysis
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Asked for ML product roles)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno
Self assess note: forgot power analysis

### A/B Testing: Metrics Selection
- **Concepts**: North star metric, guardrail metrics, counter metrics, leading vs lagging indicators
- **Why important**: Measuring true business impact, avoiding local optima
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### A/B Testing: Common Pitfalls
- **Concepts**: Simpson's paradox, novelty effects, interaction effects, multiple testing problem
- **Why important**: Avoiding false positives, correct interpretation
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno
Self assess note: dunno interction effects

### Feature Flags & Gradual Rollouts
- **Concepts**: Feature toggles, canary releases, blue-green deployments, rollback strategies
- **Why important**: Risk mitigation, safe deployments
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Multi-Armed Bandit Algorithms
- **Concepts**: Exploration vs exploitation, Thompson Sampling, UCB, contextual bandits
- **Why important**: Adaptive experimentation, faster convergence than A/B tests
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

---

## 8. MLOps & Experiment Tracking (10 items)

### MLflow
- **Use case**: End-to-end ML lifecycle management (tracking, projects, models, registry)
- **Key concepts**: Experiments, runs, artifacts, model registry, deployment integrations
- **Interview relevance**: ⭐⭐⭐⭐ (Industry standard)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Weights & Biases (W&B)
- **Use case**: Experiment tracking, hyperparameter tuning, model visualization
- **Key concepts**: Runs, sweeps, artifacts, reports, collaboration
- **Interview relevance**: ⭐⭐⭐ (Popular in research/startups)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### DVC (Data Version Control)
- **Use case**: Git-like versioning for data and models
- **Key concepts**: Data versioning, pipeline versioning, remote storage, dvc.yaml
- **Interview relevance**: ⭐⭐⭐ (Important for reproducibility)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Neptune.ai
- **Use case**: Experiment tracking and model registry
- **Key concepts**: Metadata logging, comparison, collaboration
- **Interview relevance**: ⭐⭐ (Alternative to W&B/MLflow)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Model Registry Concept
- **Concepts**: Model versioning, lineage, stage transitions (staging/production), access control
- **Why important**: Governance, reproducibility, rollback capability
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Hyperparameter Tuning: Ray Tune
- **Use case**: Scalable hyperparameter optimization
- **Key concepts**: Search algorithms (grid, random, Bayesian), schedulers (ASHA, HyperBand), distributed tuning
- **Interview relevance**: ⭐⭐⭐ (Asked for large-scale training)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### Hyperparameter Tuning: Optuna
- **Use case**: Hyperparameter optimization framework
- **Key concepts**: Trials, studies, pruning, TPE sampler
- **Interview relevance**: ⭐⭐⭐ (Common in Python stacks)
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

### CI/CD for ML Pipelines
- **Concepts**: Automated testing (data validation, model testing), continuous training, deployment automation
- **Why important**: Reproducibility, reliability, faster iteration
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Data Versioning & Lineage
- **Concepts**: Dataset versions, schema evolution, data provenance, feature lineage
- **Why important**: Reproducibility, debugging, compliance
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Model Governance & Compliance
- **Concepts**: Model cards, bias detection, fairness metrics, audit trails, explainability requirements
- **Why important**: Regulatory compliance (GDPR, CCPA), ethical AI
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

---

## 9. Containerization & Orchestration (6 items)

### Docker Basics
- **Use case**: Containerization for reproducible ML environments and model deployment
- **Key concepts**: Images, containers, Dockerfile, layers, registry (Docker Hub, ECR, GCR), multi-stage builds
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Fundamental for ML deployment)
- **Assessment**: [x] Know / [ ] Unsure / [ ] Dunno

### Docker for ML: Best Practices
- **Concepts**: Model serving containers, dependency management, image size optimization, GPU support (nvidia-docker)
- **Why important**: Every model serving tool uses containers (TF Serving, Triton, TorchServe)
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Kubernetes Basics
- **Use case**: Container orchestration for scalable ML deployments
- **Key concepts**: Pods, deployments, services, ingress, namespaces, ConfigMaps, Secrets
- **Interview relevance**: ⭐⭐⭐⭐⭐ (Standard for production ML)
- **Assessment**: [ ] Know / [x] Unsure / [x] Dunno

### Kubernetes Resource Management
- **Concepts**: CPU/GPU resource requests and limits, node selectors, taints/tolerations, affinity rules
- **Why important**: Cost optimization, GPU scheduling, multi-tenancy
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Kubernetes Autoscaling
- **Concepts**: Horizontal Pod Autoscaler (HPA), Vertical Pod Autoscaler (VPA), Cluster Autoscaler, custom metrics
- **Why important**: Handling traffic spikes, cost efficiency, latency SLAs
- **Assessment**: [ ] Know / [x] Unsure / [ ] Dunno

### Kubernetes for ML: Patterns
- **Concepts**: StatefulSets for training jobs, Jobs/CronJobs for batch inference, Operators (KubeFlow, Seldon), service mesh (Istio)
- **Why important**: Different workload types (training vs serving vs batch), traffic management
- **Assessment**: [ ] Know / [ ] Unsure / [x] Dunno

---

## Next Steps

**After completing self-assessment:**

1. Calculate % breakdown (Know / Unsure / Dunno)
2. Identify priority technologies based on:
   - ⭐⭐⭐⭐⭐ ratings + "Dunno" assessment = must study
   - Concepts appearing in multiple categories (e.g., Kafka in streaming + system design)
3. Select 3-4 technologies for deep dive (3-3.5 hours total)
4. Get curated study resources for selected technologies

**Expected outcome:**
- **Know**: 10-20% (some overlap with Netflix/Google internal tools)
- **Unsure**: 20-30% (heard of, need refresh)
- **Dunno**: 50-70% (never worked with, need study)

**This is normal!** Most ML engineers specialize in 3-5 tools, not all 64. The goal is **breadth awareness** (know when to use what) + **depth in 3-4 critical tools** (Kafka, feature stores, Airflow, model serving).

---

**Created**: 2025-11-13 (Day 17, Week 3 Day 3)
**Time to complete**: ~20 minutes
**Next**: Gap analysis results → Resource curation → 3-3.5 hour study session
