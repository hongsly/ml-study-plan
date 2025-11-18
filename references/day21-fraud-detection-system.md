# Day 21 Quick Reference: Fraud Detection System Design

**Study Date**: 2025-11-17 (Week 3, Day 7)
**Topics**: System design practice - Real-time fraud detection for payment processing
**Mock Interview Score**: 85/100 (A-)
**Knowledge Check Score**: 97.0% (A+)

---

## Problem Statement

Design a real-time fraud detection system for a payment processing platform (PayPal, Stripe) that:
- Detects fraudulent transactions with low latency
- Minimizes false positives (avoid blocking legitimate users)
- Handles 10,000-20,000 TPS (transactions per second)

---

## Key Requirements

### Scale & Latency
- **Traffic**: 10,000 TPS average, 20,000 TPS peak (2× during holidays)
- **Latency budget**: 100ms total (including network)
- **Model latency target**: 60-80ms

### Business Constraints
- **Fraud rate**: ~0.1% (10 TPS fraudulent out of 10K TPS)
- **False positive cost**: $5-10 per FP (manual review, customer frustration, churn)
- **False negative cost**: $100-500 per missed fraud (chargebacks, platform liability)
- **Target metric**: 99% precision (1% FP rate) at 80%+ recall

### Metrics
- **Primary**: Precision-Recall AUC
- **Threshold selection**: Pick threshold for 99% precision, measure recall
- **Alternative**: F0.5 score (weights precision 2× as recall)

---

## Model Selection & Cost Analysis ⭐⭐⭐

### Option 1: Logistic Regression
**Throughput**:
- 10,000 QPS per CPU core (simple linear model)
- Need 2 cores for 20K peak QPS

**Cost**: 1 small server (2 cores) = **$600/year**

**Pros**: Extremely fast, cheap, interpretable
**Cons**: Poor performance on complex fraud patterns (non-linear relationships)

---

### Option 2: XGBoost ⭐ **SELECTED**

**Without Dynamic Batching**:
- Latency: 5ms per prediction
- QPS per core: 1 / 0.005 = 200 QPS
- Cores needed: 20,000 / 200 = 100 cores
- Servers: 100 / 16 = 7 servers
- Cost: 7 × $3K = **$21K/year**

**With Dynamic Batching** ⭐:

**Calculation**:
- Collection time: 25ms
- Batch size: 20,000 QPS × 0.025s = **500 requests**
- Processing time: 50ms for batch of 500 (see explanation below)
- Total latency: 25ms (wait) + 50ms (process) = **75ms**
- Throughput: 500 requests / 0.075s = **6,666 QPS per core**
- Cores needed: 20,000 / 6,666 ≈ 3 cores
- Servers: 1 server (16 cores) with redundancy = **2 servers**
- Cost: 2 × $3K = **$6K/year**

**How does batch processing time increase?**
- Single XGBoost prediction: 5ms
- Batch of 500: **50ms** (10× longer for 100× more data)
- **Why not 500 × 5ms = 2,500ms?** Vectorization and CPU cache efficiency!
- XGBoost processes batches much faster than individual predictions
- Typical scaling: Processing time ≈ 10-20× longer for 100× more data (sublinear)

**Pros**: Great performance, handles non-linear patterns, interpretable, cost-effective
**Cons**: More complex than LogReg

---

### Option 3: Medium DNN

**Without Dynamic Batching**:
- Latency: 20ms per prediction
- QPS per GPU: 1 / 0.02 = 50 QPS
- GPUs needed: 20,000 / 50 = 400 GPUs
- Cost: 400 × $10K = **$4M/year** ❌ Too expensive!

**With Dynamic Batching**:

**Calculation**:
- Collection time: 6.4ms
- Batch size: 20,000 QPS × 0.0064s = **128 requests**
- Processing time: 50ms for batch of 128 (GPU optimized for larger batches)
- Total latency: 6.4ms (wait) + 50ms (process) = **56.4ms**
- Throughput: 128 requests / 0.0564s = **2,270 QPS per GPU**
- GPUs needed: 20,000 / 2,270 ≈ 9 GPUs
- Cost: 9 × $10K = **$90K/year**

**How does GPU batch processing time increase?**
- Single DNN prediction: 20ms
- Batch of 128: **50ms** (2.5× longer for 128× more data)
- **Why much better than CPU?** GPUs excel at parallel matrix operations
- Batch=128 fully utilizes GPU cores (thousands of CUDA cores working together)

**Pros**: Best performance, handles complex patterns
**Cons**: 10-15× more expensive than XGBoost, less interpretable

---

### Decision: XGBoost with Dynamic Batching

**Cost comparison**:
- XGBoost (no batching): $21K/year
- XGBoost (with batching): **$6K/year** ✅ (3.5× savings)
- DNN (with batching): $90K/year (15× more expensive)

**Rationale**: XGBoost balances performance and cost. For tabular fraud data, XGBoost typically matches DNN performance at fraction of cost.

---

## Dynamic Batching Mechanics ⭐⭐⭐

### Key Concept
**Collect incoming requests for T milliseconds, then process them together in a single batch.**

### Example (20K QPS, collect for 25ms):
```
Timeline:
0ms ────────────> 25ms ─────> 75ms
     Collect 500      Process    Return results
     requests         batch

Requests arriving: 20,000 QPS × 0.025s = 500 requests
Processing time: 50ms for batch of 500
Total latency: 25ms (wait) + 50ms (process) = 75ms ✅ (within 80ms budget!)

Throughput: 500 requests / 0.075s = 6,666 QPS per core
Servers needed: 20,000 / 6,666 = 3 cores (1 server with redundancy)
```

### Trade-offs
- ✅ **Benefit**: 50× throughput improvement (200 → 10,000 QPS per core)
- ❌ **Cost**: Latency increases from 5ms → 75ms
- ✅ **Acceptable**: 75ms < 80ms budget

### Collection Time Formula
```
Batch size = QPS × Collection time
500 = 20,000 × 0.025s

Collection time = Batch size / QPS
0.025s = 500 / 20,000
```

---

## Feature Engineering

### User Profile Features (2)
- `user_age_bucket` (e.g., 18-25, 26-35, ...)
- `user_home_location` (country, state, city)

### Aggregated Historical Features (5)
- `tx_count_by_amount_bucket` (e.g., $0-10: 50 tx, $10-100: 30 tx, ...)
- `tx_count_by_merchant` (e.g., Amazon: 100 tx, Walmart: 50 tx, ...)
- `tx_count_by_location` (e.g., home city: 200 tx, other: 10 tx)
- `tx_count_by_device` (e.g., iPhone 12: 150 tx, new laptop: 1 tx)
- `avg_amount_by_major_merchant` (e.g., Amazon avg: $50, gas stations avg: $40)

### Velocity Features (2)
- `tx_count_1min` (realtime: transactions in last 60 seconds)
- `tx_peak_velocity_24h` (max transactions per minute in last 24 hours)

### Recent History Features (1)
- `last_10_transactions` (array: [{amount, merchant, location, device, timestamp}, ...])

### Current Transaction Features (4)
- `amount`, `merchant`, `location`, `device` (from request payload)

### Mismatch/Deviation Features (4)
- `is_at_home_location` (boolean)
- `is_new_device` (boolean: device not seen in last 30 days)
- `amount_deviation_from_avg` (current_amount / user_avg_amount)
- `is_unusual_hour` (boolean: transaction at 3am when user normally transacts 9am-6pm)

### Cold Start Indicators (2)
- `is_new_user` (boolean: account < 7 days old)
- `is_first_transaction` (boolean)
- `account_age_days` (0 for new users)

**Total: 20 features** (comprehensive coverage)

---

## Data Pipeline Architecture

### Real-time Flow
```
Transaction Event → Kafka → Flink → Redis (Online Store)
                      ↓
                  Snowflake (Offline Store, batch daily)
```

### Flink's Role (Streaming)
- Reads from **Kafka** (transaction events)
- Computes **real-time velocity features**:
  - `tx_count_1min`: Count transactions in last 60 seconds (sliding window)
  - `last_10_transactions`: Append to list, keep only last 10
- Writes to **Redis** with <1 second latency

### Snowflake's Role (Batch)
- Daily batch processing (Spark/Airflow):
  - Aggregate historical features (30-day averages, counts by merchant/location/device)
  - Materialize to Redis for online serving
- Training data:
  - Point-in-time joins (no data leakage!)
  - Label delay: Use transactions 7+ days old (fraud confirmed 1-7 days after transaction)

### Feature Store (Redis) Schema

**Key-value design**:
```
Key: "user:{user_id}:features"
Value: JSON/protobuf with ALL features:
{
  "tx_count_1min": 7,
  "tx_count_24h": 42,
  "avg_amount_30d": 250.50,
  "last_10_tx": [{amount:100, merchant:"Amazon", ...}, ...],
  "home_location": "San Francisco, CA"
}

TTL: 7 days (expire old users to save memory)
```

**Benefit**: 1 Redis GET (not N GETs) → lower latency!

---

## Serving Architecture

### 3-Tier Decision Workflow

**For each transaction**:
1. **Fetch features** (~10ms):
   - Online features from Redis (velocity, last 10 tx, historical aggregates)
   - Current transaction from request payload
2. **Model inference** (~60ms with batching):
   - XGBoost ensemble (3 models voting)
   - Returns fraud probability score (0-1)
3. **Decision** (~5ms):
   - **Score > 0.9**: Auto-block (high confidence fraud)
   - **Score 0.7-0.9**: Temp-block + manual review (ambiguous)
   - **Score < 0.7**: Allow (low fraud risk)

**Total latency**: 10ms + 60ms + 5ms = **75ms** ✅

**Threshold selection**:
- **Primary threshold (0.9)**: Set for **99% precision** on validation set (1% FP rate requirement)
- **Manual review range (0.7-0.9)**: Catches edge cases with human judgment before auto-allowing
- **Why 99% precision?** Business cost analysis: $5-10 per FP × 1% FP rate is acceptable vs $100-500 per FN

### Manual Review Queue

**Volume Calculation**:
- Total traffic: 20,000 TPS
- Fraud rate: 0.1% = 20 fraud TPS
- At 99% precision threshold:
  - Auto-blocked (score >0.9): ~200 TPS (16 real fraud + 184 false positives)
  - Manual review (0.7-0.9): ~50 TPS (3 real fraud + 47 ambiguous legitimate transactions)
  - Allowed (<0.7): ~19,750 TPS (1 missed fraud + 19,749 legitimate)

**Realistic volume**: **~50 TPS to manual review queue** (0.25% of traffic)

**Prioritization**: `fraud_score × transaction_amount`
- Catch high-value fraud first (e.g., $5,000 tx with 0.75 score > $50 tx with 0.85 score)

**SLA**: Review within 2-4 hours
- Temp-block during review
- Unblock if human says "not fraud"

**Feedback loop**: Human labels → Kafka → Snowflake → Next week's retraining data

---

## Sliding vs Tumbling Windows ⭐

### Problem: Fraud at Window Edges

**Tumbling windows (non-overlapping 1-min buckets)**:
```
Window 1: [10:00:00 - 10:00:59] → 3 tx ✅ (below threshold of 5)
Window 2: [10:01:00 - 10:01:59] → 3 tx ✅ (below threshold of 5)

Reality: 6 tx in 66 seconds (fraud!), but split across windows ❌
```

**Fraudster attack pattern**:
- Make 3 transactions at 10:00:57-59 (just before window ends)
- Make 3 transactions at 10:01:01-03 (just after new window starts)
- Each window sees only 3 tx → passes threshold!

### Solution: Sliding Windows

**Sliding window (60-second window, updated every second)**:
```
At 10:00:30: Count tx from [09:59:30 to 10:00:30] → 3 tx
At 10:01:00: Count tx from [10:00:00 to 10:01:00] → 6 tx ⚠️ FRAUD!
```

**Flink implementation**:
```java
stream
  .keyBy("user_id")
  .window(SlidingEventTimeWindows.of(Time.seconds(60), Time.seconds(1)))
  .aggregate(new CountAggregator())
```

**Trade-off**:
- ✅ Catches edge cases
- ❌ Higher compute cost (update every second vs every minute)
- Cost acceptable: Flink can handle 20K TPS with sliding windows easily

---

## Cold Start Handling

### Problem
New user makes first transaction:
- Redis has **NO historical data** for this user
- No `tx_count_24h`, no `avg_amount_30d`, no `last_10_tx`

### Solution

**1. Default values**:
```python
feature_defaults = {
    "tx_count_1min": 0,
    "tx_count_24h": 0,
    "avg_amount_30d": 0,  # or use regional average
    "last_10_tx": [],
    "home_location": "unknown"
}
```

**2. Cold start indicators** (help model differentiate):
- `is_new_user`: True
- `is_first_transaction`: True
- `account_age_days`: 0

**3. Training data inclusion**:
- ✅ Include new users in training (don't filter them out!)
- Model learns: "New user + high amount + unusual merchant = higher fraud risk"

**4. Conservative scoring**:
- Option A: New users default to **manual review** (conservative)
- Option B: New users use **stricter threshold** (e.g., 0.7 instead of 0.9 for auto-block)

### Why XGBoost Handles This Well

**XGBoost**: Decision tree ensemble
- Naturally handles missing/zero values
- Learns branches like: "If `is_new_user == True` AND `amount > $500` → high fraud risk"

**Neural networks**: Require special handling
- Need embeddings for missing values
- Need careful imputation strategy
- More complex to train with sparse features

---

## Training Data Pipeline

### Label Delay Problem

**Reality**: Fraud confirmed 1-7 days after transaction (chargebacks, user reports)

**Solution**: Point-in-time correctness
```
Transaction at T0 (2025-01-01 10:00:00)
Labels arrive at T0 + 7 days

Training pipeline:
1. Only use transactions >= 7 days old (labels confirmed)
2. Query Snowflake: "Give me user's features AS OF T0"
3. Point-in-time join: SELECT * FROM features WHERE timestamp <= T0
4. This ensures no data leakage!
```

**Example query**:
```sql
SELECT
  t.user_id,
  t.amount,
  t.merchant,
  t.is_fraud,  -- Label arrived 7 days later
  f.tx_count_24h AS tx_count_24h_at_transaction_time,
  f.avg_amount_30d AS avg_at_transaction_time
FROM transactions t
LEFT JOIN features f
  ON t.user_id = f.user_id
  AND f.snapshot_timestamp <= t.transaction_timestamp
WHERE t.transaction_timestamp <= CURRENT_DATE - INTERVAL '7 days'
```

### Class Imbalance

**Problem**: Only 0.1% fraud (1,000 fraud vs 999,000 legit in 1M transactions)

**Solutions**:

**Option A: Class Weights Only (Correct Probabilities)** ⭐ Recommended if you have compute
```python
scale_pos_weight = 999  # Original ratio: 999K non-fraud / 1K fraud
# Train on full 1M samples
# Model learns correct P(fraud|X) for 1:999 distribution
```
✅ Probabilities are correctly calibrated
❌ Slow training (1M samples), high memory usage

---

**Option B: Downsampling + Threshold Tuning (Fast Training)** ⭐ Most common in production
```python
# Step 1: Downsample non-fraud by 10× (sample 10%)
# New training data: 1,000 fraud vs 99,900 non-fraud (1:100 ratio)

scale_pos_weight = 100  # Match training data ratio (99,900 / 1,000)

# Step 2: Train on 100K samples
# Model learns P(fraud|X) for 1:100 distribution (NOT 1:999!)

# Step 3: Pick threshold on ORIGINAL validation set (1:999 ratio)
# Validation set has full 1:999 distribution
# Find threshold that gives 99% precision on validation
# This accounts for the distribution shift automatically
```
✅ 10× faster training
✅ Lower memory
✅ No manual probability recalibration needed
❌ Probabilities will be ~10× too high (but threshold selection fixes this)

---

**Option C: Downsampling + Probability Recalibration** (If you need calibrated probabilities)
```python
# Step 1-2: Same as Option B (downsample + train with scale_pos_weight=100)

# Step 3: At inference, recalibrate probabilities
adjusted_prob = raw_prob * (100 / 999)  # Adjust for true prior (1:999 vs 1:100)
```
✅ Fast training + calibrated probabilities
❌ Extra step at inference time

---

**Why does downsampling shift probabilities?**
- Model trained on 1:100 data learns P(fraud|X) ≈ 0.01 for "typical" fraud patterns
- But in production (1:999 data), true P(fraud|X) ≈ 0.001 for same patterns
- Probabilities are **10× too high** because model saw fraud 10× more often during training
- **Fix**: Either recalibrate OR pick threshold on validation set with correct distribution

**Is this true for all models?**
- ✅ **Yes for**: Logistic regression, XGBoost, neural networks (any model trained with weighted loss)
- ❌ **No for**: K-NN (doesn't use loss weights), unweighted decision trees
- **Key**: Model must support `sample_weight` parameter in training

---

## Model Deployment & A/B Testing

### Offline Evaluation
1. **Holdout test set**: 20% of data (ensure 7+ days old for labels)
2. **Metrics**: Precision-Recall AUC, precision at 99% (FP rate 1%)
3. **Threshold selection**: Pick threshold for 99% precision on holdout

### Shadow Deployment
1. Deploy new model alongside production model
2. Score all transactions with BOTH models
3. Don't affect user experience (use production model decisions)
4. After 7 days: Retrospectively compare precision/recall
5. If new model >= baseline: Proceed to A/B test

### A/B Test Design

**Hypothesis**: New model improves recall from 80% → 82% at 99% precision

**Sample size calculation** (for 2% lift):
```
Effect size: 2% recall lift (80% → 82%)
Baseline fraud rate: 0.1% (10 frauds per 10K transactions)
Alpha: 0.05, Power: 0.8

Required sample: ~500K transactions per group
At 10K TPS: 500K / 10K = 50,000 seconds = 14 hours per group
Test duration: 2-3 days (to account for day/night patterns)
```

**Randomization**: Per-transaction (not per-user)
- 50% traffic → Control (old model)
- 50% traffic → Treatment (new model)

**Metrics**:
- **Primary**: Recall at 99% precision
- **Secondary**: User satisfaction (blocked users who contact support), transaction volume

**Decision criteria**:
- p-value < 0.05 AND recall lift >= 1% → Ship
- Otherwise: Iterate on model

---

## Monitoring & Observability

### Short-term Metrics (Real-time dashboards)
- **QPS**: Current transaction rate (10K-20K TPS)
- **Latency p50/p95/p99**: Model inference time (target <80ms)
- **Fraud rate**: Current fraud detection rate (~0.1%)
- **Block rate**: % transactions blocked (should be ~1% for 1% FP rate)
- **Manual review queue depth**: # transactions awaiting review

### Mid-term Metrics (Daily reports)
- **Precision**: % of blocked transactions that were actually fraud (target 99%)
- **Recall**: % of frauds caught (target 80%+)
- **False positive rate**: % of legit transactions blocked (target <1%)
- **User satisfaction**: Support tickets from blocked users
- **Revenue impact**: Fraud prevented vs. lost transactions (false positives)

### Long-term Metrics (Weekly/monthly)
- **Data drift**: Feature distributions change (e.g., avg transaction amount increases)
- **Model drift**: Precision/recall degrade over time (fraudsters adapt)
- **Business metrics**: Total fraud loss, chargeback rate, user retention

### Alerting Thresholds
- **Critical**: Precision < 98% (too many false positives)
- **Warning**: Recall < 75% (missing too much fraud)
- **Info**: Latency p99 > 100ms (approaching budget)

### Feature Health Checks
- **Redis latency**: p99 < 5ms (if higher, check Redis cluster)
- **Flink lag**: <1 second (if higher, scale up Flink workers)
- **Feature staleness**: Check `tx_count_1min` timestamp (should be < 2 seconds old)

---

## Common Pitfalls & Lessons

### Pitfall 1: Confusing Flink and Snowflake Roles ❌

**Wrong**: "Flink materializes features from Snowflake to Redis"

**Correct**:
- **Flink** reads from **Kafka** (streaming), computes realtime features (velocity), writes to Redis
- **Snowflake** stores historical data (batch), provides training data, materializes batch features to Redis

### Pitfall 2: Tumbling Windows at Edges ❌

**Wrong**: Use tumbling windows (1-min non-overlapping buckets) for velocity features

**Correct**: Use sliding windows (60-sec window, updated every second) to catch fraud at window edges

### Pitfall 3: Cold Start = Block by Default ❌

**Wrong**: Block all new users (too conservative, hurts UX)

**Correct**: Add `is_new_user` feature, let model learn patterns, use stricter threshold if needed

### Pitfall 4: Forgetting Dynamic Batching ❌

**Wrong**: Calculate throughput without batching → need 7 servers ($21K/year)

**Correct**: Use dynamic batching → need 2 servers ($6K/year) at acceptable latency

### Pitfall 5: Training with Future Data ❌

**Wrong**: Use features computed AFTER transaction time (data leakage!)

**Correct**: Point-in-time joins (AS OF transaction timestamp)

---

## Interview Q&A (Ready-to-Use Answers)

### Q: "Why XGBoost over neural networks?"

**A**: "For fraud detection on tabular data, XGBoost offers the best cost-performance trade-off:
- **Performance**: XGBoost typically matches DNN accuracy on tabular features (ensembles of trees handle non-linear patterns well)
- **Cost**: $6K/year with batching vs $90K/year for DNN (15× savings)
- **Interpretability**: Can explain why a transaction was flagged (important for compliance and user appeals)
- **Inference speed**: 5ms single prediction vs 20ms for DNN (important for 100ms latency budget)

If we had complex sequential patterns (e.g., analyzing transaction sequences over time), then LSTMs or Transformers would make sense. But for point-in-time fraud detection with aggregated features, XGBoost is the industry standard."

---

### Q: "How do you handle new users with no transaction history?"

**A**: "New users are actually a common fraud vector (stolen credentials to create new accounts), so we handle them carefully:

1. **Default feature values**: Use 0 for historical aggregates (or regional averages)
2. **Cold start indicators**: Add `is_new_user`, `account_age_days` as features
3. **Training inclusion**: Include new users in training data so model learns their patterns
4. **Conservative scoring**: Apply stricter thresholds (e.g., 0.7 vs 0.9 for auto-block)

XGBoost handles this naturally with decision trees—it learns branches like 'If new_user AND amount > $500 → high risk'. This is an advantage over neural networks which need careful embedding strategies for missing features."

---

### Q: "How do you prevent fraud at window edges with velocity features?"

**A**: "Fraudsters often exploit window boundaries by splitting transactions across adjacent time windows. For example, with 1-minute tumbling windows, they might make 3 transactions at 10:00:57-59 and 3 more at 10:01:01-03—each window sees only 3 transactions and passes the threshold of 5.

We use **sliding windows** instead: a 60-second window that updates every second. At 10:01:00, we count all transactions from 10:00:00 to 10:01:00, catching all 6 transactions as fraud.

Implementation in Flink uses `SlidingEventTimeWindows.of(Time.seconds(60), Time.seconds(1))`. The trade-off is higher compute cost (updating every second vs every minute), but Flink easily handles this at our 20K TPS scale, and catching fraud is worth the extra compute."

---

### Q: "What's your strategy for A/B testing the new model?"

**A**: "I'd run a per-transaction randomized A/B test:

**Setup**:
- Control (50%): Current model (80% recall at 99% precision)
- Treatment (50%): New model (targeting 82% recall at 99% precision)
- Duration: 2-3 days (capture day/night patterns)
- Sample size: ~500K transactions per group (calculated for 2% lift, alpha=0.05, power=0.8)

**Metrics**:
- Primary: Recall at 99% precision threshold (hypothesis: 80% → 82%)
- Secondary: User satisfaction (support tickets), transaction volume (false positive impact)

**Decision**: Ship if p-value < 0.05 AND recall lift >= 1%.

**Important**: We need to wait 7 days after the experiment to get confirmed fraud labels (chargebacks), so results are retrospective. This is why shadow deployment first is critical—we want offline validation before affecting users."

---

### Q: "Walk me through your feature store architecture."

**A**: "We have a two-tier feature store optimized for real-time serving:

**Online Store (Redis)**:
- Key: `user:{user_id}:features`
- Value: JSON with ALL features (tx_count_1min, last_10_tx, avg_amount_30d, etc.)
- TTL: 7 days (auto-expire old users)
- Latency: <1ms for single GET
- Updated by: Flink (realtime) + Snowflake (batch daily)

**Offline Store (Snowflake)**:
- Historical transactions + features
- Used for training data with point-in-time correctness
- Query: AS OF transaction timestamp (no data leakage)

**Data flow**:
```
Realtime: Kafka → Flink → Redis (velocity features, <1s latency)
Batch: Kafka → Spark → Snowflake → Redis (historical aggregates, daily refresh)
```

**Benefits**:
- Single Redis GET (not N GETs) → low latency
- Point-in-time correctness for training
- Monitoring feature drift (compare online vs offline distributions)"

---

## Key Formulas & Calculations

### Dynamic Batching Throughput
```
Batch size = QPS × Collection time
500 = 20,000 × 0.025s

Total latency = Collection time + Processing time
75ms = 25ms + 50ms

Throughput = Batch size / Total latency
6,666 QPS = 500 / 0.075s
```

### Server Cost Calculation
```
Servers needed = Peak QPS / QPS per server
2 servers = 20,000 / 10,000 (with batching)

Cost = Servers × $3K/year (CPU) or $10K/year (GPU)
$6K = 2 × $3K
```

### A/B Test Sample Size (Simplified)
```
Effect size: 2% lift (80% → 82% recall)
Fraud rate: 0.1% (10 per 10K transactions)

Required: ~500K transactions per group
Duration: 500K / 10K TPS = 50,000s = 14 hours per group
Add buffer for day/night: 2-3 days total
```

---

## Resources Studied

### System Design Patterns
- Day 20 quick reference: YouTube recommendation system
- Dynamic batching for cost optimization
- Feature store architecture (online/offline split)

### Fraud Detection Specifics
- Industry standards: 99% precision, 80%+ recall
- False positive vs false negative cost analysis
- Cold start handling for new users
- Sliding windows for velocity features

### Cost Analysis
- XGBoost: $6K/year with batching
- DNN: $90K/year with batching (15× more expensive)
- Throughput calculations: QPS per core, dynamic batching benefits

---

## Week 3 Summary: System Design Progress

**Day 20 (YouTube Recommendation)**:
- Mock interview: 78/100 (B+)
- Knowledge check: 91% (A-)
- Key gaps: Throughput calculations, cost analysis

**Day 21 (Fraud Detection)**:
- Mock interview: 85/100 (A-)
- Knowledge check: 97% (A+)
- **Improvement**: +7 points on design, +6 points on knowledge check!

**Major improvements**:
- Feature engineering: 7/10 → 9.5/10 (+2.5)
- Cost analysis: 6/10 → 9/10 (+3)
- Architecture understanding: Flink vs Snowflake clarified
- Dynamic batching mastery: Can calculate with/without batching scenarios

**System design readiness**: **85-90%** (up from 70% at start of Week 3)

---

**Last Updated**: 2025-11-17 (Week 3, Day 7)
**Next**: Week 4, Day 1 - Advanced RAG (FiD architecture, hybrid retrieval)
