# Day 20 Quick Reference: ML System Design - YouTube Recommendations

**Study Date**: 2025-11-16 (Week 3, Day 6)
**Topics**: ML system design practice, cascade architecture, data pipelines, A/B testing, scaling calculations
**Mock Interview Score**: 78/100 (B+)
**Knowledge Check Score**: 89.5% (A-/B+)

---

## Scale Numbers Cheat Sheet ⭐

### QPS (Queries Per Second)
- **Low scale**: 100-1K QPS (small app, internal tool)
- **Medium scale**: 1K-10K QPS (mid-size consumer app)
- **High scale**: 10K-100K QPS (YouTube, Netflix, Google Search)
- **Massive scale**: 100K-1M+ QPS (Facebook feed, Google Ads)

### Latency Targets (Response Time)
- **Real-time predictions**: <10ms (ad serving, fraud detection)
- **Near real-time**: 10-100ms (search ranking, recommendations)
- **Interactive**: 100-500ms (feed ranking, content moderation)
- **Batch**: Seconds to hours (email spam detection, offline recommendations)

### Model Inference Latency
- **Simple model (logistic regression)**: <1ms, handles 10K QPS per CPU core
- **Medium model (XGBoost)**: 1-10ms, 1K-10K QPS per CPU core
- **Medium DNN (100M params)**: 5-20ms, 100-500 QPS per GPU
- **Large DNN (1B+ params)**: 50-200ms, 50-100 QPS per GPU
- **LLM (GPT-3.5 size)**: 500ms-2s, 10-50 QPS per GPU

### Throughput Calculation Formula

```
Throughput (QPS) = Batch Size / Latency
```

**CRITICAL: Measure throughput in REQUESTS per second, not predictions per second!**

**Examples**:
- **1 request** scoring 1,000 candidates in 100ms → **10 QPS** ✅
- 100 candidates (1 request) in 50ms → 20 QPS
- Single prediction (1 request) in 5ms → 200 QPS

**Pattern**: Ranking N candidates = 1 request (not N requests)

**For scaling**:
```
Servers Needed = Total QPS / QPS per Server
Cost = Servers × $/hour × 8,760 hours/year
```

---

## Common ML System Architectures

### 1. Two-Tower Model (User Tower + Item Tower)

**Used for**: Recommendations, search ranking

**Architecture**:
```
User Features → DNN Encoder → User Embedding (128d)
                                    ↓
                              Dot Product → Score
                                    ↑
Item Features → DNN Encoder → Item Embedding (128d)
```

**Key Advantages**:
- **Pre-compute item embeddings offline** (fast serving <5ms lookup)
- Scales to billions of items (ANN search: FAISS, ScaNN)
- Used by YouTube, Pinterest, Airbnb

**Trade-off**: User embeddings can be stale (hours old) if pre-computed
- **Solution**: Hybrid approach
  - Stage 1: Pre-computed user embedding (hourly updates) for candidate generation
  - Stage 2/3: Real-time user embedding with fresh context for ranking

---

### 2. Cascade/Funnel Architecture (Multi-stage Ranking)

**Used for**: Ads, recommendations, search

**Architecture**:
```
1B items → Candidate Generation → 100k items (fast, simple models)
              ↓
         Ranking Model → 1k items (complex DNN)
              ↓
         Re-ranking → 100 items (business rules, diversity, freshness)
```

**Why Cascade**:
- Can't run expensive model on 1B items (too slow)
- Trade-off: Speed (stage 1) vs Accuracy (stage 2-3)

**Example: YouTube Recommendations** (30K QPS, <150ms budget):
```
Stage 1 (ANN): 100M videos → 1K candidates (20ms)
  - Two-tower with pre-computed embeddings
  - FAISS ANN search
  - 30 servers (1K QPS each)

Stage 2 (Lightweight Ranking): 1K → 100 candidates (20ms)
  - Simple scoring (dot product, rules)
  - Fresh context features
  - 6 servers (5K QPS each)

Stage 3 (Deep Ranker): 100 → 20 videos (60ms)
  - Complex DNN with cross-features
  - Batch inference on GPU
  - 200 GPUs (150 QPS each) = $3.5M/year ⚠️
```

**Cost Breakdown**:
| Component | Servers | Cost/Year |
|-----------|---------|-----------|
| Stage 1 (ANN, CPU) | 30 | $100K |
| Stage 2 (Simple, CPU) | 6 | $20K |
| Stage 3 (DNN, GPU) | 200 GPUs | $3.5M |
| Redis Cache | 10 | $30K |
| **Total** | | **~$3.7M/year** |

---

### Resource Allocation: Shared vs Independent ⭐ Important Trade-off

**The Question**: Should stages share servers (collocated) or have dedicated resources (microservices)?

#### Independent Resources (Microservices) - RECOMMENDED ✅

**Architecture**:
```
Load Balancer
    ↓
┌─────────────┐
│ Stage 1     │ 30 dedicated ANN servers
│ ANN Search  │ Each: 1K QPS
└─────────────┘
    ↓
┌─────────────┐
│ Stage 2     │ 6 dedicated CPU servers
│ Lightweight │ Each: 5K QPS
└─────────────┘
    ↓
┌─────────────┐
│ Stage 3     │ 200 dedicated GPUs
│ Deep Ranker │ Each: 150 QPS
└─────────────┘

Total: 30 + 6 + 200 = 236 machines
```

**How pipeline works**:
```
Time 0ms:   Request A → Stage 1 Server #1
Time 20ms:  Request A → Stage 2 Server #1  |  Request B → Stage 1 Server #2
Time 40ms:  Request A → Stage 3 GPU #1     |  Request B → Stage 2 Server #2  |  Request C → Stage 1 Server #3
```

**Key: Stages process DIFFERENT requests in parallel (pipeline parallelism)**

**Advantages** ✅:
- **Independent scaling**: Scale Stage 3 (200 GPUs) without touching Stage 1
- **Fault isolation**: Stage 2 crash doesn't affect Stage 1
- **Hardware optimization**: High-memory for Stage 1, GPUs for Stage 3
- **Better monitoring**: Per-stage metrics and alerts
- **Deployment independence**: Update Stage 2 without Stage 1 downtime

**Disadvantages** ❌:
- More machines (higher cost)
- More operational complexity

---

#### Shared Resources (Collocated) - RARELY USED

**Architecture**:
```
30 general-purpose servers
Each server runs BOTH Stage 1 and Stage 2
```

**Resource usage per server**:
```
Stage 1 at 1K QPS: Uses ~80% CPU (ANN search is expensive)
Stage 2 at 1K QPS: Uses ~20% CPU (lightweight scoring)
Total: 100% CPU ✅ Fits!

Throughput: 1K QPS per server (bottlenecked by Stage 1)
For 30K QPS: Need 30 servers
```

**Cost comparison (Stages 1-2 only)**:
```
Independent: 30 + 6 = 36 servers ($120K/year)
Shared: 30 servers ($90K/year)
Savings: $30K/year (17% cheaper)
```

**When shared makes sense** (rare):
- One stage is very lightweight (uses <20% resources)
- Low QPS (<100 QPS total)
- Cost-constrained startup
- Batch processing (not latency-sensitive)

**When shared FAILS** ❌:
```
If Stage 1 uses 80% CPU and Stage 2 uses 60% CPU:
Total: 140% CPU → Can't fit on one server!

Must reduce throughput:
- Stage 1: 0.5K QPS (40% CPU)
- Stage 2: 0.5K QPS (30% CPU)
- For 30K QPS: Need 60 servers (2× more!)
```

**Stage 3 (GPU) must be independent**:
- Can't collocate GPU with CPU stages
- Different hardware requirements

---

#### Interview Answer Template

**If asked**: "How many servers do you need?"

**Always clarify first**:
```
"Are we assuming microservices with dedicated resources per stage?
Or collocated stages on shared servers?"
```

**Then calculate**:
```
For microservices (standard assumption):
  Stage 1: 30K QPS / 1K QPS per server = 30 servers
  Stage 2: 30K QPS / 5K QPS per server = 6 servers
  Stage 3: 30K QPS / 150 QPS per GPU = 200 GPUs
  Total: 236 machines

  These stages run in parallel (pipeline), so throughput = 30K QPS
  (limited by the slowest stage, but all can handle 30K QPS)

For collocated (if asked):
  "If we collocate Stage 1 and Stage 2:
   - Stage 1 uses 80% CPU per 1K QPS
   - Stage 2 uses 20% CPU per 1K QPS (lightweight)
   - Both fit on one server → 30 servers total
   - Saves 6 servers (17% cost reduction)

   However, I'd recommend microservices because:
   - Savings are <1% of total cost ($30K vs $3.7M)
   - Independent scaling is more important at this scale
   - Operational flexibility outweighs small savings"
```

---

### 3. Lambda Architecture (Batch + Speed Layer)

**Used for**: Feature stores, real-time aggregations

**Architecture**:
```
Batch Layer (Spark): Historical data → Accurate features (daily)
         ↓
Speed Layer (Flink): Recent data → Approximate features (real-time)
         ↓
Serving Layer: Merge both → Serve predictions
```

**Example: Fraud Detection Features**:
- Batch: "User's total transactions last 30 days" (computed daily)
- Speed: "User's transactions last 1 hour" (computed real-time)
- Serving: Combine both for prediction

**When to use**: Need both accuracy (batch) and freshness (real-time)

---

## Data Pipeline Patterns

### YouTube Recommendation System Data Pipeline

**Scale**: 10 billion events/day (115K events/sec, 300K+ peak)

```
User Events → Kafka (Partitioned by user_id)
              ↓
         Flink (Streaming Aggregation)
              ↓ (every 10 min)
         Redis: {user_id: {last_10_videos, last_hour_stats}}
              ↓
    Serving: Query Redis (10ms) + Offline DB (10ms)
              ↓
         Generate Recommendations (<150ms total)

PARALLEL:
Kafka → Spark Batch (Daily)
        ↓
    Training Data Storage (90 days, ~1 trillion events, ~100TB)
        ↓
    Model Training
```

**Key Decisions**:
1. **Partition by user_id**: Maintains event ordering per user
2. **Flink aggregation**: Computes rolling statistics (last hour, last 10 videos)
3. **Redis caching**: <10ms latency for recent features
4. **Down-sample negatives**: Keep 1:1 positive:negative ratio (throw away 98% of negatives)
5. **Retention**: 90 days of training data (trade-off: recency vs volume)

---

## Training Data & Bias Handling

### Position Bias Problem

**Issue**: Users click top 3 videos 60% of the time → biased toward current model's recommendations

### Solution 1: Randomized Traffic (5-10%) ⭐ Recommended

**IMPORTANT**: This is 5% of REQUESTS, not 5% of USERS! Any user might occasionally get random recommendations, but most sessions are personalized.

**Approach A: Per-Request Randomization (Common)**
```
95% of requests: Get model recommendations (personalized slate)
5% of requests: Get RANDOM videos (unbiased evaluation)

Example for User 123:
  - Request 1-19: Personalized ✅ (good experience)
  - Request 20: Random ⚠️ (one bad session)
  - Request 21-39: Personalized ✅
  - Request 40: Random ⚠️
```

**Approach B: Per-Recommendation Exploration (Industry Practice)**
```
Each recommendation slate has 20 videos:
  - Positions 1-15: Model's top predictions (75% exploitation) ✅
  - Positions 16-18: Random/trending videos (15% exploration) ⚠️
  - Positions 19-20: Diversity boosting (10%) ⚠️

Every user gets MOSTLY personalized + SOME exploration in every session
```

**Why Approach B is better for UX**:
- User still gets 15+ good recommendations per page
- Exploratory videos aren't obviously bad (might discover something new)
- No user ever gets an entirely random slate

**Benefits**:
- Unbiased labels for ANY video (including those model wouldn't recommend)
- Can evaluate new models offline before A/B test
- Can measure model performance without position bias

**Cost**: Slightly worse UX (5-10% of recommendations are exploration), but acceptable trade-off

### Solution 2: Inverse Propensity Scoring (IPS)

**Formula**:
```
Loss = Σ (1 / P(video shown | position)) × loss(predicted, actual)
```

**Example**:
- Video at position 1: P(shown) = 0.9 → weight = 1/0.9 = 1.1
- Video at position 10: P(shown) = 0.1 → weight = 1/0.1 = 10

Videos shown less often get higher weight, compensating for position bias.

**What IPS CAN do** ✅:
- Correct position bias for videos that WERE shown
- Upweight videos at lower positions (seen less often)
- Reduce bias in training data

**What IPS CANNOT do** ❌:
- Give labels for videos that were NEVER recommended
- Tell you if unseen videos would have been watched
- Solve selection bias (only observation bias)

**Key limitation**: IPS only reweights existing data points. For videos never shown, you have no labels to reweight!

**Why randomized traffic is still needed**: To get labels for videos the model would never recommend.

**Problem**: High variance, requires accurate propensity estimates

**Interview insight**: "IPS corrects position bias for videos that were shown, but doesn't solve selection bias—we still have no labels for videos never recommended. That's why we need 5-10% randomized traffic."

### Label Construction

**Best practice**: `watch_time / video_duration` (% watched)
- Normalizes for video length (2-min video vs 2-hour movie)
- Range: 0-1 (0% watched to 100% watched)
- Better than binary (clicked/not clicked) or raw watch time

---

## A/B Testing for ML Systems

### Experiment Design

**Randomization Unit**: Per-user (NOT per-request)
- **Why**: Avoids carryover effects (user behavior changes based on previous recommendations)

**Duration**: 1-2 weeks
- Too short (1 day): Day-of-week effects
- Standard (1 week): Good for most tests
- Very safe (1 month): Slow iteration but catches seasonality

### Experiment Allocation Strategies ⭐ Important!

**Critical insight**: DON'T use 50/50 split for all experiments! Reserve most traffic for production.

**Standard progression**:

**Stage 1: Initial validation (1-2 weeks)**
```
99% control (495M users)
1% treatment (5M users)

Why:
- Minimize risk (only 1% exposed)
- Can run 10-20 concurrent experiments
- Still enough power: 5M users >> 40K needed for 1% effect
```

**Stage 2: Wider rollout (1-2 weeks, if Stage 1 successful)**
```
90% control (450M users)
10% treatment (50M users)

Why:
- Higher statistical power
- Faster detection of smaller effects
- Still relatively safe
```

**Stage 3: Final validation (1-2 weeks, if Stage 2 successful)**
```
50% control (250M users)
50% treatment (250M users)

Why:
- Maximum power for critical decision
- Only used before full rollout
- High stakes: 50% of users affected
```

**Total rollout time**: 3-6 weeks from first experiment to full launch

### Sample Size Calculation

**For 1% relative effect size**:
- Need ~40K users per group (80% statistical power)
- With repeated measures over 7 days, variance reduces by √7

**YouTube scale examples**:

| Allocation | Treatment Users | Time to 40K | Can Detect |
|------------|----------------|-------------|------------|
| 1% treatment | 5M | <1 day | 0.3% effects |
| 10% treatment | 50M | <1 day | 0.1% effects |
| 50% treatment | 250M | <1 day | 0.05% effects |

**Key insight**: Even 1% allocation gives 125× more users than needed!

### Multiple Concurrent Experiments

**Problem**: Running 10+ experiments simultaneously

**Solution 1: Disjoint buckets (Simple)**
```
Split 500M users into 100 buckets of 5M each

Experiment 1: Buckets 1-2 (10M: 5M control, 5M treatment)
Experiment 2: Buckets 3-4 (10M: 5M control, 5M treatment)
...
Experiment 10: Buckets 19-20

Buckets 21-100: Reserved (not in experiments)
```

**Solution 2: Layered experiments (Advanced)**
```
Independent features can share users:

Layer 1: Recommendation Algorithm (1% treatment)
Layer 2: UI Design (1% treatment)
Layer 3: Pricing (1% treatment)

User 123: Treatment in Layer 1, Control in Layer 2, Control in Layer 3
User 456: Control in Layer 1, Treatment in Layer 2, Control in Layer 3

Works because: Recommendation algo and UI are independent
Fails when: Testing two different recommendation algos (same layer)
```

**Solution 3: Holdout group (Netflix approach)**
```
10% global holdout: Never in any experiment
  → Measures cumulative effect of ALL experiments over 6 months

90% experiment pool: Available for layered experiments
```

### Primary vs Secondary Metrics

**Primary Metric**: Total watch time (North Star metric)
- Prevents multiple comparisons problem
- Aligns with business goal

**Secondary Metrics** (directional only):
- Click-through rate (CTR)
- Engagement rate (% watched >50%)
- Session length
- Return rate (7-day, 30-day)

---

## Monitoring & Evaluation

### Offline Evaluation (Before A/B Test)

**Methods**:
1. **Holdout set from randomized traffic** (5% random recommendations) → Unbiased
2. **Human evaluation** (expensive, slow)
3. **Interleaving** (mix old + new results, see which get clicked)
4. **Shadow mode** (run new model in parallel, compare predictions)

### Shadow Mode Explained

**What it is**: Run new model in parallel with current model, but DON'T show results to users

```
User Request:
  ↓
Current Model → Recommends [A, B, C] → SHOWN to user ✅
  ↓                                      ↓
Shadow Model → Recommends [A, D, E] → NOT shown (logged only) ⚠️
                                         ↓
                                    User clicks A
```

**What shadow mode CAN tell you** ✅:
- Prediction quality: Which model's scores are better calibrated?
- Ranking differences: How much do recommendations diverge?
- Score distributions: Is shadow model overconfident?
- Bugs: Does shadow model crash? Return empty results?

**What shadow mode CANNOT tell you** ❌:
- Would users have clicked Video D? (not shown)
- Is the shadow model actually better? (no causal data)
- What's the true lift in watch time? (need A/B test)

**Key insight**: Shadow mode compares PREDICTIONS, not OUTCOMES
- You get overlap data (Video A: both models + user feedback)
- You get non-overlap predictions (Video D: shadow prediction only, no user feedback)
- Use for safety checks and debugging, NOT evaluation

**Interview answer**: "Shadow mode runs the new model in parallel without serving results to users. We can compare predictions and catch obvious bugs, but we can't measure actual impact on user behavior because users don't see the shadow recommendations. That's why we need an A/B test after shadow mode to measure true lift."

### Deployment Pipeline

```
1. Shadow Mode (1-2 weeks)
   Purpose: Catch catastrophic failures, compare predictions
   Risk: Zero (users don't see it)
   Learn: Are predictions reasonable? Big divergence?

2. A/A Test (optional, 3-7 days)
   Purpose: Verify randomization works
   Test: Current model vs Current model
   Risk: Zero

3. A/B Test (1-2 weeks)
   Purpose: Measure ACTUAL impact on user behavior
   Test: Current model vs New model
   Risk: Medium (50% users see potentially worse recommendations)
   Learn: True causal effect! ✓✓✓
```

### Online Metrics (Real-time Monitoring)

**Alert Immediately**:
- Latency p99 > 200ms
- Error rate > 1%
- QPS anomalies (sudden drop)
- CTR drop > 5%

**Monitor Daily**:
- Feature distribution drift (KL divergence)
- Prediction distribution drift
- Model performance on recent data

### Data Drift Detection

**Method**: KL divergence on feature distributions (week-over-week)
- Alert if KL divergence > threshold (e.g., 0.1)
- Example: Age distribution shifts (more young users) → model might need retraining

---

## Interview Q&A (Ready-to-Use)

### Q: "Design a recommendation system for YouTube with 500M users and 1B videos."

**Answer**:
"I'd design a 3-stage cascade architecture to balance latency and accuracy:

**Stage 1 - Candidate Generation (20ms)**:
- Two-tower model with pre-computed user/video embeddings
- ANN search (FAISS) to retrieve 1K candidates from 100M active videos
- Embeddings updated: videos daily, users hourly
- Handles 30K QPS with 30 servers

**Stage 2 - Lightweight Ranking (20ms)**:
- Combine Stage 1 scores with fresh features (recent activity, time of day, device)
- Simple scoring (weighted combination) or small DNN
- Output: Top 100 candidates
- 6 CPU servers

**Stage 3 - Deep Ranking (60ms)**:
- Cross-feature DNN with user embeddings + video embeddings + context
- Batch inference on GPU
- Output: Top 20 videos
- 200 GPUs needed (150 QPS each), ~$3.5M/year

**Total latency**: 100ms (within 150ms budget)

**Data Pipeline**:
- Kafka (10B events/day) → Flink (aggregate last hour) → Redis (serve <10ms)
- Parallel: Spark batch → Training data storage (90 days, 1 trillion events)

**Training**:
- Label: watch_time / video_duration (% watched)
- Bias handling: 5-10% exploration traffic (per-request or per-recommendation)
  - Per-request: 5% of requests get fully random slates
  - Per-recommendation: Each slate has 15 personalized + 5 exploratory videos
- Inverse propensity weighting for training on biased data
- Down-sample negatives (1:1 ratio)

**A/B Testing** (staged rollout):
- Stage 1: 99/1 split (1-2 weeks) - validate no disasters, 5M users
- Stage 2: 90/10 split (1-2 weeks) - higher power, 50M users
- Stage 3: 50/50 split (1-2 weeks) - final validation, 250M users
- Per-user randomization (not per-request)
- Primary metric: Total watch time
- Can run 10-20 concurrent experiments with disjoint buckets or layered design"

---

### Q: "How do you handle the cold start problem?"

**Answer**:
"For **new users** (20% of traffic):
- Use profile-based features (age, country, language)
- Show trending videos (60%)
- Show popular in similar demographics (20%)
- Random exploration (20%) to discover preferences

For **new videos**:
- Use metadata features (title, description, category, publisher history)
- Give exploration bonus (temporarily boost score to gather engagement data)
- Monitor performance in first 24 hours, adjust recommendations

**Hybrid Stage 1 approach**:
```
For new users:
  60% trending (general popularity)
  20% profile-based (similar users)
  20% random (discover preferences)
```

This balances exploitation (use what we know) with exploration (learn user preferences)."

---

### Q: "How would you calculate the number of servers needed?"

**Answer**:
"Use this formula:

```
Servers Needed = Total QPS / QPS per Server
```

For YouTube Stage 3 (Deep Ranker):

**Given**:
- Total QPS: 30,000
- Model: Medium DNN processing 100 candidates
- Latency: 50ms per request with batching

**Step 1: Calculate single-GPU throughput**
- With batching and optimizations: ~150 QPS per GPU (given in problem)

**Step 2: Calculate total GPUs**
```
GPUs = 30,000 QPS / 150 QPS per GPU = 200 GPUs
```

**Step 3: Estimate cost**
```
Cost = 200 GPUs × $2/hour (A100 cloud cost)
     = $400/hour
     = $3.5M/year
```

**Optimization strategies**:
1. Use smaller models (sacrifice accuracy for cost)
2. Cache popular results (reduce QPS load)
3. Use XGBoost instead of DNN for Stage 2 (10× cheaper, similar performance)
4. Quantize models (INT8) for 2-4× speedup"

---

### Q: "Explain the two-tower architecture. Why is it fast?"

**Answer**:
"Two-tower encodes users and items separately, then computes dot product similarity:

```
User Features → User Encoder → User Embedding (128d)
                                     ↓
                               Dot Product → Score
                                     ↑
Item Features → Item Encoder → Item Embedding (128d)
```

**Why it's fast for recommendations**:

1. **Pre-compute item embeddings offline**:
   - Encode 100M videos daily → Store in database
   - At serving: No need to run item encoder

2. **Fast retrieval with ANN search**:
   - User embedding (128d) + FAISS → Find top 1K similar items in 20ms
   - Scales to billions of items

3. **Compared to cross-feature model**:
   - Two-tower: Pre-compute items (20ms), only encode user at serving
   - Cross-feature: Encode user + ALL items at serving (50ms × 100M = forever!)

**Trade-off**:
- **Speed**: Two-tower is 1000× faster
- **Accuracy**: Cross-feature models are slightly more accurate (can learn complex interactions)

**Industry practice**: Two-tower for Stage 1 (speed), cross-feature for Stage 3 (accuracy on small candidate set)"

---

## Key Learnings & Patterns

### Scaling Calculation Pattern

Always calculate in this order:
1. **Latency per request** (given or measure)
2. **Throughput per server**: QPS = Batch Size / Latency
3. **Total servers needed**: Total QPS / QPS per server
4. **Cost**: Servers × $/hour × 8760 hours/year

### Cost Optimization Hierarchy

1. **Caching** (10-100× cost reduction)
   - Cache popular results in Redis
   - Reduce load on expensive models

2. **Model complexity** (5-20× cost reduction)
   - Use XGBoost instead of DNN when possible
   - Use smaller models for early stages

3. **Quantization** (2-4× cost reduction)
   - INT8 quantization for inference
   - Minimal accuracy loss

4. **Batching** (5-10× throughput increase)
   - Batch multiple requests together
   - GPU utilization improves dramatically

---

## Common Pitfalls

### During Mock Interview

1. **Forgot to calculate servers needed** ❌
   - Always show: Total QPS / QPS per server
   - Show cost estimation

2. **Vague about implementation details** ❌
   - Name specific tools: Kafka, Flink, Redis, FAISS
   - Don't say "streaming system" - say "Kafka"

3. **Didn't discuss trade-offs** ❌
   - Every decision has a trade-off
   - Accuracy vs latency, cost vs performance

4. **Missed offline evaluation strategy** ❌
   - How do you test before A/B test?
   - Randomized traffic is industry standard

### Technical Mistakes

1. **Throughput from latency**:
   - Remember batching: 50ms latency → NOT 20 QPS per GPU!
   - With batching: 100 items in 50ms = 2K QPS per GPU
   - Always check if batching is used

2. **Pre-computed user embeddings**:
   - CAN pre-compute for Stage 1 (speed > freshness)
   - CANNOT use for Stage 2/3 (need fresh context)

3. **Position bias training**:
   - Don't add position as feature (creates inference problem!)
   - Use IPS or randomized traffic instead

4. **Resource allocation assumptions**:
   - ✅ Always clarify: "Assuming microservices with independent resources?"
   - ❌ Don't assume shared resources without discussing trade-offs
   - Pipeline parallelism means stages process DIFFERENT requests concurrently

5. **A/B testing allocation**:
   - ❌ Don't say "50/50 split" for all experiments
   - ✅ Start with 99/1 or 95/5, progress to 50/50 only for final validation
   - Multiple experiments need traffic management (disjoint buckets or layers)

---

## Server Throughput Estimation Strategies

### Problem: How Do You Know QPS per Server?

In interviews, you won't know exact numbers like "FAISS handles 1K QPS per server" unless you've worked with these systems. Here are three valid approaches:

### Approach A: Know Rough Ballparks (Preferred for Senior Roles)

**Memorize these order-of-magnitude estimates:**

| Operation | Throughput per Server/Core | Notes |
|-----------|---------------------------|-------|
| Key-value lookup (Redis) | 50K-100K QPS | Single server (16 cores) |
| ANN search (FAISS) | 500-2K QPS | Single server, 100M vectors |
| Logistic regression | 10K QPS | Per CPU core |
| XGBoost (100 trees) | 1K-5K QPS | Per CPU core |
| Medium DNN (BERT-base) | 100-500 QPS | Per GPU with batching |
| Large LLM (GPT-3.5) | 10-50 QPS | Per GPU with batching |

**Where these come from:**
- Production experience (working at YouTube, Netflix, Meta)
- Tech blog posts (Uber Engineering, Netflix TechBlog)
- Published benchmarks (FAISS documentation, MLPerf)

**Interview usage**: "From FAISS benchmarks and production experience, ANN search typically handles ~1K QPS per server for 100M vectors, so we'd need ~30 servers for 30K QPS."

### Approach B: Derive During Interview (Acceptable)

**Show your reasoning from first principles:**

```
Example: "How many servers for Stage 1 ANN search?"

Your thought process (spoken aloud):
1. "Let me estimate ANN search latency..."
2. "100M vectors, need k-NN search"
3. "With indexing (HNSW), only check ~10K vectors"
4. "10K comparisons × 1μs ≈ 10ms per query"
5. "Throughput: 1/0.01 = 100 QPS per core"
6. "With 16 cores: 1,600 QPS per server"
7. "For 30K QPS: need ~20 servers"
8. "But I should verify this is reasonable..."
9. "Industry benchmarks are ~500-2K QPS, so 20 servers seems right"
```

**What this shows:**
- ✅ Can reason from latency → throughput
- ✅ Know optimizations exist (indexing, HNSW)
- ✅ Sanity-check against industry numbers
- ✅ Give a range, not a false-precision single number

---

**IMPORTANT: It's okay NOT to derive ANN QPS! ⭐**

**Why ANN is hard to estimate without experience:**
- Complex algorithms (HNSW, IVF, Product Quantization)
- Heuristic search (not simple N comparisons)
- Trade-offs between accuracy and speed (ef_search parameter)
- Depends on data distribution and index type

**If you don't know**:
```
"Without production experience with FAISS, I'd approach this empirically:

1. Start with published benchmarks (FAISS docs show ~1K QPS for this scale)
2. Run a prototype load test with representative data
3. Measure p50, p95, p99 latency under different loads
4. Back-calculate sustainable QPS

I wouldn't try to derive this from first principles because ANN uses
heuristic search—it's not just 'N vector comparisons / latency'."
```

✅ This shows **engineering judgment** (know what you don't know)

### Approach C: Use Rules of Thumb

**Quick formula for ballpark estimates:**

```
QPS ≈ 1000 / latency_ms × parallelism

Examples:
- 10ms latency, 16 cores → 1,000/10 × 16 = 1,600 QPS
- 50ms latency, 8 GPUs → 1,000/50 × 8 = 160 QPS
```

**Parallelism factors:**
- CPU-bound tasks: 10-50× (depends on cores)
- GPU batching: 5-20× (depends on batch size)
- I/O-bound tasks: 100-1000× (async operations)

### What Interviewers Actually Expect

**Mid-Level (0-5 years)**: Ballpark is fine
- ✅ "ANN search probably handles 100-1K QPS per server"
- ✅ "So we'd need 30-300 servers for 30K QPS"

**Senior (5-10 years)**: More precision expected
- ✅ "FAISS on 16-core server handles ~1K QPS for 100M vectors"
- ✅ "For 30K QPS, we need ~30 servers with load balancing"

**Staff+ (10+ years)**: Deep optimization knowledge
- ✅ "Initial: 30 CPU servers = $100K/year"
- ✅ "Alternative: 6 GPU servers with GPU-FAISS = $100K/year, better p99 latency"
- ✅ "Choose GPUs for latency-critical stage, CPUs for cost-sensitive stages"

### How to Improve This Skill

**1. Read Tech Blogs (2-3 hours total):**
- Uber Engineering: "Scaling ML Platform"
- Netflix TechBlog: "Recommendation Systems at Scale"
- Meta Engineering: "Serving Models at Scale"
- Look for: "We serve X QPS with Y servers"

**2. Memorize Key Benchmarks:**
- Redis: 100K QPS per server
- FAISS: 1K QPS per server (100M vectors)
- XGBoost: 5K QPS per core (small model)
- DNN: 100-500 QPS per GPU (medium model)

**3. Practice Estimation:**
- For each system design, calculate: latency → QPS → servers → cost
- Check if your estimate is reasonable (not 1M servers or 0.1 servers)

### Interview Strategy

**When asked about throughput and you don't know:**

```
Option 1: Derive from first principles
"I don't have exact benchmarks, but let me estimate from latency...
[show calculation]... ~500 QPS per server"

Option 2: Give a range with caveat
"I'd estimate 500-2K QPS per server based on similar systems.
For planning, let's use 1K QPS conservatively, so 30 servers.
In practice, I'd benchmark our specific setup first."

Option 3: Reference industry standards
"From FAISS documentation and blog posts, typical throughput
is ~1K QPS for this scale, so ~30 servers needed."
```

**Always add**: "In production, I'd run load tests to get exact numbers before capacity planning."

---

## Typical Numbers Cheatsheet ⭐⭐⭐

**Purpose**: Quick reference for realistic numbers when you need to estimate throughput, resources, or costs in system design interviews.

**Key insight from study**: "I know how to do the calculation (QPS = Batch Size / Latency), if I know what the number is. In reality, I am not sure of batch size/number of cores, sometimes nor latency." - This section provides those numbers!

---

### 1. Server Configurations

| Server Type | Cores | Memory | GPU Memory | Cost/Year | Typical Use Case |
|------------|-------|--------|------------|-----------|------------------|
| **Small CPU** | 4 cores | 16 GB | - | $1K | Development, small services |
| **Medium CPU** | 16 cores | 64 GB | - | $3K | Production inference, XGBoost |
| **Large CPU** | 32 cores | 128 GB | - | $6K | High-throughput services |
| **GPU (V100)** | 16 CPU + 1 GPU | 64 GB | 16 GB | $10K | ML inference |
| **GPU (A100)** | 32 CPU + 1 GPU | 128 GB | 40 GB | $18K | ML training/large inference |
| **Multi-GPU (4×A100)** | 64 CPU + 4 GPU | 256 GB | 160 GB | $60K | Distributed training |

**Common interview default**: 16-core CPU server ($3K/year) unless specified otherwise

---

### 2. Typical Batch Sizes

| Operation | Typical Batch Size | Reasoning |
|-----------|-------------------|-----------|
| **ANN search** | 1 per request | Each user query searches independently |
| **Feature lookup (Redis)** | 1-100 keys | Per-user or per-candidate features |
| **XGBoost scoring** | 1,000 candidates | Scoring all candidates for one user (= 1 request!) |
| **DNN inference (GPU)** | 32-256 | GPU optimized for large batches |
| **Transformer inference** | 1-32 | Limited by sequence length & memory |
| **LLM inference** | 4-32 | Limited by KV-cache memory (40GB GPU) |
| **Batch training** | 256-2048 | Depends on model size & GPU memory |

**Key pattern**: Ranking N candidates = **1 request** (not N requests!)
- Scoring 1,000 candidates in 100ms = 10 QPS ✅
- NOT 1,000 predictions / 0.1s = 10K QPS ❌

---

### 3. Typical Latencies

| Operation | Latency Range | Notes |
|-----------|---------------|-------|
| **Redis lookup** | 0.1-1 ms | In-memory, same datacenter |
| **DynamoDB lookup** | 1-5 ms | Network + disk |
| **Simple computation** | <1 ms | Hashing, arithmetic |
| **XGBoost (1K predictions)** | 5-10 ms | CPU, can batch efficiently |
| **ANN search (1M vectors)** | 2-5 ms | Small scale (Spotify) |
| **ANN search (100M vectors)** | 10-20 ms | Large scale (YouTube) |
| **Small DNN (batch=1)** | 10 ms | GPU underutilized |
| **Medium DNN (batch=128)** | 50 ms | Good GPU utilization |
| **Large transformer (batch=32)** | 100 ms | BERT-Large, T5 |
| **LLM token generation** | 20-50 ms/token | Auto-regressive |

**Interview tip**: If you don't know exact latency, give a **range** (e.g., "10-50ms for medium DNN") and say you'd benchmark.

---

### 4. Parallelism Factors

| Resource | Parallelism Factor | When to Apply |
|----------|-------------------|---------------|
| **16 CPU cores (CPU-bound)** | 10-16× throughput | Heavy computation (XGBoost, hashing) vs 1 core |
| **16 CPU cores (memory-bound)** | 4-8× throughput | Large model serving, cache misses vs 1 core |
| **16 CPU cores (I/O-bound)** | 100-1000× throughput | Async I/O (Redis, HTTP calls) vs synchronous |
| **1 GPU (batch=128 vs batch=1)** | 30-60× throughput | DNN inference with batching vs no batching |
| **4 GPUs (data parallel)** | 3.5-4× throughput | Multi-GPU training vs 1 GPU (communication overhead ~10-15%) |

**Common mistakes**:
- Assuming 16 cores = 16× speedup → **Reality**: 10-16× for CPU-bound, 4-8× for memory-bound
- Assuming batch=128 gives 128× speedup → **Reality**: 30-60× due to memory bandwidth bottleneck

**Formula for QPS**:
```
QPS = (1 / latency_seconds) × parallelism_factor

Examples:
- XGBoost: Score 1,000 candidates per request, 10ms, 16 cores CPU-bound
  → QPS = (1 / 0.01) × 12 = 1,200 requests/sec (using 12× factor conservatively)

- DNN (no dynamic batching): Score 1,000 candidates per request, 50ms, 1 GPU
  → QPS = (1 / 0.05) × 1 = 20 requests/sec
  → NOTE: The 1,000 candidates are processed in parallel (vectorized), but this is 1 user request

- DNN (with dynamic batching): Batch 50 user requests together
  → Latency: 200ms for 50 requests (50,000 candidates total)
  → QPS = 50 / 0.2 = 250 requests/sec per GPU
  → 12.5× improvement from dynamic batching!
```

---

### 5. Decision Tree: "I Don't Know the Numbers"

```
Q: What's the throughput of [operation X]?

├─ Do I know similar systems? (e.g., read tech blogs, previous experience)
│  └─ YES → Give ballpark from memory
│           "FAISS typically handles ~1K QPS for 100M vectors"
│
├─ Can I derive from latency?
│  └─ YES → Calculate: QPS = (1 / latency) × parallelism
│           "10ms latency, 16 cores → ~1K QPS"
│
├─ Is it a standard operation?
│  └─ YES → Use typical range from this cheatsheet
│           "XGBoost: 1-5K QPS per server depending on model size"
│
└─ Completely unfamiliar? (e.g., ANN algorithms)
   └─ BE HONEST → "I'd benchmark empirically"
                  "Start with published numbers (FAISS docs: ~1K QPS)"
                  ✅ Shows engineering judgment
```

**Key principle**: It's BETTER to say "I'd benchmark this" than to make up numbers!

---

### 6. Quick Estimation Examples

#### Example 1: XGBoost Scoring

**Given**: Rank 1,000 candidates using XGBoost, need 30K QPS

**Step 1: Estimate per-server QPS**
- Latency: 10ms per request (scoring 1,000 candidates = 1 request!)
- Parallelism: 16 cores, CPU-bound → 12× factor (conservative)
- QPS per server = (1 / 0.01) × 12 = **1,200 QPS**

**Step 2: Calculate servers needed**
- Servers = 30,000 QPS / 1,200 QPS = **25 servers**
- Add 20% buffer → **30 servers**

**Step 3: Calculate cost**
- 30 servers × $3K/year = **$90K/year**

---

#### Example 2: DNN Inference (GPU)

**Given**: Rank 1,000 candidates using DNN, need 5K QPS

**Approach A: No Dynamic Batching** (Simple but Expensive)

**Step 1: Estimate per-GPU QPS**
- Each request: Score 1,000 candidates for one user
- Latency: 50ms per request (1,000 candidates processed in parallel via vectorization)
- QPS per GPU = 1 / 0.05 = **20 QPS**

**Step 2: Calculate GPUs needed**
- GPUs = 5,000 QPS / 20 QPS = **250 GPUs** ❌ VERY expensive!
- Cost: 250 × $10K/year = **$2.5M/year**

---

**Approach B: With Dynamic Batching** ⭐ (Production Approach)

**Step 1: Estimate per-GPU QPS with batching**
- Collect requests for 10ms → ~50 requests (at 5K QPS arrival rate)
- Total predictions: 50 requests × 1,000 candidates = 50,000 candidates
- Processing time: 200ms (large batch, memory-bound)
- QPS per GPU = 50 / 0.21 ≈ **240 QPS** (12× improvement!)

**Step 2: Calculate GPUs needed**
- GPUs = 5,000 QPS / 240 QPS ≈ **21 GPUs** ✅ Much better!
- Cost: 21 × $10K/year = **$210K/year**

**Step 3: Trade-off Analysis**
- Latency increase: 50ms → 210ms (collect 10ms + process 200ms)
- If 210ms > latency budget → Use smaller batching window (5ms → 25 requests)
- With 5ms batching: ~50 GPUs needed (still 5× cheaper than no batching)

**Step 4: Additional Optimizations**
- **Model distillation**: 200M params → 50M params (2× faster)
- **TensorRT**: FP32 → FP16/INT8 quantization (2× faster)
- **Combined**: 21 GPUs → 10-12 GPUs ($120K/year)

**Final answer**: ~20-50 GPUs depending on latency budget
- Best case (dynamic batching + optimizations): $120K/year
- Worst case (no batching): $2.5M/year
- **Key learning**: Dynamic batching gives 10-20× cost savings!

---

#### Example 3: ANN Search (When You Don't Know)

**Given**: ANN search on 100M vectors, need 30K QPS

**Approach 1: Admit you'd benchmark** ✅
```
"I haven't worked with FAISS at this scale, so I'd start with:

1. Published benchmarks: FAISS docs show ~1K QPS for 100M vectors
2. Conservative estimate: 30K / 1K = 30 servers
3. Load test with representative data to validate
4. Adjust based on p99 latency requirements

For rough costing: 30 servers × $3K = $90K/year"
```

**Approach 2: Derive with sanity check**
```
"Let me estimate from latency:
- 100M vectors, HNSW index → ~10ms per query (guess)
- 16 cores, I/O-bound (index lookups) → 100× parallelism
- QPS = (1/0.01) × 100 = 10K QPS per server

Wait, that seems high compared to published numbers (~1K QPS).
Let me revise: Maybe memory-bound → 20× parallelism
→ QPS = 100 × 20 = 2K QPS per server

For 30K QPS: 15-30 servers depending on actual parallelism"
```

**Key**: Both approaches are fine! What matters is showing your thought process and acknowledging uncertainty.

---

### 7. Common Pitfalls

| Pitfall | Wrong Answer | Correct Answer |
|---------|--------------|----------------|
| **Predictions ≠ Requests** | 1K predictions in 100ms = 10K QPS | 1 request (1K candidates) in 100ms = 10 QPS |
| **Assuming linear parallelism** | 16 cores = 16× speedup | 10-16× (CPU-bound), 4-8× (memory-bound) |
| **Assuming batch=128 is 128× faster** | Batch=128 gives 128× speedup | 30-60× throughput gain (memory-bound, latency increases) |
| **Ignoring dynamic batching** | DNN: 50ms/request = 20 QPS | With dynamic batching (50 requests): 240 QPS (12× better!) |
| **False precision** | "Exactly 1,247 QPS per server" | "~1-2K QPS per server" (give range) |
| **Making up numbers** | "I think ANN is 10K QPS" (wrong) | "I'd benchmark, but FAISS docs say ~1K QPS" |

---

### 8. Interview-Ready Template

**When asked**: "How many servers do we need for stage X?"

**Answer template**:
```
"Let me estimate the per-server throughput first:

1. Latency: [operation] takes ~[X]ms per request
   [If unsure: "I'd expect 10-50ms based on similar systems"]

2. Parallelism: [Y cores/GPUs], [CPU-bound/GPU-batched]
   → ~[Z]× effective parallelism

3. QPS per server: (1 / [X]ms) × [Z] ≈ [N] QPS

4. Servers needed: [Total QPS] / [N] ≈ [M] servers
   Add 20% buffer for failover → [M × 1.2] servers

5. Cost: [M × 1.2] × $[cost/server/year]

In production, I'd validate with load testing before finalizing capacity."
```

---

## Tools & Technologies Reference

### Data Streaming
- **Kafka**: Event streaming, 100K+ events/sec
- **Flink**: Stream processing, windowed aggregations
- **Kinesis**: AWS alternative to Kafka

### Feature Serving
- **Redis**: In-memory KV store, <1ms latency
- **DynamoDB**: AWS NoSQL, <10ms latency
- **Feast**: Open-source feature store

### Model Serving
- **FAISS**: Facebook ANN search library (billions of vectors)
- **ScaNN**: Google ANN library
- **Annoy**: Spotify ANN library (simpler)

### ML Frameworks
- **PyTorch/TensorFlow**: Model training
- **TorchServe/TensorFlow Serving/Triton**: Model serving
- **Ray Serve**: Distributed serving

### Batch Processing
- **Spark**: Large-scale batch processing
- **Airflow**: Workflow orchestration
- **BigQuery/Snowflake**: Data warehousing

---

## Resources Studied

- Mock Interview: YouTube Recommendation System (60 min)
- Discussion: Latency vs QPS relationship, batching, scaling calculations (30 min)
- Discussion: Two-tower pre-computation (user vs item embeddings) (15 min)

**Key Papers/Concepts Referenced**:
- Two-tower architecture (YouTube, Pinterest)
- Cascade ranking (Google Ads, Facebook Feed)
- Position bias handling (IPS, randomized traffic)
- Feature stores (Feast, Tecton)

---

## Next Practice

**Focus Areas for Day 21**:
1. ✅ **Scaling calculations**: Practice QPS → servers → cost for every component
2. ✅ **Cost estimation**: Always estimate $/year for GPU-heavy systems
3. ✅ **Specific tool names**: Kafka, Flink, Redis, FAISS (not "streaming system")
4. ✅ **Offline evaluation**: Randomized traffic, interleaving, shadow mode

**Recommended Next Problem**: Fraud Detection System
- Different architecture: Real-time, class imbalance, low latency (<50ms)
- Practice XGBoost vs DNN trade-off
- Online learning, concept drift

---

**Day 20 Status**: ✅ COMPLETED

**Session time**: ~2 hours initial + 1.5 hours follow-up discussions

- ⏱️ Mock interview: 60 min (78/100 B+)
- 📊 Knowledge check: 91% (A-/A, revised from 89.5%)
- 🎯 Topics practiced: System design framework, cascade architecture, data pipelines, A/B testing, scaling
- 💡 Key insight: Throughput = Batch Size / Latency (not just 1/Latency!)
- 💡 User caught throughput calculation error: 1K predictions = 1 request, not 10K QPS!

---

## Updates Based on Follow-up Discussions (2025-11-16)

**Three major clarifications added after user questions:**

### 1. Resource Allocation: Shared vs Independent ⭐

**User question**: "If we reuse a CPU between stage 1 and stage 2, that will affect QPS calculation right?"

**Answer**: YES! Added comprehensive section explaining:
- **Independent (microservices)**: 30 + 6 = 36 servers (Stages 1-2)
  - Pipeline parallelism: Stages process DIFFERENT requests concurrently
  - Can scale independently, better fault isolation
- **Shared (collocated)**: 30 servers if Stage 2 is lightweight (<20% CPU)
  - Saves 6 servers (17% cost reduction)
  - But loses operational flexibility
- **Key insight**: Savings are <1% of total cost ($30K vs $3.7M), so microservices preferred

**My error corrected**: Initially calculated serial processing (25 QPS) ignoring multi-threading. User correctly questioned the huge difference. Actual difference: 36 vs 30 servers (17%), not 36 vs 1,200 servers (3,200%)!

### 2. A/B Testing Allocation Strategies ⭐

**User question**: "Don't we need to split into smaller cells? What if we have more tests to run?"

**Answer**: Absolutely! Added sections on:
- **Staged rollout**: 99/1 → 90/10 → 50/50 (not 50/50 for everything!)
- **Concurrent experiments**: Disjoint buckets vs layered experiments
- **Holdout groups**: 10% never in experiments (Netflix approach)
- **Sample size**: Even 1% allocation (5M users) is 125× more than needed (40K)

**My error corrected**: Oversimplified A/B testing by showing only 50/50 split. Reality: Start with 1-10% treatment, reserve most traffic for production and other experiments.

### 3. ANN QPS Estimation ⭐

**User insight**: "I guess I am not familiar with ANN to derive the QPS."

**Answer**: That's completely valid! Added section:
- **Why ANN is hard**: Complex algorithms (HNSW, IVF), heuristic search, accuracy/speed trade-offs
- **It's okay not to know**: Show engineering judgment instead
- **Approach**: "Start with published benchmarks (~1K QPS), run prototype load test, back-calculate"
- **Interview strategy**: Admit you'd benchmark empirically, not derive from first principles

### 4. Typical Numbers Cheatsheet ⭐⭐⭐

**User insight**: "I think my gap is not about 'QPS = Batch Size / Latency'. I know how to do the calculation, if I know what the number is. In reality, I am not sure of batch size/number of cores, sometimes nor latency."

**Answer**: This completely reframed the problem! Added comprehensive cheatsheet with:
- **Server configurations**: Small/medium/large CPU, GPU (V100, A100), multi-GPU (with costs)
- **Typical batch sizes**: ANN (1), XGBoost (1K candidates = 1 request!), DNN (32-256), LLM (4-32)
- **Typical latencies**: Redis (0.1-1ms), XGBoost (5-10ms), ANN (10-20ms for 100M vectors), DNN (50ms batched)
- **Parallelism factors**: CPU-bound (10-16×), memory-bound (4-8×), GPU batching (10-20×)
- **Decision tree**: What to do when you don't know the numbers (benchmark, derive, use typical ranges)
- **Quick estimation examples**: XGBoost, DNN, ANN with step-by-step calculations
- **Common pitfalls**: Predictions ≠ requests, linear parallelism assumption, false precision
- **Interview-ready template**: Standard approach for "How many servers?" questions

**User feedback**: "SG, but looks hard to remember" → documented as permanent reference material

**Key insight**: The real gap wasn't formula knowledge, but knowing what realistic numbers to plug into formulas. This cheatsheet provides those numbers!

### 5. Randomized Traffic Clarification ⭐

**User question (2025-11-17)**: "Is it 5% requests or user? If 5% user never get personalization that sounds pretty bad right?"

**Answer**: Excellent catch! Clarified that:
- **5% of REQUESTS** (not users) - any user occasionally gets random slate but mostly personalized
- **Industry practice**: Per-recommendation exploration (15 personalized + 5 exploratory per slate)
  - Better UX: Every user gets mostly personalized in every session
  - No user ever gets entirely random slate
- **Original wording was ambiguous** and implied 5% of users permanently get bad experience (NO ONE DOES THIS!)

**Updated sections**:
- Training Data & Bias Handling: Now clearly explains per-request vs per-recommendation
- Interview Q&A: Updated training section to mention both approaches

### 6. GPU Batching & Dynamic Batching Corrections ⭐⭐⭐

**User questions (2025-11-17)**:
1. "Why 1 GPU (batch=128) is 10-20×? Shouldn't it be larger?"
2. "Why is it 20 QPS only with batch size 128? I suppose should be much bigger?"
3. Common Pitfalls table: "With batch=128, latency is same but QPS should be much higher"

**Answer**: User caught THREE major errors in my cheatsheet! All related to confusion between:
- **Candidates per request** (vectorization within one user request)
- **Dynamic batching of user requests** (batching multiple users together)

**Corrections made**:

1. **GPU parallelism factor**: Changed 10-20× to **30-60× throughput improvement**
   - Why not 128×? Memory bandwidth bottleneck (not compute-bound at large batches)
   - Batch=1: 50 QPS, Batch=128: 1,500 QPS → 30× gain (not 10-20×!)

2. **DNN Example 2 rewritten**: Now shows BOTH scenarios
   - **Without dynamic batching**: 20 QPS → need 250 GPUs ($2.5M/year)
   - **With dynamic batching** (50 user requests): 240 QPS → need 21 GPUs ($210K/year)
   - **Key insight**: Dynamic batching gives 12× cost savings!

3. **Common Pitfalls table**: Fixed two rows
   - Added: "Assuming batch=128 is 128× faster" → Actually 30-60× (memory-bound)
   - Fixed: "Ignoring dynamic batching" → Show 20 QPS vs 240 QPS comparison

4. **Formula examples**: Added clarification
   - DNN (no batching): 20 requests/sec
   - DNN (with dynamic batching of 50 requests): 250 requests/sec per GPU
   - Explicitly noted "1,000 candidates = 1 user request"

**Original error**: I said "batch=128" in the parallelism table without clarifying what was being batched:
- If batching 128 **user requests** → Should give 240-640 QPS (not 20!)
- If scoring 1,000 **candidates per request** → Gives 20 QPS (but shouldn't call this "batch=128")

**Key learning**: Always distinguish between:
- **Vectorization**: Processing 1,000 candidates for one user (parallel operations within request)
- **Dynamic batching**: Processing 50-128 user requests together (batching across requests)

### Key Takeaways from User Questions

1. **Critical thinking**: User questioned calculation errors (throughput, resource allocation, GPU batching)
2. **Production concerns**: User asked about real-world constraints (multiple experiments, traffic allocation)
3. **Honesty about gaps**: User acknowledged unfamiliarity with ANN (demonstrates maturity)
4. **Identifying real gaps**: User clarified "I know the formula, I don't know the numbers" - meta-awareness of learning needs
5. **Attention to consistency**: User caught THREE related errors about GPU batching - showing deep understanding of the concepts

**These questions elevated the reference from "mock interview notes" to "production-ready system design guide"!**

**Most impressive catches**:
- Randomized traffic: 5% users vs 5% requests (UX impact)
- GPU batching: 10-20× vs 30-60× (fundamental misunderstanding of parallelism)
- Dynamic batching: 20 QPS vs 240 QPS (12× cost difference!)
