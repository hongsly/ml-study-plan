# Day 10 Quick Reference: Parallelism Trade-offs & GPU Hardware

**Week 2, Day 3** | Focus: Communication costs, hardware bottlenecks, scaling strategies

---

## **1. GPU Performance Fundamentals**

### **Memory Bandwidth is the Bottleneck**
- **Tensor Cores**: So fast they're idle 50% of time waiting for data (GPT-3 training)
- **Memory bandwidth** > TFLOPS for predicting GPU performance
- **Example**: A100 (1555 GB/s) vs V100 (900 GB/s) → 1.73× speedup ≈ bandwidth ratio

### **Memory Hierarchy** (Slow → Fast)
```
HBM (GPU main memory) → L2 cache → L1 cache/Shared memory → Registers
      ↑ Largest                                                ↑ Smallest
      ↑ Slowest                                                ↑ Fastest
```

### **Why 500 TFLOPS GPU Only Sustains 50 TFLOPS** (Gap Q184)
- **Memory-bound operations**: Low arithmetic intensity
- Tensor Cores wait for data from HBM 90% of time
- **Bottleneck**: Memory bandwidth, not compute capability

---

## **2. Roofline Model**

### **Key Formula**
```
Actual Performance = min(Peak Compute, Memory Bandwidth × Arithmetic Intensity)
```

### **Arithmetic Intensity**
- **Definition**: FLOPs / bytes accessed
- **High intensity** (compute-bound): Large matrix multiply, GPU runs at peak
- **Low intensity** (memory-bound): Element-wise ops, LLM inference, GPU idles

### **Roofline Diagram**
```
Performance
    │     ___________________  ← Flat roof (peak compute)
    │    /
    │   /  ← Slanted roof (memory bandwidth limit)
    │  /
    │ /
    └─────────────────────────→ Arithmetic Intensity
         (FLOPs/byte)

Most LLM operations hit the slanted roof (memory-bound!)
```

---

## **3. Parallelism Communication Patterns**

### **Data Parallel (DP)**
```
Communication: All-reduce gradients
Pattern:  GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3
When:     Once per step (after backward)
Volume:   2P (reduce + broadcast)
```

### **FSDP (ZeRO Stage 3)**
```
Forward:  All-gather params per layer
Backward: Reduce-scatter grads + all-gather params
Pattern:  GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3 (per layer)
Volume:   3P total (1P forward + 2P backward)
```

### **Tensor Parallel (TP)**
```
Communication: All-reduce activations
Pattern:  All GPUs in TP group communicate
When:     4 times per transformer layer (2 forward + 2 backward)
Volume:   2(B×S×H) per all-reduce
```

### **Pipeline Parallel (PP)**
```
Communication: Point-to-point activation/gradient passing
Pattern:  GPU0 → GPU1 → GPU2 → GPU3 (forward)
          GPU0 ← GPU1 ← GPU2 ← GPU3 (backward)
When:     Per microbatch
Volume:   (B/M)×S×H per microbatch
```

---

## **4. Communication Complexity Comparison**

| Strategy | Total Volume per Step | Frequency | Bandwidth Need |
|----------|----------------------|-----------|----------------|
| **DP** | O(P) | 1× | Low (InfiniBand OK) |
| **FSDP** | O(P) (1.5× DP) | Layer-by-layer | Medium |
| **TP** | O(B×S×H×L) | 4L× | **High (needs NVLink)** |
| **PP** | O(B×S×H×N) | 2M(N-1)× | Low (point-to-point) |

### **Example: GPT-3 (175B params, L=96)**
- **DP**: 700 GB (baseline)
- **FSDP**: 1050 GB (1.5× DP)
- **TP**: 1228 GB (if B×S×H large)
- **PP**: 45 GB (smallest!)

---

## **5. Scaling Trade-offs**

### **Data Parallel (DP)**
**Pros:**
- Simplest to implement
- Low communication overhead (once per step)
- Scales across nodes (InfiniBand sufficient)

**Cons:**
- Model must fit in single GPU memory
- Memory redundancy (N copies of model)

**When to use:** Model fits in GPU, want maximum throughput

---

### **FSDP (ZeRO Stage 3)**
**Pros:**
- N× memory reduction (shards everything)
- Overlaps communication with computation
- Can train models that don't fit in single GPU

**Cons:**
- 1.5× more communication than DP
- More complex synchronization

**When to use:** Model doesn't fit in GPU, willing to trade communication for memory

---

### **Tensor Parallel (TP)**
**Pros:**
- Low latency per step (no bubble time)
- Reduces memory per GPU
- Good for large batch sizes

**Cons:**
- Requires fast NVLink (limits to within node)
- High communication frequency (4L per step)
- Doesn't scale beyond ~8 GPUs

**When to use:** Model doesn't fit in GPU, have NVLink, within single node

**Scaling limitation:** Communication becomes bottleneck as TP degree increases
- TP=4: Compute on P/4, then communicate
- TP=8: Compute on P/8 (2× faster), communication unchanged → overhead dominates

---

### **Pipeline Parallel (PP)**
**Pros:**
- Lowest communication volume
- Scales to many GPUs
- Communication doesn't grow with model depth (L)

**Cons:**
- **Bubble time**: ≈ (N-1)/M of time idle (need M ≥ 4N)
- Gradient staleness (using old microbatch gradients)
- Memory overhead (store M microbatches)

**When to use:** Very deep models, can tolerate bubble time

**Microbatch formula:**
- Microbatch size = B / M, M >> N to reduce bubble
- **Bubble time (exact)**: (N-1) / (M+N-1)
- **Bubble time (approximation)**: ≈ (N-1) / M (when M >> N)
- Example: N=4, M=16
  - Exact: 3/19 = 15.8% bubble
  - Approximation: 3/16 = 18.75% bubble

---

## **6. 3D Parallelism (Combining Strategies)**

**Megatron-LM GPT-3 Configuration:**
```
Total: 1024 GPUs = TP × PP × DP
TP = 8  (within node, NVLink)
PP = 16 (across nodes)
DP = 8  (data replicas)

Within each node: 8-way TP (fast NVLink)
Across nodes:     PP stages (activations only)
Outer layer:      DP for throughput
```

**Key insight:** Match parallelism strategy to interconnect speed
- **Fast NVLink (within node)**: Use TP (needs high bandwidth)
- **Slower InfiniBand (across nodes)**: Use PP or DP (lower frequency)

---

## **7. Strong Scaling** (Gap Q183)

**Definition:** Keep problem size fixed, add more devices, aim for proportional speedup

**Example:**
- Train GPT-3 (175B, batch 1024) on 1024 GPUs
- Add to 2048 GPUs
- **Strong scaling**: 2048 GPUs train ~2× faster

**Challenge:** Perfect linear scaling is rare due to communication overhead!

**Contrast with weak scaling:**
- Increase problem size AND devices proportionally
- Keep time per step constant

---

## **8. Interview Q&A**

### **Q: Why is memory bandwidth more important than TFLOPS for LLMs?**
**A**: "Modern GPUs with Tensor Cores have extremely high compute (500 TFLOPS), but limited memory bandwidth (~2-3 TB/s). For LLM workloads with low arithmetic intensity, Tensor Cores spend most time idle waiting for data from HBM. Tim Dettmers found that even for GPT-3 training, Tensor Cores are idle 50% of time. Memory bandwidth is the bottleneck, which is why A100's 1.73× speedup over V100 comes primarily from 1.73× higher bandwidth, not compute improvements."

### **Q: When would you use TP vs DP?**
**A**: "Use TP for large models that don't fit in single GPU memory when you have multiple GPUs with fast NVLink within a node. TP provides low-latency scaling with no bubble time, but requires high bandwidth and doesn't scale beyond ~8 GPUs. Use DP when the model fits in memory—it has minimal communication overhead (one all-reduce per step) and scales across nodes with InfiniBand. For maximum efficiency, combine them: TP within nodes, DP across nodes (3D parallelism)."

### **Q: Explain FSDP vs DP trade-offs**
**A**: "FSDP has 1.5× more communication than DP (3P vs 2P per step), but achieves N× memory reduction by sharding parameters, gradients, and optimizer states across GPUs. FSDP also overlaps all-gather communication with layer computation, hiding latency better than DP's single bulk synchronization. Use DP for models that fit in memory (fastest), FSDP for larger models where memory is the constraint."

### **Q: Why doesn't TP scale beyond 8 GPUs?**
**A**: "Standard GPU nodes have 8 GPUs connected by NVLink (e.g., 8× A100 or H100 per node). TP requires high-bandwidth NVLink for its frequent communication (4L all-reduces per step), and NVLink only connects GPUs within a single node. You cannot use TP across nodes with InfiniBand—it's too slow for TP's communication pattern. Additionally, even with NVLink, as TP degree increases, communication becomes a larger fraction of wall-clock time (compute time decreases as P/N, but communication time stays constant). Beyond 8 GPUs, use PP or DP dimensions instead."

### **Q: What's the main drawback of pipeline parallelism?**
**A**: "Bubble time—GPUs sit idle during pipeline fill and drain. Bubble fraction = (N-1)/(M+N-1) where N is stages and M is microbatches (approximates to (N-1)/M when M >> N). To keep bubble under 25%, need M ≥ 4N. For example, with 4 stages and 16 microbatches: bubble = 3/19 = 15.8%. Additional drawbacks: gradient staleness (using old microbatch gradients) and memory overhead for storing activations."

---

## **9. Key Numbers to Remember**

### **GPU Specs**
- **V100**: 900 GB/s bandwidth, 125 TFLOPS (Tensor Core)
- **A100**: 1555 GB/s (HBM2e), 312 TFLOPS (Tensor Core)
- **H100**: 3000 GB/s (HBM3), 1000 TFLOPS (Tensor Core)

### **Interconnect**
- **NVLink**: ~600 GB/s (within node, GPU-to-GPU)
- **InfiniBand**: ~200 GB/s (across nodes)

### **Communication Volume (GPT-3 example)**
- **P** (model size): 175B params = 350 GB (FP16)
- **DP**: 2P = 700 GB
- **FSDP**: 3P = 1050 GB
- **TP**: ~1200 GB (depends on B×S×H)
- **PP**: ~45 GB (smallest!)

### **Communication Frequency**
- **DP**: 1 per step
- **FSDP**: ~100 per step (per layer)
- **TP**: 4L = 384 per step (L=96)
- **PP**: 2M(N-1) = 448 per step (M=32, N=8)

---

## **10. Common Pitfalls**

❌ **"TP has lower communication than DP"**
✅ TP communicates O(B×S×H×L), often MORE than DP's O(P), but at different frequency

❌ **"FSDP is always better than DP"**
✅ FSDP trades 1.5× communication for N× memory—use DP if model fits!

❌ **"PP has highest communication frequency"**
✅ PP has many point-to-point sends (448), but each is cheaper than TP's all-reduce (384)

❌ **"Microbatch size = B/N"**
✅ Microbatch size = B/M where M is number of microbatches (user choice, M >> N)

❌ **"Use TP across nodes"**
✅ TP needs NVLink (high bandwidth)—only use within a single node!

---

## **11. Communication Volume Calculation Details**

### **What Does "2P" Mean for All-Reduce?**

**Answer:** **Per-device** data sent + received, NOT total traffic across all GPUs.

**Why per-device is standard:**
- Training step time depends on **per-device bandwidth**, not total traffic
- Each GPU must send/receive through its own network link
- Allows comparing strategies independent of GPU count

### **Ring All-Reduce Formula:**

**Per device**: `2P(N-1)/N` where P = data size, N = number of GPUs

**When N is large**: `(N-1)/N → 1`, so ≈ `2P` per device

### **Example (4 GPUs, P=1GB parameters):**

```
Phase 1: Reduce-Scatter (aggregate gradients)
- Each GPU sends P/N to neighbors (N-1) times = 0.25GB × 3 = 0.75 GB sent

Phase 2: All-Gather (broadcast summed result)
- Each GPU sends P/N to neighbors (N-1) times = 0.25GB × 3 = 0.75 GB sent

Per-device total sent: 0.75 GB + 0.75 GB = 1.5 GB
Formula: 2P(N-1)/N = 2 × 1GB × 3/4 = 1.5 GB ✓

Approximation check:
- N=4: 2P(N-1)/N = 1.5 GB vs 2P = 2.0 GB (75% accurate)
- N=1024: 2P(N-1)/N = 1.998 GB vs 2P = 2.0 GB (99.9% accurate)
```

### **General Formulas (Per-Device):**

| Operation | Per-Device Formula | Approximation (N large) |
|-----------|-------------------|-------------------------|
| **All-Reduce** | 2P(N-1)/N | ≈ 2P |
| **All-Gather** | P(N-1)/N | ≈ P |
| **Reduce-Scatter** | P(N-1)/N | ≈ P |

### **Why Per-Device Matters for Training Time:**

**Training step latency = Per-device volume / Bandwidth**

Example (GPT-3, 175B params, InfiniBand 200 GB/s):
- DP: Each GPU sends/receives 2P = 700 GB → 3.5 seconds
- FSDP: Each GPU sends/receives 3P = 1050 GB → 5.25 seconds
- **FSDP is 1.5× slower** (per step, without overlap)

**Doesn't matter if 8 GPUs or 1024 GPUs** - each takes the same time!

### **Total Traffic = Per-Device × N:**

**When total traffic matters:**
- Network switch capacity planning
- Data center bisection bandwidth
- NOT for comparing training speed

**Example**: DP on 1024 GPUs
- Per-device: 700 GB (determines training time)
- Total traffic: 716.8 TB (matters for switch design only)

---

## **Study Tips**

- **Draw the patterns**: Sketch DP, FSDP, TP, PP communication on paper
- **Calculate examples**: Use GPT-3 numbers to compare communication volumes
- **Explain trade-offs**: Practice answering "when to use X vs Y" in 2-3 sentences
- **Connect to Day 1-2**: FSDP is ZeRO Stage 3, TP is Megatron-LM
- **Understand bottlenecks**: Memory bandwidth > compute for modern LLMs
