# Day 22 Quick Reference: Advanced RAG Fundamentals (Week 4 Day 1)

**Date**: 2025-11-18
**Topics**: RRF, SPLADE, DPR, Cross-encoder, ColBERT, MMR, Lost-in-the-middle
**Knowledge Check Score**: 96% (A+)

---

## 1. Reciprocal Rank Fusion (RRF)

**Purpose**: Fuse results from multiple ranking systems (e.g., BM25 + dense retrieval)

**Formula**:
```
RRF_score(d) = Σ_r 1 / (k + rank_r(d))
```
- `r`: Each ranker (BM25, dense, etc.)
- `rank_r(d)`: Document d's rank in ranker r (1 = best)
- `k`: Constant, typically 60

**Key Insight**: k=60 creates "democratic consensus"
- **Small k** (k=1): Winner-take-all (single rank-1 dominates)
- **Medium k** (k=60): Consistent mid-ranks beat extreme ranks
- **Large k** (k=1000): Almost no ranking effect

**Interview Answer**:
"RRF fuses multiple rankers by summing reciprocal ranks with a constant k. With k=60, documents ranked consistently well across rankers beat documents with one excellent and one terrible ranking. It's parameter-free and empirically works well."

---

## 2. SPLADE (Learned Sparse Retrieval)

**Full Name**: Sparse Lexical and Expansion Model

**Purpose**: Bridge BM25 and dense retrieval - learned sparse vectors with vocabulary expansion

**Architecture**:
- BERT encoder → Token logits → log(1 + ReLU(logits))
- Outputs: Sparse vector (30,000-dim vocabulary, 95%+ zeros)

**Training**:
- Contrastive loss (in-batch negatives)
- **FLOPS regularization**: Penalize non-zero activations to encourage sparsity
- Vocabulary expansion: Learn to activate synonym tokens

**Example**:
- Query: "primate in rain forest"
- SPLADE activates: primate, rain, forest, **monkey**, **jungle**, **tropical**

**Storage**: ~10× larger than BM25 (expanded vocabulary), still 10× smaller than dense

**Interview Answer**:
"SPLADE uses BERT to produce learned sparse vectors with vocabulary expansion. FLOPS regularization encourages sparsity. It matches BM25's exact-match capability while adding semantic expansion like dense retrieval, without requiring vector search infrastructure."

---

## 3. DPR (Dense Passage Retrieval)

**Architecture**: Dual-encoder (separate query encoder, document encoder)

**Training**: Contrastive loss with **TWO types of negatives**:

1. **In-batch negatives** (cheap, easier):
   - Other documents in the same batch
   - Free during training (already embedded)
   - Tend to be easier negatives (random docs)

2. **Hard negatives from BM25** (expensive, harder):
   - Top BM25 results that are NOT the gold answer
   - Semantically similar but factually wrong
   - Force model to learn fine-grained distinctions
   - Expensive (require BM25 retrieval per query)

**Why Both**: In-batch for efficiency, hard negatives for difficult discrimination

**Interview Answer**:
"DPR uses dual-encoders trained with contrastive loss. It combines in-batch negatives (efficient, easier) and hard negatives from BM25 (expensive, harder). Hard negatives are critical - they're semantically similar but wrong, forcing the model to learn fine-grained meaning distinctions."

---

## 4. Cross-Encoder Reranking

**Two-Stage Retrieval**:
1. **Stage 1**: Fast retrieval (bi-encoder or BM25) - retrieve 100-1000 docs
2. **Stage 2**: Cross-encoder rerank top-k (k=10-100)

**Cross-Encoder**:
- Input: Concatenated [CLS] query [SEP] doc [SEP]
- Output: Single relevance score
- Can attend across query-doc (better than bi-encoder)
- **10-100× slower** than bi-encoder

**Trade-off**:
- **Accuracy**: Cross-encoder > Bi-encoder > BM25
- **Speed**: BM25 ≈ Bi-encoder >> Cross-encoder
- **Use case**: Bi-encoder for retrieval (millions), cross-encoder for reranking (hundreds)

**Interview Answer**:
"Two-stage retrieval uses fast bi-encoder to retrieve 100-1000 candidates, then slow cross-encoder to rerank top-100. Cross-encoder jointly encodes query+doc for better accuracy but is 10-100× slower, so we limit it to reranking small candidate sets."

---

## 5. ColBERT (Late Interaction)

**Innovation**: Token-level embeddings with late interaction

**Architecture**:
- Query encoder: Outputs L_q token embeddings
- Doc encoder: Outputs L_d token embeddings
- Scoring: MaxSim(q, d) = Σ_{q_i} max_{d_j} sim(q_i, d_j)

**Advantage**: More expressive than bi-encoder (token-level matching)

**Disadvantage**: **100× larger storage**
- Bi-encoder: 1 vector/doc (768-dim)
- ColBERT: L tokens/doc × 768-dim (L ≈ 100 for passages)
- Example: 1M docs = 100M×768×4 bytes = 307 GB

**Compression**: Quantization (8-bit or product quantization) → **12.5-25× reduction**

**Interview Answer**:
"ColBERT stores token-level embeddings and computes MaxSim at query time. It's more accurate than bi-encoder but 100× larger storage (12.5-25× with quantization). Good for small-scale systems or when accuracy is critical."

---

## 6. MMR (Maximal Marginal Relevance)

**Purpose**: Diversity reranking - balance relevance and diversity

**Formula**:
```
MMR(d) = λ × Sim(q, d) - (1-λ) × max_{d'∈Selected} Sim(d, d')
```
- **First term**: Relevance to query (higher = better)
- **Second term**: Similarity to already-selected docs (higher = penalty)
- **λ**: Trade-off parameter (λ=1: pure relevance, λ=0: pure diversity)

**Algorithm**:
1. Start with empty result set
2. For each position:
   - Compute MMR score for all remaining docs
   - Select doc with highest MMR score
   - Add to result set

**Use Case**: Product search, recommendation (avoid redundant results)

**Interview Answer**:
"MMR reranks by maximizing λ×relevance minus (1-λ)×similarity-to-selected. It's a greedy algorithm that penalizes documents similar to already-selected ones. Common in product search to show diverse results."

**CRITICAL**: It's MINUS, not plus! (Penalize redundancy, don't reward it)

---

## 7. Lost-in-the-Middle Problem

**Observation**: LLMs attend more to start and end of context, ignore middle

**Cause**:
- Positional bias in attention
- Middle documents receive less gradient signal during training
- Long context → attention dilution

**Mitigation Strategies**:

1. **Reorder documents**: Put most relevant at start/end
   - Simple, effective
   - Requires confidence in retrieval ranking

2. **Reduce context length**: Fewer documents (top-5 instead of top-20)
   - Sacrifice recall for precision
   - Good when top results are high quality

3. **Compress middle documents**: Summarize or truncate middle docs
   - Preserve diversity while reducing noise
   - Requires summarization quality

**Interview Answer**:
"LLMs attend more to start/end of context, ignoring middle documents. Solutions: (1) reorder to put important docs at edges, (2) use fewer docs, (3) compress/summarize middle docs. Simplest is reordering or reducing context length."

---

## 8. Hybrid Retrieval Patterns

**Why Hybrid**: Combine strengths of sparse (exact match) and dense (semantic)

**Use Cases**:

**Sparse (BM25) wins**:
- Exact match queries (product IDs, technical jargon)
- Short documents with keywords
- Example: "iPhone 15 Pro Max 256GB"

**Dense wins**:
- Semantic/paraphrase queries
- Long documents with context
- Example: "smartphone with best camera under $1000"

**Hybrid wins**:
- Mixed intent (jargon + semantics)
- Example: "BERT model for sentiment analysis"
  - BM25 catches "BERT" (exact match)
  - Dense catches "sentiment analysis" (semantic)

**Fusion Methods**:
- **RRF**: Parameter-free, works well empirically
- **Weighted sum**: Requires tuning weights
- **Learned fusion**: Train model to combine scores

**Interview Answer**:
"Hybrid combines BM25 (exact match, jargon) and dense (semantics, synonyms). Use RRF for parameter-free fusion or learned weights for better performance. Critical for queries with both exact terms (product names) and semantic intent."

---

## Applied Reasoning: 10M Product Search System

**Scenario**: E-commerce with 10M products, 1000 QPS

**Requirements**:
- Handle exact match (SKU, brand names)
- Handle semantic queries ("affordable laptop for students")
- <100ms latency

**Solution**:
1. **Stage 1: Hybrid retrieval** (BM25 + dense bi-encoder)
   - BM25: Exact match for brand/SKU
   - Dense: Semantic matching
   - RRF fusion → top-500

2. **Stage 2: Cross-encoder reranking** (optional)
   - Rerank top-100 from hybrid
   - Adds 20-50ms latency

3. **Stage 3: MMR diversity** (optional)
   - Diversify top-20 for display
   - Adds <5ms

**Why NOT ColBERT**: 10M products × 100 tokens × 768 × 4 bytes = 3 TB (too large)

**Why NOT SPLADE**: Could work, but hybrid BM25+dense is more proven at scale

**Good SPLADE question**: "Could SPLADE replace BM25+dense with a single index?" (Tests understanding of trade-offs)

---

## Key Formulas for Interviews

**RRF**:
```
Score(d) = Σ_r 1 / (k + rank_r(d)),  k=60
```

**MMR** (note the MINUS sign!):
```
MMR(d) = λ × Sim(q,d) - (1-λ) × max_{d'} Sim(d,d')
```

**ColBERT MaxSim**:
```
MaxSim(q,d) = Σ_{q_i} max_{d_j} cos(q_i, d_j)
```

**ColBERT Storage**:
```
Storage = N_docs × L_tokens × d_model × bytes_per_float
Example: 1M × 100 × 768 × 4 = 307 GB (100× larger than bi-encoder)
```

---

## Common Mistakes to Avoid

1. **RRF ranking**: Higher score = rank 1 (NOT rank 2!)
2. **MMR formula**: It's MINUS (penalize redundancy), not plus
3. **DPR negatives**: TWO types (in-batch + hard), not just one
4. **Cross-encoder**: Not a replacement for bi-encoder (too slow for retrieval)
5. **ColBERT storage**: 100× bi-encoder BEFORE compression

---

## Interview-Ready Talking Points

**When asked about retrieval**:
- "Start with hybrid BM25+dense for exact+semantic coverage"
- "Use RRF for fusion - parameter-free and works well"
- "Consider cross-encoder reranking if latency allows (adds 20-50ms)"
- "MMR for diversity if showing multiple results"

**When asked about storage**:
- "Bi-encoder: 768-dim vector per doc (~3KB per doc)"
- "ColBERT: 100× larger (100 vectors per doc), needs compression"
- "SPLADE: 10× BM25 size (vocabulary expansion), 10× smaller than dense"

**When asked about negatives**:
- "DPR uses in-batch (cheap) + hard from BM25 (expensive but crucial)"
- "Hard negatives are semantically similar but wrong - critical for learning"
- "In-batch alone is insufficient for production quality"

**When asked about long context**:
- "Lost-in-the-middle: LLMs ignore middle documents"
- "Solutions: reorder (put important at edges), reduce docs, compress middle"

---

## Next Topics (Day 2-3)

- FiD (Fusion-in-Decoder)
- GraphRAG
- RAFT (Retrieval-Augmented Fine-Tuning)
- Agentic RAG
- Context-aware chunking
- Query understanding
- Evaluation metrics (MRR, NDCG, Recall@k)

---

**Week 4 Progress**: Day 1 complete - 11 topics mastered (7 new + 4 consolidated), 96% knowledge check
