# Day 32: Reference Filtering & Retrieval Evaluation

**Date**: 2025-11-28 (Week 5, Day 4 / Day 32)
**Focus**: Reference filtering, test generation, retrieval evaluation
**Key Finding**: Sparse (BM25) outperforms hybrid retrieval for academic papers

---

## Reference Filtering with Ollama

**Implementation**: Added `filter_reference_chunks()` to `CorpusLoader`
- Uses Ollama (qwen2.5-coder:7b) to classify chunks as content vs references
- Filters during `build_index.py` (before vector indexing)

**Results**:
- Original: 1541 chunks
- Filtered: 146 reference chunks (9.5%)
- Remaining: **1395 content chunks**

**Impact**:
- ✅ Cleaner retrieval corpus
- ✅ Better test question generation
- ✅ Smaller vector index

**Cost**: $0 (Ollama), ~25 min one-time

---

## Test Generation with Ollama

**Setup**: Used Ollama (qwen2.5-coder:7b) with nomic-embed-text embeddings
**Strategy**: Sample ~250 chunks, generate from sampled subset

**Results**: 42 questions generated (exceeded target of 40)

**Quality Distribution**:
- 14 single-hop specific questions
- 14 multi-hop abstract questions
- 14 multi-hop specific questions
- Query lengths: 4 short, 7 medium, 3 long, 28 unspecified

**Quality Check**: ✅ No reference-section questions, all legitimate research questions

**Cost**: $0 (Ollama), ~45 min

---

## Retrieval Evaluation Results

**Setup**: 41 questions (42 generated, 1 failed ID mapping)

### Results (Recall@5, MRR, NDCG)

| Method | Context Recall@5 | MRR | NDCG |
|--------|------------------|-----|------|
| **SPARSE (BM25)** | **62.20%** ✅ | 0.4988 | 0.4856 |
| HYBRID (RRF) | 53.66% | 0.5037 | 0.4563 |
| DENSE (SentenceBERT) | 36.59% | 0.2610 | 0.2691 |

**Observed**: Sparse achieves highest recall on incomplete ground truth (62% vs 54% vs 37%)
**Caveat**: True ranking uncertain due to triple incompleteness (see below)

---

## Hypotheses

### Hypothesis 1: BM25 Better for Academic Papers (Lexical Matching)

**Possible explanation**:
- BM25 excels at exact term matching (critical for technical jargon)
- Generic embeddings (SentenceBERT) struggle with specialized ML terminology
- Hybrid fusion dilutes strong BM25 signal without learned weights

**Caveat**: Cannot confirm without complete ground truth

### Hypothesis 2: Incomplete Ground Truth Bias

**Equally plausible explanation**:
- Dense retrieves MORE diverse relevant chunks (across corpus) → more penalized for out-of-sample hits → appears at 37%
- Sparse retrieves chunks that happen to be in 250-sample more often → less penalized → appears at 62%
- True ranking could be reversed or equal

**Why this matters**: Method appearing "worse" might actually be retrieving better diverse content

### RRF k Tuning (No Impact)

Tested k=20, k=60, k=100:
- Hybrid recall: 53.66% (unchanged)
- MRR variance: <0.5%
- **Conclusion**: Problem is fundamental (weak dense signal), not fusion parameter

---

## Evaluation Limitation: Triple Incompleteness

**Critical insight discovered by user**: Ground truth is incomplete in THREE ways

### 1. Sample Incompleteness
- Test generation: 250 sampled chunks → Questions
- Evaluation: Retrieve from 1395 full chunks
- Ground truth: Only includes 250 sample
- **Impact**: Relevant chunks outside sample marked "wrong"

### 2. Ragas Incompleteness
- Ragas marks chunks USED for generation, not ALL relevant chunks
- Single-hop: Generated from 1 chunk → only that chunk marked
- Multi-hop: Generated from 2-3 chunks → only those marked
- **Unknown**: Does Ragas mark ALL relevant chunks even within the sample?
- **Likely**: No - only generation sources are marked

### 3. Combined Effect
- Metrics are **severe lower bounds**
- Retrieved relevant chunks (out-of-sample OR unmarked-in-sample) = "wrong"
- **Cannot determine true ranking** from these metrics

**Example**:
```
Question generated from chunk A (in sample); actually relevant to A, B, C in the full corpus
Labeled ground truth: [A]
Retrieved: [B, C]
Measured recall: 0% (0/1)
Actual recall: 66.7% (2/3)
```

**Solutions considered**:
1. Regenerate from full corpus ($1 / 60 min) - Won't work, see Ragas Incompleteness above
2. Accept directional metrics, document limitation ✅ (chosen)
3. Use LLM-based context_recall (Ragas) instead

---

## Interview Talking Points

**Intellectual honesty about evaluation**:
> "I compared three retrieval strategies on 41 questions. On incomplete ground truth, sparse (BM25) showed 62% recall@5 vs hybrid 54% vs dense 37%. However, I discovered triple incompleteness: (1) questions generated from 250-sample but evaluated on 1395-chunk corpus, (2) Ragas only marks chunks used for generation, not all relevant chunks, and (3) combined effect makes metrics severe lower bounds. The true ranking is uncertain—the method appearing 'worst' might actually retrieve better diverse content that's unmarked. This taught me that evaluation validity matters more than impressive numbers."

**What I would do differently**:
> "For valid comparison, I'd either: (1) use LLM-as-judge for relevance (Ragas context_recall), or (2) manually label 20 retrieved results to estimate bias direction. The key insight: alignment between test generation and evaluation scope is critical for meaningful metrics."

---

## Design Decisions for Portfolio

**Decision pending** (requires valid evaluation):

**Option 1**: Use LLM-based evaluation (Ragas context_recall)
- Avoids ground truth incompleteness
- Measures actual relevance via LLM judgment
- Provides valid comparison across methods

**Option 2**: Keep all methods, document evaluation limitations
- Shows hybrid retrieval implementation
- Demonstrates rigorous evaluation thinking
- Explains why metrics are inconclusive (intellectual honesty)
- Lists what would be needed for valid comparison

**Option 3**: Manual spot-check (20 samples)
- Review retrieved results for 20 questions
- Estimate bias direction (which method penalized most?)
- Make informed decision based on qualitative analysis

---

## Next Steps (Day 33)

1. **RAG evaluation** (`evaluate_rag.py`)
   - Use Ragas metrics (LLM-based context_recall avoids ground truth issue)
   - Test on 52 questions (10 manual + 42 Ragas)
   - Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevance

2. **Error analysis**
   - Categorize failure modes
   - Identify improvement opportunities

3. **Docker + Streamlit** (if time)
   - Deploy system
   - Interactive demo

---

**Files Created**:
- `src/data_loader.py`: `filter_reference_chunks()` method
- `evaluation/evaluate_retrieval.py`: Retrieval metrics implementation
- `data/eval/ragas_testset.jsonl`: 42 Ollama-generated questions

**Key Metrics**:
- Sparse recall: 62.20% (best)
- Test generation cost: $0 (Ollama)
- Reference filtering: 9.5% removed
