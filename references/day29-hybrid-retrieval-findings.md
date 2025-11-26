# Day 29: Hybrid Retrieval Implementation & Findings

**Date**: 2025-11-25 (Week 5 Day 1)
**Time**: 2 hours
**Goal**: Implement BM25 + RRF hybrid retrieval and compare with dense-only FAISS

---

## Implementation Summary

### Components Built

1. **BM25Retriever** (`src/sparse_retrieval.py`):
   - NLTK word tokenizer with punctuation filtering
   - rank-bm25 library (BM25Okapi implementation)
   - Returns top-k chunks by BM25 score

2. **HybridRetriever** (`src/hybrid_search.py`):
   - Combines FAISS (dense) + BM25 (sparse)
   - Reciprocal Rank Fusion (RRF): `score = Σ 1/(k + rank_i)` where k=60
   - Retrieves 4×k candidates from each method, fuses to top-k

3. **Evaluation Framework** (`experiments/compare_retrievers.py`):
   - Manual relevance judgments for Precision@5 and MRR@5
   - Automated calculation from judgment data
   - **Note**: MRR@5 = reciprocal rank within top-5 only (not true MRR across all 1541 chunks)

---

## Evaluation Results

### Evaluation 1: General NLP Queries (on 32 RAG papers, 1541 chunks)

| Query | Dense P@5 / MRR@5 | Sparse P@5 / MRR@5 | Hybrid P@5 / MRR@5 |
|-------|-----------------|------------------|------------------|
| What is attention mechanism in transformers? | 80% / 1.000 | 20% / 0.333 | 40% / 0.500 |
| How does BERT pretraining work? | 60% / 1.000 | 0% / 0.000 | 40% / 0.500 |
| What is the difference between GPT and BERT? | 40% / 0.500 | 20% / 0.333 | 40% / 1.000 |
| **AVERAGE** | **60% / 0.833** | **13.3% / 0.222** | **40% / 0.667** |

### Evaluation 2: RAG-Focused Queries (same corpus)

| Query | Dense P@5 / MRR@5 | Sparse P@5 / MRR@5 | Hybrid P@5 / MRR@5 |
|-------|-----------------|------------------|------------------|
| What is dense passage retrieval? | 40% / 0.500 | 20% / 0.333 | 40% / 1.000 |
| How does ColBERT work? | 100% / 1.000 | 100% / 1.000 | 100% / 1.000 |
| What is the difference between sparse and dense retrieval? | 60% / 1.000 | 80% / 1.000 | 100% / 1.000 |
| **AVERAGE** | **66.7% / 0.833** | **66.7% / 0.778** | **80% / 1.000** |

### Key Finding: **Query-Corpus Alignment Matters!** ✅

**Evaluation 1 (Mismatch)**: Hybrid (40%) < Dense (60%) ❌
**Evaluation 2 (Aligned)**: Hybrid (80%) > Dense (66.7%) ✅

**BM25 Improvement**: 13.3% → 66.7% (+53.4%) when queries match corpus!

---

## Root Cause Analysis

### Why Did Hybrid Underperform?

**1. Corpus Mismatch**:
- **Queries**: General NLP/transformer concepts (attention, BERT, GPT)
- **Corpus**: RAG-specific papers (DPR, ColBERT, SPLADE, retrieval methods)
- **Problem**: Few documents actually explain attention mechanisms or BERT pretraining

**2. BM25 Retrieved Poor Results**:
- BM25 relies on **lexical matching** (exact word overlap)
- Query: "attention mechanism" → BM25 matched "An At..." in references (false positive!)
- Query: "BERT pretraining" → BM25 found no relevant docs (0% precision)

**3. RRF Fusion Polluted Good Dense Results**:
- Dense FAISS: Found semantically similar chunks (even without exact words)
- BM25: Retrieved mostly irrelevant chunks
- RRF: Averaged ranks → promoted BM25's bad results, demoted Dense's good results

**4. Reference Section Noise** (Discovered via analysis):
- **15% of chunks (231/1541) are reference/bibliography sections**
- BM25 matched citations instead of content: "An Attention Free Transformer" in bibliography when searching "attention"
- These false positives degraded BM25 precision
- **Potential fix**: Filter reference sections during chunking (could improve precision by 5-10%)

**Example (Query 1)**:
```
Dense Top-2:  [chunk_450 ✓, chunk_29250 ✓]  ← Both relevant
Sparse Top-2: [chunk_26550 ✗, chunk_4500 ✗]  ← Both irrelevant (references)

RRF Fusion:
  chunk_26550: 1/(60+1) = 0.0164  ← Sparse rank 1 (but irrelevant!)
  chunk_450:   1/(60+1) = 0.0164  ← Dense rank 1 (relevant)

Result: chunk_26550 appears in Hybrid top-5, displaces relevant chunks from Dense!
```

---

## When Does Hybrid Retrieval Help?

### Hybrid Works When:
1. **Both methods find different relevant docs**:
   - Dense finds: semantically similar (paraphrased)
   - Sparse finds: exact keyword matches
   - RRF combines strengths

2. **Corpus has diverse document types**:
   - Technical docs (BM25 wins on keywords)
   - Descriptive docs (Dense wins on semantics)

3. **Queries have both semantic + keyword signals**:
   - Example: "transformer attention mechanism pytorch implementation"
   - Semantic: "transformer", "attention"
   - Keyword: "pytorch", "implementation"

### Hybrid Fails When:
1. **One method is much worse** (our case):
   - BM25 precision: 13.3% ❌
   - Dense precision: 60.0% ✅
   - Fusion averages down the good results!

2. **Corpus is semantically coherent but lexically diverse**:
   - Our corpus: All RAG papers (semantically similar)
   - But use different terminology (DPR, ColBERT, ANCE, etc.)
   - BM25 can't generalize across terminology

3. **Queries are conceptual, not keyword-based**:
   - "How does X work?" (conceptual)
   - vs "Find paper about X with Y metric" (keyword-based)

---

## Lessons Learned

### 1. Hybrid ≠ Always Better
- **Conventional wisdom**: "Hybrid combines best of both worlds"
- **Reality**: Only true when both methods contribute good results
- **Takeaway**: Evaluate each method independently first

### 2. Corpus-Query Alignment Matters ⭐ **CRITICAL FINDING**
- Our queries (general NLP) don't match corpus (RAG papers)
- **When we switched to RAG-focused queries, BM25 improved 5× (13% → 67%)**
- Better test queries for this corpus:
  - "What is dense passage retrieval?" → Dense 40%, Sparse 20%, Hybrid 40%
  - "How does ColBERT work?" → **All 100%!** (Perfect keyword match)
  - "Compare BM25 vs dense retrieval" → Dense 60%, Sparse 80%, **Hybrid 100%**

**Key insight**: BM25 excels when queries contain specific terms that appear in the corpus (e.g., "ColBERT"), but fails on conceptual queries (e.g., "How does pretraining work?")

### 3. BM25 Limitations
- **Vocabulary mismatch**: Can't match "attention" to "self-attention mechanism"
- **No semantic understanding**: Matches "attention" in references too
- **Needs exact words**: Fails on paraphrased content

### 4. When to Use Each Method

| Method | Best For | Weaknesses |
|--------|----------|------------|
| **Dense (FAISS)** | Semantic similarity, paraphrasing, cross-lingual | Misses exact keywords, slower indexing |
| **Sparse (BM25)** | Exact keywords, named entities, IDs/codes | No semantic understanding, vocabulary gap |
| **Hybrid (RRF)** | Diverse corpus + diverse queries | Only as good as weakest component |

---

## Interview Talking Points

### Question: "Why didn't hybrid retrieval outperform dense-only?"

**Strong answer**:
> "I ran two evaluations. With general NLP queries, hybrid (40%) underperformed dense-only (60%) because BM25 retrieved mostly irrelevant results (13.3%). However, when I tested with RAG-focused queries matching my corpus, BM25 improved 5× to 67% and hybrid achieved 80% precision. This taught me that query-corpus alignment is critical—BM25 performs well on keyword-specific queries like 'How does ColBERT work?' (100% precision) but fails on conceptual queries. I also discovered 15% of chunks were reference sections causing false positives. For production, I'd use query classification to route different query types to different retrieval strategies."

**What makes this strong**:
- Honest about unexpected results (shows rigor)
- Root cause analysis (vocabulary mismatch)
- Forward-thinking (adaptive fusion, monitoring)
- Data-driven (cites specific numbers)

### Question: "When would you use hybrid vs dense-only?"

**Strong answer**:
> "It depends on the query and corpus characteristics. For my RAG corpus with conceptual queries, dense-only worked better (60% vs 40%). But hybrid shines when: (1) queries have both semantic and exact-keyword needs, like 'transformer attention mechanism pytorch implementation', (2) the corpus has diverse document types where BM25 and dense retrieval find complementary results, and (3) both methods achieve reasonable individual performance. I'd A/B test in production and potentially use query classification to route different query types to different retrieval strategies."

### Question: "Why did you use Precision@5 instead of Recall@5 for evaluation?"

**Strong answer**:
> "I didn't have ground truth annotations for my corpus. Both Recall and true MRR require knowing the complete ranking of all relevant documents. For Recall@5, I'd need the total number of relevant documents—that's Recall@K equals relevant retrieved divided by total relevant in corpus. For true MRR, I'd need to find the rank of the first relevant document across all 1,541 chunks, not just top-5. To get either metric properly, I'd need to manually label all 1,541 chunks for each query—that's 4,623 judgments for just 3 queries. Precision@5 and MRR@5 only require judging the top-5 results per method, so 90 judgments for 6 queries across 3 methods—totally feasible. Precision@5 tells me 'of the results shown to users, what fraction is relevant?', and MRR@5 tells me 'how quickly do users find the first useful result in top-5?', which are the most important user experience metrics. Later, for more comprehensive evaluation, I'd use Ragas with LLM-as-judge to automate relevance assessment and estimate recall metrics across the full corpus."

**What makes this strong**:
- Acknowledges constraint (no ground truth)
- Correctly identifies BOTH Recall and MRR have the same limitation
- Explains trade-off (manual effort: 90 vs 9,246 judgments for 6 queries)
- User-centric reasoning (Precision and MRR@5 measure what users see)
- Forward-thinking (mentions automated evaluation with Ragas)
- Demonstrates deep understanding of metric definitions and their requirements

---

## Next Steps (Week 5 Day 2-5)

**Day 2 (Nov 26)**: Generation pipeline
- Use Dense-only retrieval (best performer)
- Implement RAG with GPT-3.5-turbo
- Test zero-shot vs few-shot prompting

**Day 3 (Nov 27)**: Ragas evaluation
- Automated relevance judgments
- Better test query set (RAG-focused)
- Re-evaluate hybrid vs dense

**Day 4-5**: Docker + Streamlit + Deploy

---

## Technical Details

### Implementation Decisions

**1. Tokenization: NLTK vs Simple Whitespace**
- Chose NLTK word_tokenize for proper handling of:
  - Contractions: "don't" → ["do", "n't"]
  - Compounds: "state-of-the-art" (preserved)
  - Punctuation: Filtered intelligently
- Alternative considered: ICU BreakIterator (Unicode-aware, better for multi-lingual)

**2. RRF Parameters**
- **k=60** (standard from literature)
- **retrieve_k=4×final_k** (retrieve 20, return top-5)
- **1-based ranking** (per original RRF paper)

**3. BM25 Parameters**
- Used defaults: k1=1.5, b=0.75, epsilon=0.25
- Could tune for corpus, but likely minor impact given root cause

**4. Evaluation Metrics: Why Precision@5 and MRR, Not Recall?**

**Problem**: No ground truth annotations for our corpus
- Don't know which documents are relevant for each query
- Recall requires: Recall@K = (# relevant retrieved) / (total # relevant in corpus)
- Can't calculate "total # relevant in corpus" without labeling all 1541 chunks

**Solution**: Use metrics that only require top-K judgments

**Precision@5**:
- Formula: (# relevant in top-5) / 5
- Only requires judging 5 results per query (15 judgments per query for 3 methods)
- Measures: "Of the top results shown, what fraction is relevant?"
- Good for: Comparing retrieval quality across methods with limited manual effort

**MRR (Mean Reciprocal Rank)**:
- Formula: 1 / rank_of_first_relevant_document (in entire ranking)
- **What we calculated**: MRR@5 = 1 / rank_of_first_relevant_in_top_5
- **Important limitation**: We only judged top-5, so we don't know if there's a MORE relevant doc at position 6, 7, ..., 1541
- Measures: "How quickly does user find first useful result?"
- Good for: Assessing user experience (most users click first good result)
- **Same ground truth problem as Recall**: Requires knowing the complete ranking

**Why not Recall@5?**
- Would need to manually label all 1541 chunks for each query
- 3 queries × 1541 chunks = 4,623 judgments (infeasible!)
- vs Precision: 3 queries × 3 methods × 5 results = 45 judgments ✅

**Trade-off**:
- **Precision@5**: Tells us quality of top results (what user sees) ✅ No ground truth needed
- **MRR@5**: Tells us rank of first relevant in top-5 ✅ No ground truth needed
- **Recall@5**: Tells us coverage (are we missing good ones?) ❌ Needs ground truth (all 1541 chunks)
- **True MRR**: Rank of first relevant in ALL results ❌ Needs ground truth (all 1541 chunks)
- For Day 29: Precision@5 and MRR@5 sufficient for comparing methods
- For Day 32 (Ragas): Automated metrics will estimate recall and coverage with LLM judges

### Code Statistics
- `sparse_retrieval.py`: 39 lines
- `hybrid_search.py`: 58 lines
- `compare_retrievers.py`: 100 lines
- `evaluation_results.py`: 157 lines (includes manual judgments)
- **Total**: ~350 lines for full hybrid retrieval system

---

## Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Corpus size | 1541 chunks | 32 RAG papers |
| Chunk size | 500 tokens | tiktoken, 50 overlap |
| Index size (FAISS) | 384 dims | all-MiniLM-L6-v2 |
| BM25 index build time | ~2 seconds | Rebuild each time (no persistence) |
| FAISS load time | <1 second | Pre-built index |
| Query latency | ~50ms | Both FAISS + BM25 + RRF |
| Test queries | 6 total | 3 general NLP + 3 RAG-focused |
| Total judgments | 90 | 6 queries × 3 methods × 5 results |

---

## References

- **RRF Paper**: Cormack, Clarke, Büttcher (2009) - "Reciprocal Rank Fusion outperforms Condorcet and individual systems"
- **BM25**: Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
- **NLTK Tokenization**: Bird, Klein, Loper (2009) - "Natural Language Processing with Python"
- **rank-bm25**: Python library - https://github.com/dorianbrown/rank_bm25

---

**Last Updated**: 2025-11-25 (Day 29)
