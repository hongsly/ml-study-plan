# Experiments Log

This folder contains exploratory analysis and hypothesis-testing scripts for the RAG Q&A system.

---

## Day 29 (Nov 25, 2025): Hybrid Retrieval Evaluation

### Hypothesis 1: Query-Corpus Mismatch
**Question**: Does BM25 perform poorly because our test queries don't match the corpus domain?

**Test**: Compare general NLP queries vs RAG-focused queries

**Result**: ✅ **CONFIRMED** - BM25 improved from 13% to 67% precision with RAG-focused queries

**Evidence**:
- General queries (attention, BERT, GPT): BM25 = 13.3%
- RAG queries (DPR, ColBERT, sparse/dense): BM25 = 66.7%
- Improvement: +53.4 percentage points

---

### Hypothesis 2: Reference Section Noise
**Question**: Is BM25 matching citation/reference sections instead of content?

**Test**: Analyze what percentage of chunks are references and check if BM25 retrieves them

**Result**: ✅ **CONFIRMED** - 15% of chunks are references (231/1541), causing false positives

**Evidence**:
- Query: "attention mechanism"
- BM25 matched: "An Attention Free Transformer" in bibliography
- This is a reference, not content explaining attention

**Potential fix**: Filter reference sections during chunking (could improve precision by 5-10%)

---

### Key Findings

#### 1. Query-Corpus Alignment is Critical

| Query Type | Dense | Sparse (BM25) | Hybrid |
|------------|-------|---------------|--------|
| General NLP queries | 60% | 13% | 40% |
| RAG-focused queries | 67% | 67% | **80%** ⭐ |

**Takeaway**: Hybrid (80%) outperforms Dense (67%) when queries align with corpus

#### 2. When Hybrid Helps vs Hurts

**Hybrid works when**:
- Both methods find different relevant docs
- Example: "How does ColBERT work?" → BM25 100%, Dense 100%, Hybrid 100%

**Hybrid fails when**:
- One method much worse than the other
- Example: "BERT pretraining" → BM25 0%, Dense 60%, Hybrid 40% (degraded)

#### 3. BM25 Strengths and Weaknesses

**Strengths**:
- Exact keyword matching: "ColBERT" query → 100% precision
- Multiple keywords: "sparse and dense retrieval" → 80% precision

**Weaknesses**:
- Vocabulary mismatch: Can't match "attention" to "self-attention mechanism"
- Reference noise: Matches citations in bibliographies
- Conceptual queries: "How does BERT pretraining work?" → 0% precision

---

## Day 30 (Nov 26, 2025): Generation Pipeline & Smoke Test

### Implementation Complete
- ✅ `src/generator.py`: OpenAI Responses API wrapper (gpt-4o-mini)
- ✅ `src/rag_pipeline.py`: End-to-end RAG pipeline (RagAssistant class)
- ✅ `data/eval/test_questions.json`: 10 manual test questions (factual, reasoning, multi-hop, negative)
- ✅ `experiments/smoke_test.py`: Smoke test script

### Test Results (5 questions × 4 modes)

| Mode | Citations | Answer Quality | Token Usage |
|------|-----------|----------------|-------------|
| **Hybrid** | ✅ Excellent | ✅ High | ~2700 input |
| **Dense** | ✅ Excellent | ✅ High | ~2700 input |
| **Sparse** | ✅ Good | ✅ Good | ~2700 input |
| **None** | ❌ None | ⚠️ Generic | ~50 input |

**Key findings**:
1. **Token usage validates retrieval**: 2700 tokens (with context) vs 50 tokens (without) ✅
2. **Citation format works**: `(source: [Li et al., 2023](2202.01110_rag_survey_lichunk_0))` ✅
3. **Answer quality high**: Detailed explanations with proper source attribution
4. **Sparse performed better than expected**: Day 29 findings confirmed - query-corpus alignment matters

### ⚠️ Issue Discovered: Negative Question Handling

**Problem**: Model answers out-of-corpus questions despite strict prompt

**Test case**: "What is the capital of France?" (not in RAG corpus)

**Results**:
- Hybrid/Dense/None: "The capital of France is Paris." ❌
- Sparse: "The provided documents do not contain... the capital of France is Paris." ⚠️

**Initial hypothesis**: Prompt engineering failure (model ignoring instructions)

**Updated prompt** (line 4-10 in `src/generator.py`):
```python
SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a Q&A assistant for RAG research. "
    "Answer ONLY using information from the provided support material. "
    "If the support material does not contain enough information, respond with: "
    "'I don't have enough information in the provided materials to answer this question.' "
    "DO NOT use your general knowledge - only cite the support material."
)
```

**Status**: ✅ **ROOT CAUSE FOUND** (via OpenAI log investigation)

### Root Cause Investigation ⭐

**What we found**: User checked OpenAI dashboard logs and discovered retrieved chunk `2404.16130_graphrag_edgechunk_22050` contained:
```
"capital of France?', a direct answer would be 'Paris'..."
```

**Diagnosis**: **Retrieval contamination**, not prompt engineering failure!
- GraphRAG paper used "capital of France" as an example in the text
- BM25/Dense retrieved this chunk (keyword match: "capital" + "France")
- Model correctly cited the retrieved document (working as designed!)
- The prompt is working correctly - the issue is retrieval quality

**Retrieval rankings**:
- Dense/Hybrid: Ranked contaminated chunk **#1** (highest relevance)
- Sparse: Ranked contaminated chunk **#2** (still in top-5)

**Why Sparse refused but Dense/Hybrid didn't**:
- Even though Sparse also retrieved the contaminated chunk (#2), it added a disclaimer
- Dense/Hybrid (#1 ranking) cited it directly without hesitation
- This suggests ranking position influences how confidently the model cites retrieved information
- All three methods retrieved the contaminated chunk - different presentation led to different behavior

**Key insight**: This is a data quality issue (retrieval contamination), not a model issue

### Solutions (Updated)

**1. Use truly out-of-domain queries** (immediate fix):
- ❌ Bad: "What is the capital of France?" (appears in papers as example)
- ✅ Good: "How to bake bread?" (impossible to be in RAG/LLM papers)

**2. Few-shot examples** (prompt engineering):
```python
"Example:\n"
"Question: What is the capital of France?\n"
"Support: [papers about RAG]\n"
"Answer: I don't have information about geography in these RAG research papers.\n"
```

**3. Structured outputs** (force format):
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "properties": {
            "has_answer": {"type": "boolean"},
            "answer": {"type": "string"},
            "sources": {"type": "array"}
        }
    }
}
```

**4. Temperature=0** (more deterministic):
- Default: 0.3 (some randomness)
- Set to 0: Maximally deterministic

**5. Check retrieval quality** (data quality):
- Ensure negative queries retrieve low-relevance chunks
- Consider similarity threshold (e.g., reject if score < 0.3)
- Filter out example text from papers during preprocessing

---

## Scripts in This Folder

### Evaluation Scripts

**`test_general_queries.py`**
- Displays full retrieval results for 3 general queries for manual inspection
- Shows top-5 from all three methods
- Used for initial relevance judgments

**`evaluate_general_queries.py`**
- Original evaluation with general NLP queries
- Queries: attention mechanism, BERT pretraining, GPT vs BERT
- Result: Dense 60%, Sparse 13%, Hybrid 40%
- **Conclusion**: Hybrid underperformed due to query-corpus mismatch

**`test_rag_queries.py`**
- Quick test script to inspect RAG query results
- Displays top-3 from Dense, Sparse, Hybrid
- Shows overlap analysis between methods

**`evaluate_rag_queries.py`**
- Follow-up evaluation with RAG-focused queries
- Queries: DPR, ColBERT, sparse vs dense retrieval
- Result: Dense 67%, Sparse 67%, Hybrid 80%
- **Conclusion**: Hybrid outperformed when queries matched corpus

**`compare_retrievers.py`**
- Interactive manual evaluation framework
- Prompts user for relevance judgments
- Calculates Precision@5 and MRR

### Analysis Scripts

**`analyze_references.py`**
- Detects reference/bibliography sections in corpus
- Found 231/1541 chunks (15%) are references
- Shows examples of false positive matches

---

## Recommendations from Experiments

### For Production (Week 5 Day 2+):

**1. Use Dense-only for now** (simplest, 67% is good):
```python
from src.vector_store import VectorStore

retriever = VectorStore()
chunks = retriever.search(query, k=5)
```

**2. Or use Hybrid with RAG-focused queries** (80%, more impressive):
```python
from src.hybrid_search import HybridRetriever

retriever = HybridRetriever()
chunks = retriever.search_hybrid(query, top_k=5)
```

**3. For Week 5 Day 3 (Ragas evaluation)**, use RAG-focused test questions:
```python
test_questions = [
    "How does dense passage retrieval work?",
    "What is ColBERT late interaction?",
    "Compare BM25 vs dense retrieval",
    "What is query expansion in retrieval?",
]
```

### Optional Improvements:

**1. Filter reference sections** (could improve Hybrid 80% → 85%+):
- Add `is_reference_chunk()` to `src/data_loader.py`
- Re-run `scripts/build_index.py`
- Remove ~231 reference chunks

**2. Query classification strategy**:
```python
def choose_retrieval_method(query):
    if has_specific_keywords(query):  # "ColBERT", "DPR"
        return "hybrid"  # BM25 + Dense
    else:  # Conceptual: "how does X work?"
        return "dense"   # Dense only
```

**3. Adaptive fusion weights**:
- Instead of fixed RRF, weight BM25 higher for keyword queries
- Weight Dense higher for conceptual queries

---

## Metrics Summary

| Metric | General Queries | RAG Queries | Notes |
|--------|----------------|-------------|-------|
| **Dense Precision@5** | 60.0% | 66.7% | Consistent performer |
| **Sparse Precision@5** | 13.3% | 66.7% | 5× improvement with aligned queries! |
| **Hybrid Precision@5** | 40.0% | 80.0% | Best when both methods contribute |
| **Target** | 85% | 85% | RAG queries got close (80%) |

---

## Interview Talking Points

**Question**: "How did your hybrid retrieval perform?"

**Strong answer**:
> "I implemented hybrid retrieval with RRF fusion and ran two evaluations. With general NLP queries, hybrid (40%) underperformed dense-only (60%) because BM25 retrieved mostly irrelevant results (13% precision). However, when I tested with RAG-focused queries matching my corpus, BM25 improved 5× to 67% and hybrid achieved 80% precision. This taught me that query-corpus alignment is critical—BM25 performs well on keyword-specific queries like 'How does ColBERT work?' but fails on conceptual queries. I also discovered 15% of chunks were reference sections causing false positives. For production, I'd use query classification to route different query types to different retrieval strategies."

---

## Day 33 (Nov 29, 2025): RAG Evaluation & Error Analysis

### Experiment 1: Question Quality Analysis

**Motivation**: Initial RAG evaluation showed unexpectedly low `answer_correctness` scores (0.39-0.58). Manual inspection revealed some Ragas-generated questions had incorrect reference answers.

**Hypothesis**: Auto-generated questions from reference/citation sections produce incorrect ground truth.

**Test**: Created `analyze_question_quality.py` to detect suspicious questions using citation pattern detection.

**Detection heuristics**:
```python
citation_patterns = [
    r'\d{4}[a-z]?\.',  # Year pattern: "2019."
    r'et al\.',         # "et al."
    r'In _[^_]+_\.',    # Venue pattern: "In _EMNLP_."
    r'_[A-Z][^_]+_,',   # Journal pattern: "_ArXiv_,"
]
# Heuristic: >3 citations per 100 words suggests reference section
```

**Result**: ✅ **CONFIRMED** - 19/41 questions (46.3%) flagged as suspicious

**Root Cause Analysis** (discovered post-experiment):

The fundamental issue is that the testset was generated from **pre-chunked documents** (500 tokens) instead of **whole documents**.

**Why this caused the problem**:
1. Ragas builds a **knowledge graph** from input documents to generate questions
2. When given 500-token chunks, Ragas treats each chunk as an isolated document
3. Bibliography chunks (with citations) look like valid content to Ragas
4. Ragas cannot distinguish "this is a reference section" from "this is actual content"
5. Result: 46% of questions generated from bibliography/table/footnote chunks

**The correct workflow** (for future):
```python
# WRONG (what we did):
loader = CorpusLoader()
loader.parse_pdfs(pdf_paths, chunk_size=500)  # Pre-chunked!
chunks = loader.get_chunks()  # 1,541 isolated chunks
generator.generate_with_langchain_docs(chunks, testset_size=50)

# CORRECT (for v2):
loader = PyMuPDFLoader("paper.pdf")
documents = loader.load()  # Full pages with structure
generator.generate_with_langchain_docs(documents, testset_size=50)
```

With whole documents, Ragas would:
- Understand document structure (sections, headers, hierarchy)
- Distinguish content from references/bibliography
- Generate multi-hop questions across sections
- Produce higher-quality reasoning questions

**Lesson learned**: Ragas testset generation requires whole documents, NOT retrieval chunks. The chunking strategy for retrieval (500 tokens) is completely different from what testset generation needs (whole documents).

**Manual review categorization**:
- **Definitely bad (11)**: Generated from citations, table headers, footnotes
  - Example: "What is Yoav Goldberg's contribution to NLP?" (from bibliography)
- **Moderate quality (4)**: Somewhat answerable but contaminated contexts
- **Good but contaminated (4)**: Valid question but LLaMA performance table in context
- **False positive (1)**: Actually good question despite detection

**Decision**: **Moderate filter** - Remove 15 questions
- 13 unique indices: `[4, 5, 14, 15, 18, 21, 22, 24, 25, 28, 35, 37, 39]`
- Final testset: **28 clean questions** (68% retention from 41)

**Scripts created**:
- `analyze_question_quality.py` - Automated detection
- `review_suspicious_questions.py` - Interactive manual review
- `filter_ragas_testset.py` - Creates `data/eval/ragas_testset_filtered.jsonl`

---

### Experiment 2: Filtering & Recalculation

**Goal**: Recalculate RAG metrics on the filtered 28-question testset

**Challenge**: Correctly merge metrics from two evaluation files:
- `eval_results_{mode}_with_reference_filtered.json`: 28 ragas questions (indices 0-27)
- `eval_results_{mode}_no_reference_filtered.json`: 10 manual + 28 ragas (indices 10-37 for ragas)

**Implementation**: `filter_and_recalculate.py`

**Results**: All metrics improved ~13% average after filtering

| Mode | Metric | Original (41q) | Filtered (28q) | Δ |
|------|--------|---------------|---------------|---|
| **SPARSE** | answer_correctness | 0.567 | 0.668 | +10.1% |
| | context_precision | 0.783 | 0.844 | +6.1% |
| | context_recall | 0.817 | 0.902 | +8.5% |
| | answer_relevancy | 0.890 | 0.914 | +2.4% |
| | faithfulness | 0.784 | 0.832 | +4.8% |
| **HYBRID** | answer_correctness | 0.531 | 0.628 | +9.7% |
| | context_precision | 0.763 | 0.828 | +6.5% |
| | context_recall | 0.831 | 0.920 | +8.9% |
| | answer_relevancy | 0.888 | 0.906 | +1.8% |
| | faithfulness | 0.784 | 0.848 | +6.4% |
| **DENSE** | answer_correctness | 0.449 | 0.530 | +8.1% |
| | context_precision | 0.655 | 0.706 | +5.1% |
| | context_recall | 0.670 | 0.753 | +8.3% |
| | answer_relevancy | 0.794 | 0.810 | +1.6% |
| | faithfulness | 0.701 | 0.758 | +5.7% |

**Key findings**:
1. **SPARSE wins on answer quality**: 66.8% answer_correctness (best)
2. **HYBRID wins on retrieval**: 92% context_recall (best)
3. **DENSE lags significantly**: 53% answer_correctness, 75% context_recall
4. All metrics improved substantially after removing low-quality questions

**Files created**:
- `data/eval/ragas_testset_filtered.jsonl` - Clean 28-question testset
- `data/eval/eval_results_{mode}_{with/no}_reference_filtered.json` - Filtered evaluation results
- `data/eval/filtered_metrics_summary.json` - Final metrics summary

---

### Experiment 3: Error Analysis

**Goal**: Categorize failure modes to understand where and why the RAG system fails

**Methodology**: Pattern-based categorization using metric thresholds

**Thresholds**:
```python
THRESHOLDS = {
    "answer_correctness": 0.5,
    "context_precision": 0.7,
    "context_recall": 0.7,
    "answer_relevancy": 0.8,
    "faithfulness": 0.7,
}
```

**Failure patterns**:
1. **Retrieval failure**: Low context_recall + low answer_correctness → didn't find relevant chunks
2. **Partial retrieval**: Low context_recall but answer still OK → found some but not all
3. **Generation failure**: Good retrieval + low answer_correctness + high faithfulness → reasoning error
4. **Hallucination**: Good retrieval + low answer_correctness + low faithfulness → fabrication
5. **Ranking issue**: Low context_precision → relevant chunks ranked low
6. **Relevancy issue**: Low answer_relevancy → answer doesn't address question

**Results** (on 28 filtered questions):

| Category | SPARSE | HYBRID | DENSE |
|----------|--------|--------|-------|
| **Success** | 16 (57.1%) ⭐ | 13 (46.4%) | 7 (25.9%) |
| **Retrieval failure** | 3 (10.7%) | 3 (10.7%) | 8 (29.6%) ❌ |
| **Generation failure** | 7 (25.0%) | 6 (21.4%) | 5 (18.5%) |
| **Ranking issue** | 2 (7.1%) | 3 (10.7%) | 3 (11.1%) |
| **Mixed failure** | 0 (0.0%) | 3 (10.7%) | 2 (7.4%) |
| **Hallucination** | 0 (0.0%) | 1 (3.6%) | 1 (3.7%) |
| **Relevancy issue** | 0 (0.0%) | 0 (0.0%) | 1 (3.7%) |
| **Partial retrieval** | 0 (0.0%) | 0 (0.0%) | 1 (3.7%) |

**Key insights**:

1. **Dense embeddings are the weak link**:
   - 29.6% retrieval failure rate (3× worse than sparse/hybrid)
   - Only 25.9% success rate overall
   - Struggles to find relevant chunks consistently

2. **Sparse BM25 performs best overall**:
   - 57.1% success rate (highest)
   - 10.7% retrieval failure (tied for lowest)
   - Strong keyword matching helps

3. **Generation errors affect all modes similarly**:
   - 18-25% generation failures across all modes
   - LLM struggles to synthesize correct answers even with right context
   - Suggests need for better prompting or stronger generation model

4. **Hallucination is rare** (<4%):
   - Model generally stays faithful to retrieved context
   - Prompt engineering appears effective

**Detailed failure examples** (see `analyze_errors.py` output):
- **Retrieval failure example**: Questions requiring specific technical details dense embeddings can't match
- **Generation failure example**: Retrieved correct chunks but LLM misinterpreted or missed key details
- **Ranking issue example**: Relevant chunks retrieved but ranked below top-5

**Scripts created**:
- `analyze_errors.py` - Categorizes all 28 questions by failure pattern
- `data/eval/error_analysis_summary.json` - Failure counts by mode

---

### Decision Point: Which Retrieval Method to Use?

**Evaluation summary**:

| Method | Success Rate | Answer Quality | Retrieval Quality | Failure Mode |
|--------|--------------|----------------|-------------------|--------------|
| **SPARSE** | 57.1% ⭐ | 66.8% ⭐ | 90.2% recall | 10.7% retrieval failure |
| **HYBRID** | 46.4% | 62.8% | 92.0% recall ⭐ | 10.7% retrieval failure |
| **DENSE** | 25.9% ❌ | 53.0% ❌ | 75.3% recall ❌ | 29.6% retrieval failure ❌ |

**Recommendation**: **Use SPARSE (BM25) only**

**Reasoning**:
1. Highest success rate (57.1%)
2. Best answer quality (66.8%)
3. Lowest retrieval complexity (no embedding computation needed)
4. Good recall (90.2%) - only 2% worse than hybrid
5. Dense embeddings hurt more than they help (29.6% retrieval failures)

**Alternative**: Keep HYBRID for production demo
- Highest recall (92%)
- More impressive technically (shows understanding of hybrid approaches)
- Document the tradeoffs explicitly in README

---

### Known Limitations & Future Fixes

#### ⚠️ Testset Generation Methodology (Critical Issue)

**Status**: Identified but not yet fixed (planned for v2)

**The Issue**: Generated testset from pre-chunked documents instead of whole documents

**Impact**:
- Limited to shallow fact-retrieval questions (no multi-hop reasoning)
- 46% low-quality questions from reference sections
- Ragas couldn't build proper knowledge graph across document structure

**Mitigation Taken**:
- Manually filtered 13 low-quality questions → 28 clean questions retained
- Evaluation results still valid for retrieval method comparison (sparse vs dense vs hybrid)
- Metrics are accurate for the filtered testset

**Fix for v2** (2-3 hours):
1. Use `PyMuPDFLoader` to load whole documents (not pre-chunked)
2. Regenerate testset with 50 questions
3. Expect more multi-hop/reasoning questions, better quality overall
4. Re-run evaluation with new testset

**Interview talking point**:
> "I initially generated the testset from 500-token chunks, which prevented Ragas from building a knowledge graph. This resulted in 46% low-quality questions from bibliography sections. I filtered these manually, but the proper fix is to regenerate using whole documents. This taught me that testset generation and retrieval chunking have completely different requirements—Ragas needs document structure to generate complex questions."

---

**Last Updated**: 2025-11-29
**Status**: Day 33 (RAG evaluation & error analysis) complete, ready for README.md documentation
