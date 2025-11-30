# Day 33 Quick Reference: RAG Evaluation & Error Analysis

**Date**: 2025-11-29 (Week 5 Day 5)
**Topics**: Ragas metrics, question quality analysis, error categorization, testset methodology

---

## Ragas Evaluation Metrics (5 metrics)

### With Reference Answer (3 metrics)

1. **answer_correctness** (0-1)
   - Measures: Does response match reference answer?
   - LLM-based semantic similarity
   - Used for: Overall answer quality

2. **context_precision** (0-1)
   - Measures: Are retrieved chunks relevant to question?
   - 1.0 = all retrieved chunks relevant
   - Used for: Retrieval quality (precision)

3. **context_recall** (0-1)
   - Measures: Does retrieved context cover the answer?
   - LLM judges if context can answer the question
   - 1.0 = context completely covers question
   - **Key advantage**: Avoids incomplete ground truth problem

### Without Reference Answer (2 metrics)

4. **answer_relevancy** (0-1)
   - Measures: Does response address the question?
   - Can be relevant but wrong (that's why we need correctness)
   - Used for: Check if answer is on-topic

5. **faithfulness** (0-1)
   - Measures: Is response grounded in retrieved context?
   - Low faithfulness = hallucination
   - Used for: Detect fabricated claims

---

## Key Distinctions

### context_recall vs context_precision

| Metric | Question | Perfect Score Means |
|--------|----------|---------------------|
| **context_recall** | "Did we retrieve enough?" | Retrieved context can fully answer the question |
| **context_precision** | "Did we retrieve too much noise?" | All retrieved chunks are relevant |

**Why use context_recall?**
- Traditional recall needs complete ground truth (all relevant chunks in corpus)
- LLM-based context_recall only needs to judge: "Can this context answer the question?"
- Avoids expensive exhaustive relevance judgments

### answer_correctness vs answer_relevancy

| Metric | Question | Example |
|--------|----------|---------|
| **answer_correctness** | "Is the answer correct?" | Q: "Capital of France?" A: "Paris" ✅ |
| **answer_relevancy** | "Does answer address the question?" | Q: "Capital of France?" A: "France uses Euro currency" (relevant but wrong) |

**Both needed**: Can be relevant but incorrect, or correct but off-topic

---

## Question Quality Analysis

### Problem Discovered

**46% of Ragas questions were low quality** - generated from bibliography/reference sections

**Examples of bad questions**:
- "What is Yoav Goldberg's contribution to NLP?" (from bibliography)
- "What do the numbers in Table 1 represent?" (from table headers)
- "What year was X published?" (from citation metadata)

### Detection Heuristics

```python
citation_patterns = [
    r'\d{4}[a-z]?\.',  # Year: "2019."
    r'et al\.',         # Authors: "et al."
    r'In _[^_]+_\.',    # Venue: "In _EMNLP_."
    r'_[A-Z][^_]+_,',   # Journal: "_ArXiv_,"
]

# Heuristic: >3 citations per 100 words → likely reference section
```

### Filtering Decision

**19 suspicious found → Manual review → 13 removed**

| Category | Count | Action |
|----------|-------|--------|
| Definitely bad (citations, headers) | 11 | ❌ Remove |
| Moderate quality (contaminated) | 4 | ❌ Remove |
| Good but contaminated context | 4 | ✅ Keep |
| False positive | 1 | ✅ Keep |

**Result**: 28 clean questions (68% retention from 41)

---

## Root Cause: Testset Generation Methodology

### What We Did Wrong ❌

```python
# WRONG: Pre-chunked documents
loader = CorpusLoader()
loader.parse_pdfs(pdf_paths, chunk_size=500)  # Chunks!
chunks = loader.get_chunks()  # 1,541 isolated chunks
generator.generate_with_langchain_docs(chunks, testset_size=50)
```

**Problem**: Ragas treats each 500-token chunk as an isolated document

### Why This Failed

1. **Ragas builds a knowledge graph** from input documents
2. **With chunks**: Each chunk = isolated island, no relationships
3. **Cannot distinguish**: Content vs references (bibliography chunk looks valid)
4. **Cannot generate**: Multi-hop questions across sections
5. **Result**: Shallow fact-retrieval questions only

### Correct Approach ✅

```python
# CORRECT: Whole documents with structure
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("paper.pdf")
pages = loader.load()  # Full pages with structure
# Merge pages in same pdf and filter out reference section
generator.generate_with_langchain_docs(documents, testset_size=50)
```

**With whole documents, Ragas can**:
- Understand document hierarchy (sections, headers)
- Link concepts across sections
- Generate multi-hop questions ("How does Section 3 address Section 2 limitations?")
- Distinguish content from references

**Lesson**: Testset generation ≠ Retrieval chunking
- Retrieval needs small chunks (500 tokens) for precision
- Testset generation needs whole documents for knowledge graph

---

## Error Analysis - Failure Patterns

### Thresholds Used

```python
THRESHOLDS = {
    "answer_correctness": 0.5,
    "context_precision": 0.7,
    "context_recall": 0.7,
    "answer_relevancy": 0.8,
    "faithfulness": 0.7,
}
```

### 5 Main Failure Patterns

1. **Retrieval Failure** (most critical)
   - Low context_recall + Low answer_correctness
   - Didn't find relevant chunks → wrong answer
   - **DENSE: 29.6%** vs **SPARSE: 10.7%** (3× worse!)

2. **Generation Failure** (second most common)
   - High context_recall + Low answer_correctness + High faithfulness
   - Found right chunks but LLM misinterpreted
   - ~20-25% across all modes

3. **Hallucination** (rare, <4%)
   - High context_recall + Low answer_correctness + **Low faithfulness**
   - Found right chunks but fabricated claims

4. **Ranking Issue** (~7-11%)
   - Low context_precision
   - Relevant chunks retrieved but ranked below top-5

5. **Relevancy Issue** (rare, <4%)
   - Low answer_relevancy
   - Answer doesn't address the question

### Edge Case: Parametric Knowledge "Hallucination"

**Scenario**: Low context_recall + High answer_correctness + Low faithfulness

**What's happening**: Model uses its own memory to answer correctly, ignoring poor retrieved context

**Why it's problematic**: Answer not grounded in context (RAG systems need grounded answers)

**Our categorization**: Currently lands in "partial_retrieval" but should be "retrieval_failure_with_hallucination"

---

## Final Results Summary

### Filtered Metrics (28 clean questions)

| Method | Answer Correctness | Context Recall | Success Rate |
|--------|-------------------|----------------|--------------|
| **SPARSE** | **66.8%** ⭐ | 90.2% | **57.1%** ⭐ |
| **HYBRID** | 62.8% | **92.0%** ⭐ | 46.4% |
| **DENSE** | 53.0% ❌ | 75.3% ❌ | 25.9% ❌ |

### Why SPARSE > DENSE?

**Evidence (not just hypothesis)**:
1. **3× worse retrieval failures**: Dense 30% vs Sparse 11%
2. **Keyword matching advantage**: Technical terms (ColBERT, DPR, BM25)
3. **Small corpus effect**: 1,541 chunks doesn't benefit from dense embeddings
4. **Relative comparison valid**: Question quality affects all methods equally

**Not just shallow questions**: Retrieval failure gap is real structural difference

---

## Metrics Improvement After Filtering

**~13% average improvement** across all metrics after removing 13 low-quality questions

| Mode | Metric | Original (41q) | Filtered (28q) | Δ |
|------|--------|---------------|----------------|---|
| SPARSE | answer_correctness | 0.567 | 0.668 | +10.1% |
| SPARSE | context_recall | 0.817 | 0.902 | +8.5% |
| HYBRID | answer_correctness | 0.531 | 0.628 | +9.7% |
| HYBRID | context_recall | 0.831 | 0.920 | +8.9% |
| DENSE | answer_correctness | 0.449 | 0.530 | +8.1% |
| DENSE | context_recall | 0.670 | 0.753 | +8.3% |

---

## Interview Talking Points

### On Testset Generation Mistake

> "I initially generated the testset from 500-token chunks, which prevented Ragas from building a proper knowledge graph. This resulted in 46% low-quality questions from bibliography sections. I manually filtered these, but the proper fix is to regenerate using whole documents. This taught me that testset generation and retrieval chunking have completely different requirements—Ragas needs document structure to generate complex questions."

### On SPARSE vs DENSE Finding

> "Sparse outperformed because: (1) Dense had 30% retrieval failures vs 11% for sparse—couldn't find relevant chunks. (2) Technical corpus benefits from exact keyword matching (BM25 excels at 'ColBERT', 'tensor parallelism'). (3) Small corpus (1,500 chunks) doesn't justify dense embeddings' overhead. While better questions might change absolute scores, the 3× retrieval failure gap is a real structural difference."

### On Failure Pattern Edge Case

> "We found an interesting edge case: sometimes the model answered correctly despite poor retrieval by using its parametric knowledge. While answer_correctness was high, faithfulness was low—the answer wasn't grounded in retrieved context. This shows why RAG evaluation needs both correctness AND faithfulness metrics. In production, I'd prefer 'I don't know' to an ungrounded answer, even if correct."
