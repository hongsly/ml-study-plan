# Day 28: RAG Implementation - Data Loading & Embeddings

**Date**: 2025-11-24 (Week 5 Day 1)
**Project**: RAG Q&A System - Production Data Pipeline
**Time**: ~2 hours implementation + review

---

## Overview

Built complete data loading and embedding pipeline for RAG system:

```
RAG Q&A System Architecture
├── Input: 33 ArXiv papers (PDF, two-column layout)
├── Stage 1: PDF Parsing → Markdown (PyMuPDF4LLM)
├── Stage 2: Chunking → 500 tokens, 50 overlap (tiktoken)
├── Stage 3: Embedding → 384-dim vectors (sentence-transformers)
├── Stage 4: Indexing → FAISS IndexFlatIP (cosine similarity)
└── Output: 1541 searchable chunks, <10ms query time
```

**Key Achievement**: High-quality semantic search with production-ready code structure.

---

## Technical Decisions

### Decision 1: PyMuPDF4LLM vs PyPDF2/pdfplumber

**Problem**: ArXiv papers use two-column layout. Simple parsers (PyPDF2, pdfplumber) read text in PDF stream order, often resulting in:
```
"Column A line 1 Column B line 1 Column A line 2 Column B line 2..."
```
This destroys semantic meaning and ruins chunk quality.

**Solution**: PyMuPDF4LLM (pymupdf4llm)
- Detects layout automatically
- Reads column-by-column
- Outputs clean Markdown
- Built on PyMuPDF (fast C++ backend)

**Interview Answer**: *"For academic papers with multi-column layouts, layout-aware parsing is critical. I used PyMuPDF4LLM which detects column boundaries and reads text in reading order, preventing semantic splits across columns."*

**Code**:
```python
import pymupdf4llm

md_text = pymupdf4llm.to_markdown(str(pdf_path))  # Handles two-column automatically
```

---

### Decision 2: Token-based Chunking with tiktoken

**Why Tokens, Not Characters**:
- LLMs count tokens for context limits (e.g., GPT-4: 128k tokens)
- Character-based chunking is unreliable:
  - "Hello world" = 2 tokens
  - "Hello worldddddddd" = 3 tokens (subword tokenization)
- Must use same tokenizer as target LLM

**Parameters**:
- **Chunk size**: 500 tokens
  - Small enough for precise retrieval
  - Large enough to capture context
  - Fits comfortably in most LLM contexts
- **Overlap**: 50 tokens
  - Prevents semantic splits at boundaries
  - ~10% overlap balances redundancy vs coverage

**Trade-offs**:
| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (200-300) | Precise retrieval, less noise | Misses context, more chunks to rerank |
| Medium (500-700) | **Balanced** ✅ | - |
| Large (1000+) | Full context | Retrieves irrelevant content, less precise |

**Interview Answer**: *"I chose 500-token chunks with 50-token overlap. This balances retrieval precision—small enough to avoid irrelevant content—with sufficient context for the LLM to generate coherent answers. The overlap prevents important information from being split across chunk boundaries."*

**Code**:
```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # cl100k_base encoding
tokens = enc.encode(text)

chunks = []
for i in range(0, len(tokens), chunk_size - overlap):
    chunk_tokens = tokens[i : i + chunk_size]
    chunk_text = enc.decode(chunk_tokens)
    chunks.append(chunk_text)
```

---

### Decision 3: FAISS IndexFlatIP vs IndexFlatL2 ⚠️ CRITICAL

**The Bug Many People Make**:
```python
# ❌ WRONG - L2 distance ≠ cosine similarity
index = faiss.IndexFlatL2(dimension)

# ✅ CORRECT - Inner product = cosine when normalized
index = faiss.IndexFlatIP(dimension)
```

**Why This Matters**:
- **Semantic search needs cosine similarity**: Measures angle between vectors, not distance
- **L2 distance** (Euclidean): `||a - b||²` - depends on vector magnitude
- **Inner product**: `a · b` - when vectors are normalized (||a|| = ||b|| = 1), inner product = cosine similarity
- **Formula**: `cos(θ) = (a · b) / (||a|| × ||b||)` → `cos(θ) = a · b` when normalized

**Without Normalization**:
- Long documents → large embedding magnitude → higher scores (biased)
- Short documents → small embedding magnitude → lower scores (biased)
- Results ranked by length, not relevance

**With Normalization + IP**:
- All vectors have magnitude 1
- Rankings based purely on semantic similarity (angle)
- Fair comparison across all chunk lengths

**Interview Answer**: *"I used FAISS IndexFlatIP with normalized embeddings to compute cosine similarity. Many implementations incorrectly use IndexFlatL2, which measures Euclidean distance and is biased by vector magnitude. For semantic search, we care about the angle between vectors (cosine), not the distance. Normalizing embeddings and using inner product gives us true cosine similarity."*

**Code**:
```python
# VectorStore class
def embed(self, texts: list[str]) -> np.ndarray:
    return self.model.encode(texts, normalize_embeddings=True)  # ← CRITICAL

# Build index
self.index = faiss.IndexFlatIP(dimension)  # Inner product
self.index.add(embeddings)  # embeddings are normalized
```

---

### Decision 4: Embedding Model Selection

**Chosen**: `all-MiniLM-L6-v2` (sentence-transformers)

**Why**:
- **384-dim**: Smaller than alternatives (768-dim), faster search, lower memory
- **Fast**: 14K sentences/sec on CPU
- **Proven for RAG**: Widely used in production systems
- **Good quality**: 58.8 on STSB benchmark (sufficient for retrieval)

**Alternatives Considered**:
| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | **General RAG** ✅ |
| all-mpnet-base-v2 | 768 | Slower | Better | High-quality requirements |
| text-embedding-ada-002 | 1536 | API call | Best | Budget available |

**Interview Answer**: *"I used all-MiniLM-L6-v2 for its balance of speed, quality, and memory efficiency. At 384 dimensions, it's 2× smaller than alternatives like mpnet, enabling faster search and lower memory usage while maintaining sufficient retrieval quality for this use case."*

---

## Code Structure (OOP Design)

### Module Organization

```
src/
├── __init__.py                  # Package marker
├── utils.py                     # Chunking utilities
│   ├── Chunk (TypedDict)       # Type definition
│   └── chunk_text()            # Token-based chunking
├── data_loader.py               # PDF processing
│   ├── PDFDocument             # Single PDF wrapper
│   └── CorpusLoader            # Batch processing + JSONL export
├── vector_store.py              # FAISS operations
│   └── VectorStore             # Embedding + indexing + search
├── build_index.py               # Main pipeline script
└── test_search.py               # Search quality verification
```

**Design Principles**:
1. **Single Responsibility**: Each class has one clear purpose
2. **Separation of Concerns**: Data loading ≠ embedding ≠ indexing
3. **Testability**: Each module can be tested independently
4. **Reusability**: VectorStore can be used in other projects

**Interview Answer**: *"I organized the code into separate modules following SRP. PDFDocument handles single-file parsing, CorpusLoader orchestrates batch processing, VectorStore encapsulates all FAISS operations. This makes the pipeline easy to test, extend, and debug."*

---

## Common Pitfalls Avoided

### Pitfall 1: Wrong FAISS Index Type ❌
```python
# ❌ WRONG - Many tutorials use this
index = faiss.IndexFlatL2(dimension)

# ✅ CORRECT
index = faiss.IndexFlatIP(dimension)
```
**Impact**: L2 distance biases results by vector magnitude, not semantic similarity.

---

### Pitfall 2: Missing Embedding Normalization ❌
```python
# ❌ WRONG
embeddings = model.encode(texts)  # Default: not normalized

# ✅ CORRECT
embeddings = model.encode(texts, normalize_embeddings=True)
```
**Impact**: Without normalization, IndexFlatIP doesn't compute cosine similarity correctly.

---

### Pitfall 3: Character-based Chunking ❌
```python
# ❌ WRONG - Character count ≠ token count
chunks = [text[i:i+2000] for i in range(0, len(text), 1800)]  # 2000 chars, 200 overlap

# ✅ CORRECT - Use tiktoken for token-accurate chunking
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = enc.encode(text)
chunks = [tokens[i:i+500] for i in range(0, len(tokens), 450)]  # 500 tokens, 50 overlap
```
**Impact**: Character-based chunking leads to unpredictable token counts (can exceed LLM limits).

---

### Pitfall 4: Simple PDF Parsing for Academic Papers ❌
```python
# ❌ WRONG - Doesn't handle two-column layout
import PyPDF2
text = page.extract_text()  # Merges columns, destroys meaning

# ✅ CORRECT - Layout-aware parsing
import pymupdf4llm
text = pymupdf4llm.to_markdown(pdf_path)  # Reads column-by-column
```
**Impact**: Column merging creates gibberish text, ruins chunk quality.

---

### Pitfall 5: No Error Handling for Corrupted PDFs ❌
```python
# ❌ WRONG - Crashes entire pipeline if one PDF is corrupted
self.md_text = pymupdf4llm.to_markdown(str(self.path))

# ✅ CORRECT - Graceful degradation
try:
    self.md_text = pymupdf4llm.to_markdown(str(self.path))
except Exception as e:
    print(f"Error parsing {self.path}: {e}")
    self.md_text = ""  # Empty markdown on failure
```
**Impact**: One corrupted file crashes entire pipeline, loses progress.

---

## Results & Validation

### Pipeline Statistics
```
Input:  33 ArXiv papers (73.1 MB, ~600-800 pages)
Output: 1541 chunks (avg ~47 chunks/paper)
        384-dim embeddings
        FAISS index (~5 MB)
Time:   ~2-3 minutes (PDF parsing + embedding generation)
```

### Search Quality Verification

**Test Query**: "What is RAG?"

**Top-5 Results** (all from RAG survey paper `2312.10997_rag_survey_gao`):
1. **chunk_900**: RAG paradigms (Naive, Advanced, Modular) ⭐⭐⭐⭐⭐
2. **chunk_1350**: Core components (Retrieval, Generation, Augmentation) ⭐⭐⭐⭐⭐
3. **chunk_1800**: 3-step RAG process (Indexing → Retrieval → Generation) ⭐⭐⭐⭐⭐ **PERFECT**
4. **chunk_15300**: RAG ecosystem and multi-modal RAG ⭐⭐⭐⭐
5. **chunk_11700**: RAG evaluation and downstream tasks ⭐⭐⭐⭐

**Quality Indicators**:
- ✅ All results from most authoritative source (comprehensive survey)
- ✅ All chunks directly address the query concept
- ✅ Semantic matches, not just keyword matches
- ✅ Results sorted by relevance (cosine similarity working correctly)

**Performance**:
- Query time: <10ms (exact search on 1541 chunks)
- Index size: ~5 MB (reasonable for 1541 × 384-dim vectors)

---

## Interview Talking Points

### 1. Architecture Overview
*"I built a production-ready RAG data pipeline processing 33 ArXiv papers into 1541 semantically searchable chunks. The system uses layout-aware PDF parsing, token-accurate chunking, normalized embeddings, and FAISS inner product search for true cosine similarity."*

### 2. Technical Decision: Layout-Aware Parsing
*"For academic papers with two-column layouts, I used PyMuPDF4LLM instead of simple parsers like PyPDF2. This prevents column merging—where text from both columns gets interleaved—which would destroy semantic meaning and ruin chunk quality."*

### 3. Technical Decision: FAISS IndexFlatIP
*"A common mistake in RAG implementations is using IndexFlatL2, which measures Euclidean distance. For semantic search, we need cosine similarity. I used IndexFlatIP with normalized embeddings, where the inner product equals cosine similarity. This ensures results are ranked by semantic relevance, not vector magnitude."*

### 4. Search Quality Achievement
*"The system achieved high retrieval quality—all top-5 results for 'What is RAG?' came from the authoritative survey paper and directly addressed the query. This validates that the semantic search is working correctly, not just matching keywords."*

### 5. Bugs Avoided
*"I avoided several common pitfalls: using L2 distance instead of inner product, forgetting to normalize embeddings, character-based chunking instead of token-based, and simple PDF parsing that mangles multi-column layouts. Each of these would significantly degrade retrieval quality."*

### 6. Code Quality
*"I structured the code with clear separation of concerns: PDFDocument for single-file parsing, CorpusLoader for batch orchestration, VectorStore for all FAISS operations. This modularity makes the pipeline easy to test, debug, and extend with features like BM25 hybrid retrieval."*

---

## Key Formulas & Concepts

### Cosine Similarity
```
cos(θ) = (a · b) / (||a|| × ||b||)

When ||a|| = ||b|| = 1 (normalized):
cos(θ) = a · b  (inner product)
```

### Token Chunking Formula
```
chunk_indices = range(0, len(tokens), chunk_size - overlap)
chunk[i] = tokens[i : i + chunk_size]

Example (500 tokens, 50 overlap):
- Chunk 0: tokens[0:500]
- Chunk 1: tokens[450:950]  (50 token overlap with Chunk 0)
- Chunk 2: tokens[900:1400] (50 token overlap with Chunk 1)
```

### Memory Calculation
```
FAISS IndexFlatIP memory:
N chunks × D dimensions × 4 bytes (fp32)
= 1541 × 384 × 4 = ~2.4 MB

With metadata (chunk IDs, JSONL):
Total ≈ 5-10 MB
```

---

## Next Steps (Future Enhancements)

**Day 29 (Week 5 Day 2)**: Add BM25 Sparse Retrieval
- Implement BM25 for keyword matching
- Add RRF (Reciprocal Rank Fusion) to combine dense + sparse
- Compare retrieval quality: dense-only vs hybrid

**Day 30 (Week 5 Day 3)**: Add Reranking
- Cross-encoder for reranking top-K results
- Compare: retrieval → generation vs retrieval → rerank → generation

**Future**:
- Add page number metadata for citations
- Implement Ragas evaluation framework
- Handle edge cases (code blocks, tables) in chunking
- Add caching for faster repeated queries

---

## References

**Libraries Used**:
- [PyMuPDF4LLM](https://pypi.org/project/pymupdf4llm/) - Layout-aware PDF parsing
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's tokenizer (BPE)
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

**Papers Referenced**:
- RAG (Lewis et al., 2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Sentence-BERT (Reimers & Gurevych, 2019): Sentence Embeddings using Siamese BERT-Networks

**Key Concepts**:
- Cosine similarity for semantic search
- Token-based chunking for LLM compatibility
- Layout-aware parsing for multi-column documents
- Inner product vs L2 distance in vector search

---

**Created**: 2025-11-24
**Project**: RAG Q&A System (Week 5 Portfolio Project)
**Status**: ✅ Data loading pipeline complete, high-quality retrieval achieved
