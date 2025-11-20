# Day 23 Quick Reference: Advanced RAG Day 2 (Week 4 Day 2)

**Date**: 2025-11-19  
**Topics**: FiD, GraphRAG, RAFT, Agentic RAG, Multi-hop retrieval, Parent document retrieval, Complex PDF parsing  
**Knowledge Check Score**: 96.3% (A+)

---

## 1. FiD (Fusion-in-Decoder) ⭐ Gap Q177 Closed: 0% → 100%

**Architecture**:
```
For query q:
1. Encode each doc independently: encode([q; doc_i]) for i=1..n
2. Decoder cross-attends to all n encoder outputs (concatenated in attention space)
3. Generate answer autoregressively
```

**Cost Analysis** (n docs, L tokens/doc):

**Encoding (Prefilling)**:
- Per doc: O(L²) self-attention
- Total: O(n·L²) - linear in n ✅
- Can parallelize across docs

**Decoding**:
- Per output token: O(nL) cross-attention
- **Same as long context** (not a differentiator)

**Key Insight**: Difference is in prefilling, not decoding!

---

### FiD vs Long Context (2025 Trade-offs)

| Phase | FiD | Long Context (no cache) | Long Context (prefix cache) |
|-------|-----|------------------------|----------------------------|
| **Prefilling** | O(n·L²) | O((nL)²) = O(n²·L²) | First: O(n²·L²), Then: O(L²_query) |
| **Decoding** | O(nL)/token | O(nL)/token | O(nL)/token |
| **Scaling** | Linear in n | Quadratic in n | Amortized constant |

**Example** (100 docs × 250 tokens):
```
FiD Prefilling:
  100 parallel × 250² = 6.25M ops

Long Context (no cache):
  (100×250)² = 625M ops (100× worse!)

Long Context (cached docs):
  Query only: 250² ≈ 62K ops (fastest!)
```

---

### When to Choose FiD vs Long Context

#### Choose **FiD** when:
✅ **Single-shot queries** (no prefix cache benefit)
✅ **Many documents** (n > 50-100) - O(n) vs O(n²) matters
✅ **Lost-in-the-middle mitigation** - independent encoding removes position bias
✅ **High QPS with diverse docs** (can't cache effectively)

#### Choose **Long Context** when:
✅ **Multi-turn conversations** (prefix cache wins - amortized near-constant)
✅ **Same docs, multiple queries** (cache docs, only encode query)
✅ **Simpler implementation** (no custom training needed)
✅ **Maximum accuracy** (full cross-document attention)

**Default in 2025**: Long context with prefix caching (simpler, fast enough for most cases)

---

## 2. GraphRAG (Microsoft 2024)

**Two Search Modes** (Critical Distinction!):

### Mode 1: Local Search (Entity-Centric Multi-Hop)
```
Query → Extract entities → Graph traversal → Related nodes
Example: "Who is Alice's manager?" → Traverse "manager" edge
```
Traditional KG RAG does this.

### Mode 2: Global Search (Theme/Summary-Centric) ⭐ Innovation
```
Query → Retrieve community summaries → Map-reduce → Global answer
Example: "What are the main themes in this corpus?"
```

**Key Innovation**: Hierarchical community detection + pre-computed summaries
- Leaf communities → Mid-level → Top-level
- Enables "big picture" questions

---

### GraphRAG Architecture

**Graph Building (Offline)**:
1. Entity extraction: NER/LLM-based on documents
2. Relation extraction: Find relationships between entities
3. Build knowledge graph: Nodes (entities), Edges (relations)
4. **Community detection**: Hierarchical clustering (Leiden algorithm)
5. **Community summaries**: Bottom-up (leaf → mid → top)

**Query Time**:
- **Local**: Entity traversal (fast, specific)
- **Global**: Map-reduce over community summaries (holistic)

---

### GraphRAG Use Cases

#### When GraphRAG Wins ✅

**Local Search**:
- Multi-hop entity queries: "CEO of company that acquired Instagram"
- Relationship-based: "All of Alice's projects in 2023"

**Global Search** (Key Innovation!):
- ✅ Holistic summaries: "What are the main themes?"
- ✅ Pattern discovery: "What patterns emerge in customer feedback?"
- ✅ Corpus-level insights: "What topics are covered?"

#### When GraphRAG Loses ❌

- Simple fact lookup: "What is the capital of France?" (overkill)
- Real-time/dynamic data: Graph is pre-computed
- High upfront cost: 10-100× more expensive than traditional RAG

---

## 3. RAFT (Retrieval-Augmented Fine-Tuning)

**Core Idea**: Resolve "RAG vs Fine-tune" debate - **Do both**

**Training Format**:

**P% examples** (typically 80-90%): Question + Golden doc + Distractors + Answer with CoT  
**(1-P)% examples** (10-20%): Question + Only distractors + Answer with CoT

**Why (1-P) is necessary**:
- Prevents over-reliance on retrieval
- Nudges model to internalize domain knowledge
- Model learns: "If docs don't help, use internal knowledge"

**Example Training Instance** (P% case):
```
Question: "What is the capital of France?"

Context:
Doc 1 (distractor): "France has a population of 67 million..."
Doc 2 (golden): "France is a country in Europe. Its capital is Paris..."
Doc 3 (distractor): "Paris is known for the Eiffel Tower..."

Answer (with CoT):
"<reasoning>
Doc 1 discusses population, no capital mentioned.
Doc 3 mentions Paris but not as capital.
Doc 2 states: 'Its capital is Paris.' ← This is the answer.
</reasoning>
<answer>The capital of France is Paris.</answer>"
```

**What Are Distractor Docs**:
- Retrieved documents SIMILAR to question but DON'T contain answer
- Usually: Top BM25/semantic results that aren't the golden doc
- Purpose: Teach model to distinguish relevant vs irrelevant

**Performance**: RAFT > Domain fine-tuned (no RAG) > Base model + RAG

**When to Use**:
✅ Domain-specific RAG with training data (legal, medical, corporate)
✅ High-stakes retrieval with many similar documents
✅ When distractors are common (product versions, case law)

---

## 4. Agentic RAG

**Core Idea**: LLM as autonomous agent that decides when/how to retrieve

**ReAct Pattern**: Thought → Action → Observation → (repeat)

**Example**:
```
Q: "Average ML engineer salary in Seattle 2024?"

Thought 1: "Need current salary data"
Action 1: retrieve("ML engineer salary Seattle 2024")
Observation 1: [Ranges: $140k-$180k]

Thought 2: "Need to calculate average"
Action 2: calculate(mean([140, 150, 160, 170, 180]))
Observation 2: [160]

Thought 3: "Have answer"
Action 3: finish("Approximately $160k")
```

**Key Capabilities**:
✅ Decide when to retrieve (skip if model knows answer)
✅ Multi-hop reasoning (retrieve → extract → retrieve again)
✅ Multi-tool (retrieval + calculator + code + web search)
✅ Self-correction ("First query didn't work, refine")

**Cost**: 3-5× more expensive than traditional RAG (multiple LLM calls)

---

### Traditional RAG vs Agentic RAG

| Aspect | Traditional RAG | Agentic RAG |
|--------|----------------|-------------|
| **Pipeline** | Fixed: Always retrieve → generate | Dynamic: Agent decides |
| **Retrieval** | One-shot | Multi-step if needed |
| **Tools** | Only retrieval | Multi-tool (retrieve, calculate, code, search) |
| **Autonomy** | None | Agent decides if retrieval needed |
| **Cost** | 1-2 LLM calls | 3-10+ LLM calls (3-5× more expensive) |
| **Latency** | Fast | Slower (multiple reasoning steps) |

**When to use Agentic RAG**:
✅ Complex multi-step queries
✅ Uncertain retrieval needs
✅ Multi-tool requirements

**When it's overkill**:
❌ Simple QA (just generate or always retrieve)
❌ Latency-critical applications
❌ Cost-sensitive

---

## 5. Multi-Hop Retrieval

**Concept**: Multi-hop = Chain multiple retrievals to answer compositional questions
(This is the CAPABILITY, not the implementation)

**Three Implementations**:

### 1. Query Decomposition
- Break into sub-questions, retrieve each
- **Pro**: Fast, cheap (2-3 LLM calls), predictable, interpretable
- **Con**: No self-correction, inflexible (may fail if hops unclear)
- **Use**: Clear decomposition patterns (e.g., "Compare X to Y")

### 2. Agentic RAG (ReAct)
- Agent decides when/how to retrieve
- **Pro**: Flexible, self-correcting, multi-tool, adaptive
- **Con**: Expensive (5-10+ calls), slow, unpredictable, high latency
- **Use**: Complex queries, need self-correction

### 3. GraphRAG
- Graph traversal along relationship edges
- **Pro**: Fast (graph lookup), explicit relationships, interpretable
- **Con**: High upfront cost, maintenance burden, entity-only
- **Use**: Entity-rich domains, repeated queries

### Decision: Cost vs Flexibility
```
Cheapest/Fastest → Most Expensive/Flexible:
GraphRAG < Query Decomp < Iterative < Agentic RAG
```

---

## 6. Parent Document Retrieval

**Core Idea**: Separate retrieval (precision) from context augmentation (completeness)

**Architecture: Two Storage Systems**:

### 1. Vector Store (Retrieval Layer)
- Store: **Small chunks** (128-256 tokens)
- Purpose: Precise semantic matching
- Contains: chunk_id, embedding, text, **parent_key** (link to parent)

### 2. Parent Doc Store (Context Layer)
- Store: **Large chunks** or full documents (512-2048 tokens)
- Purpose: Complete context for generation
- Lookup: By parent_key (hash map / key-value store)

**Retrieval Flow**:
```
Query → Embed → Vector search (small chunks) → Top-k chunks
  ↓
Extract parent_keys from chunks
  ↓
Lookup parents in parent store (deduplicate!)
  ↓
Return parent documents to LLM (full context)
```

**Problem It Solves**:
- **Small chunks**: High precision, missing context
- **Large chunks**: Complete context, low precision
- **Parent Doc**: ✅ High precision retrieval + Complete context generation

**Critical Implementation Detail**: **Deduplication**!
```
If multiple child chunks from same parent retrieved:
  Top-10 chunks:
    chunk_1, chunk_2, chunk_4 → parent_A (same parent!)
    chunk_3 → parent_B

Must deduplicate: Return parent_A once (not 3 times)
Ranking: Use best child score or sum/max of all child scores
```

**When to Use**:
✅ Documents with natural hierarchy (sections, chapters)
✅ Queries need surrounding context
✅ Can maintain two storage systems

---

## 7. Complex PDF Parsing / OCR

**Goal**: Unstructured (PDF, images) → Structured, RAG-friendly (markdown, JSON, or images)

**Three Paradigms**:

### Paradigm 1: Parse to Text/Markdown (Traditional RAG)

**Pipeline**: PDF → Parser → Markdown/JSON → Chunk → Text embeddings → Vector DB

**Tools**:
- **unstructured.io**: Production-grade, table detection, HTML/markdown output
- **LlamaParse**: Vision-based parsing, clean markdown for RAG
- **Docling** (IBM, 2024): Semantic structure preservation, scientific papers

**Pros**:
✅ Works with text-only LLMs (cheaper: GPT-3.5 vs GPT-4V)
✅ Semantic search on table contents
✅ Smaller storage (text << images)
✅ Faster retrieval

**Cons**:
❌ Loses visual layout (charts, colors, spatial relationships)
❌ Parser errors (garbage in, garbage out)
❌ Complex layouts degrade (nested tables, multi-column)

### Paradigm 2: Keep as Images (Multi-Modal RAG)

**Pipeline**: PDF → Page images → Multi-modal embeddings (CLIP) → Vector DB

**At retrieval**: Return images to multi-modal LLM (GPT-4V, Claude 3, Gemini)

**Pros**:
✅ Preserves all visual information (diagrams, formatting, charts)
✅ No parsing errors (what you see is what you get)
✅ Handles complex layouts (infographics, nested tables)

**Cons**:
❌ Requires expensive multi-modal LLMs (3-5× more)
❌ Can't semantically search table contents (just pixels)
❌ Larger storage (images >> text)
❌ Slower (vision models heavier)

### Paradigm 3: Hybrid (Parse + Keep Images) ⭐ Often Best

**Strategy**:
1. Parse to markdown/JSON (for semantic search)
2. Store original page images (for visual reference)
3. At retrieval: Return both text + image

**Example**:
```
Query: "iPhone Q2 revenue?"
- Semantic search on parsed text: "Q2: $55B"
- Return: Parsed text + original PDF page image
- LLM cites text + describes visual context
```

**When to use**: Financial reports, research papers with charts/tables

---

### The Table Problem (Critical!)

**Naive parsing**:
```
PDF table → "Product Revenue iPhone $50B" (structure lost!)
```

**Good parsing** (unstructured.io, LlamaParse):
```markdown
| Product | Revenue |
|---------|---------|
| iPhone  | $50B    |
```
→ Preserves structure, enables semantic search

**Multi-modal**:
- LLM "sees" table visually (preserves layout)
- But: Can't search table contents in vector DB

---

### Relation to Multi-Modal RAG

**Key Question**: Parse to text OR keep as images?

**Answer**: Depends on use case!

| Content Type | Recommendation |
|--------------|----------------|
| Text-heavy PDFs | Parse to markdown |
| Tables for search | Parse to markdown + keep images (hybrid) |
| Charts/diagrams | Multi-modal (keep images) |
| Infographics | Multi-modal |
| Financial reports | **Hybrid** (search text, verify visually) |

**Trade-off**: Text parsing enables semantic search but loses visual context. Multi-modal preserves visuals but costs more and can't search table contents.

---

### OCR (Optical Character Recognition)

**When needed**: Scanned PDFs (text not selectable)

**Tools**:
- **Tesseract**: Open source, basic OCR
- **AWS Textract**: Production-grade, extracts tables/forms as structured JSON
- **Azure Document Intelligence**: Similar to Textract

**Native PDF** (text selectable): Use pypdf, pdfplumber (no OCR needed)

---

### Interview Question: "How to handle tables in RAG?"

**Answer**:
"Tables in PDFs are challenging. I'd use a **hybrid approach**:

1. **Parse with tools like unstructured.io** to convert tables to markdown/HTML. This preserves structure and enables semantic search on table contents. For 'What was iPhone revenue in Q2?', I can search the parsed text.

2. **Store original page images** alongside parsed text. When retrieved, return both to the LLM.

3. **Multi-modal LLM** (GPT-4V) can read the text for extraction and verify against the visual table for accuracy.

For simple text tables, parsing alone is sufficient. For complex nested tables or charts, the hybrid approach ensures we don't lose visual context while maintaining search capability."

---

## Key Formulas & Comparisons

**FiD Scaling**:
```
FiD:          O(n · L²)    Linear in n
Long Context: O(n² · L²)   Quadratic in n

Example: n=100, L=250
FiD:        6.25M ops
Long Ctx:   625M ops (100× worse without cache)
```

**Multi-Hop Implementation Costs**:
```
Query Decomposition:  2-3 LLM calls
Agentic RAG:          5-10+ LLM calls
GraphRAG:             Graph traversal (cheapest at query time)
```

**Parent Doc Chunk Sizes**:
```
Child chunks (retrieval):  128-256 tokens (precision)
Parent chunks (context):   512-2048 tokens (completeness)
```

---

## Common Mistakes to Avoid

1. **FiD vs Long Context**: Don't forget prefix caching impact in 2025!
2. **GraphRAG scope**: It CAN handle themes (global search), not just entities
3. **RAFT (1-P) examples**: They're necessary (prevent over-reliance on retrieval)
4. **Agentic RAG cost**: 3-5× more expensive due to multiple reasoning steps
5. **Multi-hop**: It's a capability, not single implementation
6. **Parent doc deduplication**: Must dedupe if multiple children from same parent
7. **PDF parsing**: Hybrid approach often best (parse + keep images)

---

## Interview-Ready Talking Points

**When asked about FiD**:
- "In 2025, consider prefix caching - long context becomes competitive for multi-turn"
- "FiD wins for single-shot with many docs (O(n) vs O(n²))"
- "FiD helps with lost-in-the-middle (independent encoding)"

**When asked about GraphRAG**:
- "Two modes: Local (entity) and Global (themes via community summaries)"
- "Global search is the innovation - pre-computed hierarchical summaries"
- "High upfront cost but good for entity-rich domains with repeated queries"

**When asked about RAFT**:
- "Solves 'RAG vs fine-tune' - do both"
- "P=80-90% with golden doc, 10-20% without (internalize knowledge)"
- "Teaches model to ignore distractors explicitly"

**When asked about Agentic RAG**:
- "ReAct loop: Thought → Action → Observation"
- "3-5× more expensive but flexible and self-correcting"
- "Use for complex queries, overkill for simple QA"

**When asked about multi-hop**:
- "Capability with multiple implementations"
- "Query decomposition (cheap, inflexible), Agentic (expensive, flexible), GraphRAG (fast, high setup)"

**When asked about PDFs**:
- "Parse to text (searchable), keep images (visual context), or hybrid (both)"
- "Financial reports → hybrid (search tables + verify visually)"

---

## Next Topics (Day 3 - Optional)

If continuing to Day 3:
- Retrieval metrics (Recall@K, Precision@K, MRR, NDCG)
- End-to-end RAG metrics (Faithfulness, Answer Relevance, Context Quality)
- Context window optimization
- Review remaining "unsure" topics

**Current Progress**:
- **Baseline**: 21.3% weighted (23.2% high-priority)
- **After Day 1**: ~40-45% weighted
- **After Day 2**: ~52-55% weighted, ~68-72% high-priority ✅
- **Target after Day 3**: 82% high-priority, 55-60% overall

---

**Week 4 Progress**: Day 2 complete - 7 new topics + 3 consolidations mastered, 96.3% knowledge check (A+)
