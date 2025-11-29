# Ragas Testset Generation - Technical Reference

**Purpose**: Synthetic test generation for RAG evaluation
**Key Decision**: Ollama (free) vs OpenAI (quality) for test generation
**Files**: `scripts/generate_testset.py`

---

## Cost Analysis (Corrected)

| Approach | Chunks | Questions | Cost | Time | Use When |
|----------|--------|-----------|------|------|----------|
| **gpt-4o-mini (full)** | 1500 | 40 | $1.00 | 30-45 min | Comprehensive coverage |
| **gpt-4o-mini (sampled)** | 250 | 20 | $0.13 | 5-10 min | Fast iteration |
| **Ollama (local)** | Any | Any | $0.00 | 10-15 min | Free, quality TBD |

**Key insight**: gpt-4o vs gpt-4o-mini is 16-17× cost difference. Always verify which model you're using.

**Ragas multi-phase processing**: SummaryExtractor → NER → Theme extraction → Query generation (2-3× single-phase cost)

---

## Sampling Strategy

**Purpose**: 6× faster iteration (5-10 min vs 30-45 min), not just cost savings ($0.87)

**Implementation**: `sample_chunks_by_paper()` in generate_testset.py
- 250 chunks → 7-8 per paper (32 papers)
- First + last + random middle chunks
- Representative coverage across all papers

**Decision factor**: Speed during development > small cost savings

---

## Ollama Support

**Status**: ✅ Works (despite GitHub issues claiming otherwise)

**Setup**: qwen2.5-coder:7b + nomic-embed-text embeddings
**Result**: Successfully generated 6 questions from 5 chunks
**Format**: Matches Ragas spec (reference_contexts, reference, persona, etc.)

**Trade-off**: $0 cost vs unknown quality (requires manual review)

---

## Test Format Comparison

### Manual Format
- Question text + vague source references ("Lewis et al., 2020")
- No ground truth answer
- Limited metadata

### Ragas Format
- Question + exact ground truth contexts (full chunk text)
- Ground truth answer
- Rich metadata (persona, style, length, synthesizer)

**Advantage**: Enables objective Context Recall measurement (impossible without ground truth contexts)

---

## Metrics & Ground Truth Requirements

| Metric | Ground Truth Needed? | Why |
|--------|---------------------|-----|
| Context Recall | ✅ Required | Can't measure without knowing what should be retrieved |
| Context Precision | 🟡 Better with | LLM-as-judge works, but less objective |
| Faithfulness | ❌ No | LLM judges answer vs context |
| Answer Relevance | ❌ No | LLM judges answer vs question |

---

## Project Design Decisions

### Chunking: 500 tokens, 50 overlap
- Standard for academic RAG (512-1024 range)
- Balances context vs precision
- Interview answer: "Would A/B test in production (256, 512, 1024)"

### No Reranking (for now)
- Hybrid retrieval (BM25 + DPR + RRF) = 80% Precision@5
- Cross-encoder would add 50ms latency for 85-90% precision
- Time-boxed project scope decision
- Listed in future improvements

### Why Not RAPTOR?
- Hierarchical chunking requires +2-3 hours implementation
- LLM calls for recursive summarization ($$)
- Post-project experiment, not main implementation

---

## Key Lessons

1. **Model verification**: Test with exact model you'll deploy (gpt-4o vs gpt-4o-mini = 16× cost)
2. **Tool skepticism**: Ollama works despite GitHub issues - always test yourself
3. **Multi-phase costs**: Ragas isn't simple generation, it builds knowledge graphs (2-3× multiplier)
4. **Optimize for iteration speed**: Sampling saves 6× time, $0.87 is secondary benefit

---

## Interview Talking Points

**Cost-conscious engineering**:
> "I discovered my test used gpt-4o instead of gpt-4o-mini, causing 16× cost overestimation. After correction, full generation costs $1.00, but I still use sampling (250 chunks) primarily for 6× faster iteration during development. I also validated Ollama works for free generation, providing a cost-effective alternative."

**Evaluation design**:
> "I chose Ragas format because it provides exact ground truth contexts and answers, enabling objective Context Recall measurement—impossible with manual tests that only reference paper names. This trades upfront generation cost for rigorous evaluation metrics."
