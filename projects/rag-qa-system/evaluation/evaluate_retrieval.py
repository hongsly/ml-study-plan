import json

import numpy as np
from datasets import Dataset
from ragas import RunConfig, SingleTurnSample, evaluate
from ragas.metrics import (IDBasedContextPrecision, IDBasedContextRecall,
                           MetricType, SingleTurnMetric)
from regex import P
from src.hybrid_search import HybridRetriever
from src.utils import (CHUNKS_JSONL_PATH, EVAL_DATA_DIR, FAISS_INDEX_PATH,
                       Chunk, load_chunks_from_jsonl)
from tqdm import tqdm

RAGAS_TESTSET_PATH = EVAL_DATA_DIR / "ragas_testset.jsonl"


def map_reference_contexts_to_ids(
    reference_contexts: list[str], all_chunks: list[Chunk]
) -> list[str]:
    chunk_ids = []
    for reference_context in reference_contexts:
        context = reference_context[15:]  # remove prefix like "<1-hop>"
        containing_chunks = [c for c in all_chunks if context in c["chunk_text"]]
        if len(containing_chunks) == 0:
            raise ValueError(f"Reference context not found in any chunk: {context}")
        elif len(containing_chunks) > 1:
            docs = {c["metadata"]["title"] for c in containing_chunks}
            if len(docs) > 1 or len(containing_chunks) > 2:
                print(f"Reference context found in multiple chunks: {context}")
                print(
                    f"Containing chunks: {[c['chunk_id'] for c in containing_chunks]}"
                )
                print("-" * 100)
                raise ValueError(
                    f"Reference context found in multiple chunks: {context}"
                )
        chunk_ids.append(containing_chunks[0]["chunk_id"])
    return chunk_ids


# def temp():
#     # for item in testset:
#     #     # print(f"Processing item: {item['user_input']}")
#     #     chunk_ids = map_reference_contexts_to_ids(
#     #         item["reference_contexts"], all_chunks
#     #     )
#     #     # print(f"Chunk IDs: {chunk_ids}")
#     #     ground_truth_context_ids.append(chunk_ids)


class IDBasedMRR(SingleTurnMetric):
    name: str = "id_mrr"
    _required_columns = {
        MetricType.SINGLE_TURN: {"retrieved_context_ids", "reference_context_ids"}
    }

    def init(self, run_config: RunConfig) -> None: ...

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        retrieved = sample.retrieved_context_ids
        reference = set(sample.reference_context_ids)

        for i, doc_id in enumerate(retrieved):
            if doc_id in reference:
                return 1.0 / (i + 1)
        return 0.0


class IDBasedNDCG(SingleTurnMetric):
    name = "id_ndcg"
    _required_columns = {
        MetricType.SINGLE_TURN: {"retrieved_context_ids", "reference_context_ids"}
    }

    def init(self, run_config: RunConfig) -> None: ...

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        retrieved = sample.retrieved_context_ids
        reference = set(sample.reference_context_ids)

        dcg = 0.0
        idcg = 0.0

        # Calculate DCG
        for i, doc_id in enumerate(retrieved):
            if doc_id in reference:
                dcg += 1.0 / np.log2(i + 2)

        # Calculate IDCG (Ideal DCG)
        num_relevant = min(len(reference), len(retrieved))
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval():
    with open(RAGAS_TESTSET_PATH, "r") as f:
        testset = [json.loads(line) for line in f]
    all_chunks = load_chunks_from_jsonl()

    questions = [item["user_input"] for item in testset]
    reference_context_ids = [
        map_reference_contexts_to_ids(item["reference_contexts"], all_chunks)
        for item in testset
    ]
    retrieved_context_ids = []

    retriever = HybridRetriever(k=60)
    print("Loading index and chunks into retriever...")
    retriever.load_from_file(FAISS_INDEX_PATH, CHUNKS_JSONL_PATH)

    modes = ["hybrid", "dense", "sparse"]
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*80}\n")

        retrieved_context_ids = []
        for i, question in tqdm(enumerate(questions), desc="Retrieving chunks"):
            match mode:
                case "hybrid":
                    retrieved_chunks = retriever.search_hybrid(question, 5)
                case "dense":
                    retrieved_chunks = retriever.search_dense(question, 5)
                case "sparse":
                    retrieved_chunks = retriever.search_sparse(question, 5)

            retrieved_context_ids.append(
                [chunk["chunk_id"] for chunk in retrieved_chunks]
            )
            # print(f"Retrieved context IDs: {retrieved_context_ids[i]}")
            # print(f"Reference context IDs: {reference_context_ids[i]}")
            # print("-" * 100)

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "retrieved_context_ids": retrieved_context_ids,
                "reference_context_ids": reference_context_ids,
            }
        )
        metrics = [IDBasedContextRecall(), IDBasedContextPrecision(), IDBasedMRR(), IDBasedNDCG()]
        results = evaluate(dataset, metrics)
        print(results)


if __name__ == "__main__":
    evaluate_retrieval()
