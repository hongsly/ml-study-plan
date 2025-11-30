import json

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.rag_pipeline import RagAssistant
from src.utils import EVAL_DATA_DIR, get_openai_api_key
from tqdm import tqdm

RAGAS_TESTSET_PATH = EVAL_DATA_DIR / "ragas_testset.jsonl"
MANUAL_TESTSET_PATH = EVAL_DATA_DIR / "test_questions.json"


def _generate_responses(questions: list[str], assistant: RagAssistant):
    response_list = []
    retrieved_contexts_list = []
    for question in questions:
        answer, context = assistant.query(question)
        response_list.append(answer)
        if context:
            retrieved_contexts_list.append([c["chunk_text"] for c in context])
        else:
            retrieved_contexts_list.append([])

    response_dict = {
        "user_input": questions,
        "response": response_list,
    }
    if assistant.retrieval_mode != "none":
        response_dict["retrieved_contexts"] = retrieved_contexts_list
    return response_dict


def _evaluate(
    ragas_testset: list[dict],
    manual_testset: list[dict],
    mode: str,
    llm: LangchainLLMWrapper,
):
    assistant = RagAssistant(model="gpt-4o-mini", retrieval_mode=mode, top_k=5)

    questions_ragas = [item["user_input"] for item in ragas_testset]
    dataset_dict_ragas = _generate_responses(
        tqdm(questions_ragas, desc="Generating responses for ragas_testset"), assistant
    )
    dataset_dict_ragas["reference"] = [item["reference"] for item in ragas_testset]
    dataset_with_reference = Dataset.from_dict(dataset_dict_ragas)
    dataset_with_reference.to_json(
        str(EVAL_DATA_DIR / f"response_dataset_{mode}_with_reference.jsonl")
    )
    print(
        f"Ragas testset responses written to {EVAL_DATA_DIR / f'response_dataset_{mode}_with_reference.jsonl'}"
    )

    questions_manual = [item["question"] for item in manual_testset]
    dataset_dict_manual = _generate_responses(
        tqdm(questions_manual, desc="Generating responses for manual testset"),
        assistant,
    )
    dataset_dict_combined = {
        "user_input": questions_manual + questions_ragas,
        "response": dataset_dict_manual["response"] + dataset_dict_ragas["response"],
    }
    if mode != "none":
        dataset_dict_combined["retrieved_contexts"] = (
            dataset_dict_manual["retrieved_contexts"]
            + dataset_dict_ragas["retrieved_contexts"]
        )
    dataset_no_reference = Dataset.from_dict(dataset_dict_combined)

    dataset_no_reference.to_json(
        str(EVAL_DATA_DIR / f"response_dataset_{mode}_no_reference.jsonl")
    )
    print(
        f"No-reference testset responses written to {EVAL_DATA_DIR / f'response_dataset_{mode}_no_reference.jsonl'}"
    )

    # answer_correctness need: user_input, response, reference
    # context_precision, context_recall need: user_input, retrieved_contexts, reference
    if mode == "none":
        metrics_need_reference = [answer_correctness]
    else:
        metrics_need_reference = [
            answer_correctness,
            context_precision,
            context_recall,
        ]

    print("Evaluating with reference...")
    results_with_reference = evaluate(
        dataset_with_reference,
        metrics_need_reference,
        llm=llm,
    )
    results_with_reference.to_pandas().to_json(
        str(EVAL_DATA_DIR / f"eval_results_{mode}_with_reference.json")
    )

    # answer_relevancy need: user_input, response
    # faithfulness need: user_input, response, retrieved_contexts
    metrics_no_reference = [answer_relevancy, faithfulness]
    if mode == "none":
        metrics_no_reference = [answer_relevancy]
    print("Evaluating with no-reference...")
    results_no_reference = evaluate(
        dataset_no_reference,
        metrics_no_reference,
        llm=llm,
    )
    results_no_reference.to_pandas().to_json(
        str(EVAL_DATA_DIR / f"eval_results_{mode}_no_reference.json")
    )

    print("Evaluation completed.")

    print(results_with_reference)
    print(results_no_reference)


def evaluate_rag():
    with open(RAGAS_TESTSET_PATH, "r") as f:
        ragas_testset = [json.loads(line) for line in f]
    with open(MANUAL_TESTSET_PATH, "r") as f:
        manual_testset = json.load(f)
    print("Loaded ragas_testset and manual_testset")

    get_openai_api_key()
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

    modes = ["hybrid", "dense", "sparse", "none"]
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*80}\n")
        _evaluate(ragas_testset, manual_testset, mode, llm)


if __name__ == "__main__":
    evaluate_rag()
