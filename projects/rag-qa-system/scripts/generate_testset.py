import random
from collections import defaultdict

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from src.utils import EVAL_DATA_DIR, Chunk, get_openai_api_key, load_chunks_from_jsonl


def _convert_chunk_to_doc(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk["chunk_text"],
        id=chunk["chunk_id"],
        metadata={**chunk["metadata"]},
    )


def _get_openai_generator(model: str = "gpt-4o-mini") -> TestsetGenerator:
    get_openai_api_key()
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)


def _get_ollama_generator(model: str = "qwen2.5-coder:7b") -> TestsetGenerator:
    ollama_llm = LangchainLLMWrapper(
        ChatOllama(
            model=model, reasoning=False, temperature=0.0, num_ctx=8192, keep_alive="5m"
        )
    )
    ollama_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )
    generator = TestsetGenerator(llm=ollama_llm, embedding_model=ollama_embeddings)
    return generator


def _sample_chunks(chunks: list[Chunk]) -> list[Chunk]:
    chunks_by_paper = defaultdict(list)
    for chunk in chunks:
        chunks_by_paper[chunk["metadata"]["arxiv_id"]].append(chunk)
    sampled_chunks = []
    for paper_id, paper_chunks in chunks_by_paper.items():
        sampled_chunks.append(paper_chunks[0])
        num_to_sample = len(paper_chunks) // 6
        sampled_chunks.extend(random.sample(paper_chunks[1:], num_to_sample))
    return sampled_chunks


def generate_testset():
    """Generate a testset of 40 questions from the chunks."""
    chunks = load_chunks_from_jsonl()
    print(f"Loaded {len(chunks)} chunks")
    sampled_chunks = _sample_chunks(chunks)
    print(f"Sampled {len(sampled_chunks)} chunks")

    documents = [_convert_chunk_to_doc(chunk) for chunk in sampled_chunks][:5]

    generator = _get_ollama_generator()

    dataset = generator.generate_with_langchain_docs(
        documents, testset_size=5, with_debugging_logs=True, raise_exceptions=False
    )
    output_path = EVAL_DATA_DIR / "ragas_testset.jsonl"
    dataset.to_jsonl(str(output_path))


if __name__ == "__main__":
    generate_testset()
