from typing import TypedDict

import tiktoken


class Chunk(TypedDict):
    chunk_id: str
    chunk_text: str
    token_count: int

def chunk_text(
    text: str,
    model_name: str = "gpt-3.5-turbo",
    chunk_size: int = 500,
    overlap: int = 50,
    parent_document_name: str = None,
) -> list[Chunk]:
    """Chunk text into chunks of chunk_size tokens with overlap"""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)  # return a list of token ids

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        chunk_id = f"chunk_{i}"
        if parent_document_name:
            chunk_id = parent_document_name + chunk_id
        chunk_text = enc.decode(chunk)
        chunks.append(Chunk(chunk_id=chunk_id, chunk_text=chunk_text, token_count=len(chunk)))

    return chunks
