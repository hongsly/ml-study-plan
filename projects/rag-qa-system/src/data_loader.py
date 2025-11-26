import json
from pathlib import Path

import numpy as np
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from src.utils import Chunk, chunk_text
from tqdm import tqdm


class PDFDocument:
    def __init__(self, path: Path):
        """Initialize with a PDF path.
        Converts the PDF to markdown and stores the markdown text.

        Args:
            path: Path to the PDF file.
        """
        self.path = path
        try:
            self.md_text = pymupdf4llm.to_markdown(str(self.path))
        except Exception as e:
            print(f"Error parsing PDF {self.path}: {e}")
            self.md_text = ""

    def get_markdown(self) -> str:
        return self.md_text

    def get_name(self) -> str:
        return self.path.stem


class CorpusLoader:
    def __init__(self):
        self.chunks = None

    def parse_pdfs(
        self, pdf_paths: list[Path], chunk_size: int = 500, overlap: int = 50
    ):
        pdf_documents = [PDFDocument(path) for path in tqdm(pdf_paths, desc="Parsing PDFs")]
        self.chunks = [
            chunk
            for pdf_document in pdf_documents
            for chunk in chunk_text(
                pdf_document.get_markdown(),
                chunk_size=chunk_size,
                overlap=overlap,
                parent_document_name=pdf_document.get_name(),
            )
        ]

    def save_chunks(self, output_path: Path):
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        with open(output_path, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk) + "\n")

    def get_chunks(self) -> list[Chunk]:
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        return self.chunks

    def get_statistics(self) -> dict:
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        
        token_counts = [chunk["token_count"] for chunk in self.chunks]
        return {
            "num_chunks": len(self.chunks),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
        }
