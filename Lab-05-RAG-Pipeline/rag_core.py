"""Shared ingestion, retrieval, and generation functions for the RAG application."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "university_knowledge")
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "data" / "documents"))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", BASE_DIR / "data" / "uploads"))

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


def get_embeddings() -> OllamaEmbeddings:
    """Return the Ollama embedding model configured for this project."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store() -> Chroma:
    """Connect to the persistent Chroma collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def load_pdf(path: str | Path) -> list[Document]:
    """Load one PDF and retain its source and page metadata."""
    pdf_path = Path(path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported: {pdf_path.name}")

    documents = PyPDFLoader(str(pdf_path)).load()
    for doc in documents:
        doc.metadata["source"] = str(pdf_path)
        doc.metadata["file_name"] = pdf_path.name
        # PyPDFLoader normally stores zero-based page numbers.
        doc.metadata["page_number"] = int(doc.metadata.get("page", 0)) + 1
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split loaded pages into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_pdfs(paths: Iterable[str | Path]) -> dict:
    """Load, split, and store PDF documents in ChromaDB.

    Existing chunks for the same absolute source path are deleted first,
    preventing duplicate ingestion when a PDF is re-uploaded.
    """
    vector_store = get_vector_store()
    total_pages = 0
    total_chunks = 0
    processed_files: list[str] = []

    for item in paths:
        path = Path(item).resolve()
        pages = load_pdf(path)
        chunks = split_documents(pages)
        if not chunks:
            continue

        try:
            vector_store.delete(where={"source": str(path)})
        except Exception:
            # A new/empty collection may not contain matching records.
            pass

        vector_store.add_documents(chunks)
        total_pages += len(pages)
        total_chunks += len(chunks)
        processed_files.append(path.name)

    return {
        "files": processed_files,
        "pages": total_pages,
        "chunks": total_chunks,
    }


def retrieve_context(question: str, k: int | None = None) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""
    store = get_vector_store()
    return store.similarity_search(question, k=k or TOP_K)


def format_sources(documents: list[Document]) -> str:
    """Format retrieved chunks with source names and page numbers."""
    if not documents:
        return "No relevant document passages were retrieved."

    parts = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("file_name") or Path(
            doc.metadata.get("source", "Unknown document")
        ).name
        page = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown"))
        parts.append(
            f"[Source {index}: {source}, page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, k: int | None = None) -> tuple[str, list[Document]]:
    """Answer a question using retrieved PDF context and the local Ollama LLM."""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    documents = retrieve_context(question, k=k)
    if not documents:
        return (
            "I could not find relevant information in the indexed documents. "
            "Please upload or ingest the relevant university PDF.",
            [],
        )

    context = format_sources(documents)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a university knowledge assistant.
Answer the user's question using ONLY the supplied document context.
Do not invent module names, dates, rules, or facts.
If the context does not contain enough information, clearly say that the
uploaded documents do not provide enough information.
Use concise, well-structured answers. Where possible, mention the source
file and page number in the answer. Distinguish a direct statement from
an inference.""",
            ),
            (
                "human",
                "DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            ),
        ]
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, documents
