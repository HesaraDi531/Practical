"""Command-line PDF ingestion script."""
from pathlib import Path
import argparse

from rag_core import DOCUMENTS_DIR, ingest_pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load PDF files, split them into chunks, and index them in ChromaDB."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="Optional PDF file paths. If omitted, all PDFs in data/documents are indexed.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.pdfs] if args.pdfs else sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not paths:
        print(f"No PDFs found. Add PDF files to: {DOCUMENTS_DIR}")
        return

    result = ingest_pdfs(paths)
    print("\nIngestion completed")
    print(f"Files processed : {len(result['files'])}")
    print(f"Pages loaded    : {result['pages']}")
    print(f"Chunks indexed  : {result['chunks']}")
    for name in result["files"]:
        print(f" - {name}")


if __name__ == "__main__":
    main()
