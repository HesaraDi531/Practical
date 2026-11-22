"""Terminal-based RAG question-answering application."""
from rag_core import answer_question


def main() -> None:
    print("=" * 65)
    print("UNIVERSITY KNOWLEDGE ASSISTANT — RAG")
    print("Ask questions about the indexed university PDF documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    while True:
        try:
            question = input("\nAsk a university question: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not question:
                print("Please enter a question.")
                continue

            answer, sources = answer_question(question)
            print("\nANSWER\n" + "-" * 65)
            print(answer)
            if sources:
                print("\nRETRIEVED SOURCES")
                for doc in sources:
                    filename = doc.metadata.get("file_name", "Unknown")
                    page = doc.metadata.get("page_number", "Unknown")
                    print(f"- {filename} (page {page})")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")
            print("Check that Ollama is running and that the models are installed.")


if __name__ == "__main__":
    main()
