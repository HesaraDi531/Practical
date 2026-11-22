"""Streamlit interface for uploading PDFs and chatting with the RAG system."""
from pathlib import Path
import streamlit as st

from rag_core import UPLOADS_DIR, DOCUMENTS_DIR, ingest_pdfs, answer_question

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 University AI Assistant")
st.caption("Ask questions grounded in your indexed university PDF documents.")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Document management")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded PDFs are saved locally and indexed in ChromaDB.",
    )

    if st.button("Upload and index PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            saved_paths = []
            for uploaded in uploaded_files:
                safe_name = Path(uploaded.name).name
                destination = UPLOADS_DIR / safe_name
                destination.write_bytes(uploaded.getbuffer())
                saved_paths.append(destination)

            try:
                with st.spinner("Reading PDFs and creating searchable chunks..."):
                    result = ingest_pdfs(saved_paths)
                st.success(
                    f"Indexed {len(result['files'])} file(s), "
                    f"{result['pages']} page(s), and {result['chunks']} chunk(s)."
                )
                st.session_state.messages = []
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved document sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a question about your university documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                answer, documents = answer_question(question)
                st.markdown(answer)
                source_lines = []
                for doc in documents:
                    filename = doc.metadata.get("file_name", "Unknown document")
                    page = doc.metadata.get("page_number", "Unknown")
                    source_lines.append(f"{filename} — page {page}")
                if source_lines:
                    with st.expander("Retrieved document sources"):
                        for source in source_lines:
                            st.write(f"• {source}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_lines,
                    }
                )
            except Exception as exc:
                error_message = (
                    f"Something went wrong: {exc}\n\n"
                    "Confirm that Ollama is running, both models are installed, "
                    "and at least one PDF has been indexed."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
