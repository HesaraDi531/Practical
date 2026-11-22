
# Lab Sheet 5 – Implementing a Retrieval-Augmented Generation (RAG) System

**Module:** PUSL 3205 – Large Language Models and Generative AI  
**Practical:** Lab Sheet 5 – RAG Architecture  
**Domain:** University Knowledge Question Answering  
**LLM:** `llama3.2:3b`  
**Embedding Model:** `qwen3-embedding:0.6b`  
**Vector Database:** ChromaDB  

## 1. Introduction

This practical demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using Python, LangChain, Ollama and ChromaDB. The main objective is to develop a university knowledge assistant capable of retrieving relevant information from PDF documents and generating answers based on their contents.

RAG combines document retrieval with Large Language Model (LLM) response generation. Instead of relying only on the knowledge acquired during model training, the system retrieves relevant information from external documents and provides it to the language model as context.

The implementation includes PDF document loading, text splitting, embedding generation, vector storage, similarity-based retrieval and response generation. A Streamlit-based chatbot interface is also developed to allow users to interact with the assistant and upload their own PDF documents.

## 2. Objectives

- Understand the architecture and workflow of Retrieval-Augmented Generation.
- Implement a document ingestion pipeline for processing PDF documents.
- Generate text embeddings using an embedding model.
- Store and retrieve document embeddings using ChromaDB.
- Integrate a locally hosted LLM using Ollama.
- Develop a RAG pipeline for document-based question answering.
- Create an interactive chatbot using Streamlit.
- Extend the application to support multiple PDF documents and user uploads.
- Understand the limitations of basic RAG architectures and possible improvements.

## 3. Tools and Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | Framework for developing RAG applications |
| LangChain Community | Document loaders and integrations |
| LangChain Ollama | Connecting local language and embedding models |
| LangChain Chroma | ChromaDB integration |
| Ollama | Running language and embedding models locally |
| Llama 3.2 3B | Generating answers from retrieved document context |
| Qwen3 Embedding 0.6B | Generating document and query embeddings |
| ChromaDB | Vector database for storing and retrieving embeddings |
| PyPDF | Extracting text from PDF documents |
| PyMuPDF | PDF processing |
| Streamlit | Developing the chatbot web interface |
| Python-dotenv | Managing application configuration |

## 4. RAG Architecture

The application follows two main processes: document ingestion and question answering.

### Document Ingestion

The document ingestion process converts university PDF documents into searchable vector representations.

The following stages are implemented:

1. **Document Loading:** PDF documents are loaded using PyPDFLoader.
2. **Text Extraction:** Text content is extracted from individual PDF pages.
3. **Text Splitting:** The extracted content is divided into smaller chunks using RecursiveCharacterTextSplitter.
4. **Embedding Generation:** The Qwen3 Embedding model converts document chunks into numerical vector representations.
5. **Vector Storage:** The generated embeddings and associated metadata are stored in ChromaDB.

### Retrieval and Generation

When a user submits a question, the following process is performed:

1. The user's question is received by the application.
2. ChromaDB performs similarity-based retrieval to identify relevant document chunks.
3. Retrieved information is passed to the prompt template.
4. The prompt and retrieved context are provided to the Llama 3.2 language model.
5. The LLM generates an answer using the supplied document information.
6. The answer and retrieved source information are displayed through the chatbot interface.

### RAG Workflow

```text
       University PDF Documents
                  |
                  v
             PDF Loader
                  |
                  v
             Text Splitter
                  |
                  v
         Qwen3 Embedding Model
                  |
                  v
              ChromaDB
                  |
                  |
             User Question
                  |
                  v
          Similarity Retrieval
                  |
                  v
        Relevant Document Chunks
                  |
                  v
          Prompt Construction
                  |
                  v
           Llama 3.2 3B
                  |
                  v
           Generated Answer
                  |
                  v
        Chatbot User Interface
```

## 5. Implementation

The application is developed using separate Python files to organize document ingestion, retrieval, response generation and user interaction.

### 5.1 Document Ingestion – `ingest.py`

The `ingest.py` file is responsible for processing PDF documents and storing their extracted content in ChromaDB.

The implementation performs the following operations:

- Identifies PDF documents available in the documents directory.
- Loads the documents using PyPDFLoader.
- Extracts text content from individual PDF pages.
- Splits the extracted content into smaller overlapping chunks.
- Generates embeddings using the Qwen3 embedding model.
- Stores the document chunks and metadata in ChromaDB.

The script processes the available PDF documents and reports the number of documents, pages and chunks indexed.

### 5.2 RAG Core Implementation – `rag_core.py`

The `rag_core.py` file contains the main functionality shared by the ingestion script, terminal application and Streamlit interface.

The implementation includes:

- Initializing the Ollama embedding model.
- Connecting to the persistent ChromaDB collection.
- Loading and processing PDF documents.
- Splitting documents into smaller overlapping chunks.
- Generating and storing document embeddings.
- Retrieving relevant document chunks using similarity search.
- Constructing prompts using retrieved context.
- Generating answers using the Llama 3.2 language model.

The text splitter uses the following configuration:

| Parameter | Value |
|---|---:|
| Chunk size | 800 |
| Chunk overlap | 150 |
| Default retrieval top-k | 4 |

The chunking configuration allows the system to process document content in smaller sections while retaining surrounding context between consecutive chunks.

### 5.3 RAG Question-Answering Pipeline – `rag_pipeline.py`

The `rag_pipeline.py` file implements a terminal-based question-answering interface.

The application continuously accepts questions from users and passes them to the RAG pipeline.

For each question:

1. Relevant document chunks are retrieved from ChromaDB.
2. Retrieved information is added to the prompt.
3. The language model generates an answer using the supplied context.
4. The generated answer is displayed in the terminal.
5. The filenames and page numbers of retrieved documents are displayed.

This allows users to interact with the university knowledge assistant through a command-line interface.

### 5.4 Streamlit Web Application – `app.py`

A Streamlit-based web interface is developed to provide a user-friendly way of interacting with the RAG system.

The interface includes:

- A chatbot interface for submitting university-related questions.
- A PDF upload component.
- Support for uploading multiple PDF documents.
- Automatic indexing of uploaded documents.
- Integration with the ChromaDB vector database.
- Displaying generated answers.
- Displaying retrieved document filenames and page numbers.
- Chat history for the current application session.

The web interface allows users to interact with the university knowledge assistant without directly using the terminal.

### 5.5 Multiple PDF Document Support

The original university knowledge assistant is extended to support multiple PDF documents instead of depending on a single university document.

Users can upload their own PDF files through the Streamlit interface.

After uploading, the documents are saved in the local uploads directory and processed through the document ingestion pipeline.

The extracted content is divided into chunks, converted into embeddings and stored in the same ChromaDB collection.

This allows the assistant to retrieve relevant information from multiple indexed documents when answering user questions.

### 5.6 Source Document Information

Document metadata is preserved during the ingestion process.

The metadata includes:

- Original document filename.
- Source document path.
- PDF page number.

When the system retrieves relevant document chunks, the source filenames and page numbers are displayed alongside the generated responses.

This allows users to identify the documents used to generate an answer.

## 6. Experimental Results

The implementation is evaluated by processing university PDF documents and testing the question-answering functionality.

### 6.1 Document Ingestion Results

The ingestion process reports the number of processed documents, extracted pages and indexed chunks.

| Metric | Result |
|---|---|
| Number of PDF documents | To be recorded |
| Total pages processed | To be recorded |
| Total chunks generated | To be recorded |
| Vector database | ChromaDB |
| Embedding model | Qwen3 Embedding 0.6B |

### 6.2 Question-Answering Evaluation

The application can be evaluated using questions related to the uploaded university documents.

| Test Case | Expected Behaviour |
|---|---|
| Ask about a university degree programme | Retrieve relevant programme information |
| Ask about subjects in a degree | Retrieve corresponding subject information |
| Ask about a specific module | Retrieve relevant module details |
| Ask a question requiring multiple documents | Retrieve relevant information from indexed documents |
| Upload an additional PDF | Process and index the newly uploaded document |
| Ask about the uploaded PDF | Generate an answer using the newly indexed content |
| Ask an unrelated question | Indicate when sufficient supporting information is unavailable |

The evaluation focuses on retrieval relevance, answer correctness, source identification and the ability to process multiple documents.

### 6.3 Results and Evidence

The following screenshots should be included in the `screenshots/` directory:

| Screenshot | Description |
|---|---|
| `01_ollama_models.png` | Installed language and embedding models |
| `02_chromadb.png` | ChromaDB installation verification |
| `03_ingestion.png` | PDF document ingestion results |
| `04_terminal_rag.png` | Terminal-based question-answering results |
| `05_streamlit_ui.png` | University AI Assistant web interface |
| `06_pdf_upload.png` | Uploading and indexing multiple PDF documents |
| `07_rag_response.png` | Generated answers and retrieved document sources |

Actual screenshots and experimental outputs should be included after executing the application.

## 7. Observations

The practical demonstrates how Retrieval-Augmented Generation connects locally hosted language models with external document collections.

The embedding model converts document content into vector representations, allowing ChromaDB to identify semantically relevant passages based on user questions.

The retrieval process provides additional context to the language model, helping it generate answers based on the uploaded university documents rather than relying entirely on its pre-trained knowledge.

The Streamlit interface makes the system more accessible by allowing users to upload multiple PDF documents and interact with the assistant through a web application.

The implementation also demonstrates the importance of document chunking, embedding generation and retrieval quality in the overall performance of a RAG application.

However, the quality of generated answers depends on the relevance of retrieved content, the structure of the source documents and the language model's ability to interpret the supplied context.

## 8. Limitations and Challenges

The implemented basic RAG architecture has several limitations:

- Fixed-size text chunking may separate related information across different chunks.
- Dense vector similarity search may not always retrieve the most relevant document passages.
- Scanned PDF documents without extractable text require additional OCR processing.
- Complex questions involving information from several document sections may require more advanced retrieval methods.
- Answer quality depends on the relevance of retrieved context and the language model.
- The baseline implementation does not include BM25 retrieval, reranking or knowledge graph-based retrieval.

These limitations demonstrate the need for advanced RAG architectures to improve retrieval quality and answer reliability.

## 9. Group Activity – Enhancing a University Knowledge Assistant Using Advanced RAG Architectures

The laboratory sheet introduces a group activity titled:

**From Naive RAG to Intelligent University Knowledge Retrieval**

The objective is to investigate, design, implement and evaluate an advanced RAG architecture that improves at least one limitation of the existing university knowledge assistant.

Five possible approaches are identified in the laboratory sheet.

### Group 1 – Hybrid Retrieval and Reranking

Implement BM25 and vector retrieval, rank fusion and reranking.

Compare Recall@5, Mean Reciprocal Rank (MRR), answer correctness and latency against the basic RAG implementation.

### Group 2 – Context-Aware / Hierarchical RAG

Compare fixed-size chunking, section-based chunking and parent-child retrieval.

Investigate which approach performs better when processing lengthy academic documents.

### Group 3 – Corrective and Evidence-Verified RAG

Develop a retrieval-quality checker and retry mechanism.

Investigate whether the proposed approach reduces unsupported answers and improves abstention accuracy.

### Group 4 – Knowledge Graph RAG

Extract entities such as degree programmes, modules, prerequisites and departments.

Develop a knowledge graph and evaluate its ability to answer multi-hop academic questions.

### Group 5 – Multilingual University RAG

Evaluate English, Sinhala and Tamil questions against university documents.

Compare multilingual embeddings, translated queries and cross-lingual retrieval.

### Expected Deliverables

- Working enhanced RAG application.
- Proposed architecture diagram.
- Source code and setup instructions.
- Evaluation dataset containing questions, reference answers and supporting documents.
- Experimental results comparing baseline and proposed approaches.
- Technical report discussing methodology, findings and limitations.
- Live demonstration.

The assigned group approach should be implemented and evaluated separately from the baseline RAG system.

## 10. Learning Outcomes

Through this practical, I gained experience in understanding and implementing the main components of a Retrieval-Augmented Generation architecture.

I developed practical knowledge of loading and processing PDF documents, splitting text into smaller chunks, generating embeddings and storing vector representations using ChromaDB.

I also gained experience integrating locally hosted language models through Ollama and developing document-grounded question-answering applications using LangChain.

Furthermore, implementing the Streamlit interface provided practical experience in developing interactive AI applications that support multiple PDF uploads and document-based information retrieval.

The practical also introduced the limitations of basic RAG systems and the importance of advanced retrieval techniques for improving answer accuracy and reliability.

## 11. Conclusion

This practical demonstrates the implementation of a Retrieval-Augmented Generation architecture using Python, LangChain, Ollama and ChromaDB.

The developed university knowledge assistant combines document ingestion, text splitting, embedding generation, vector storage, similarity-based retrieval and LLM-based response generation.

The Streamlit web interface extends the implementation by supporting multiple PDF documents and allowing users to upload their own files for document-based question answering.

The practical provides a foundation for understanding how external knowledge can be integrated with Large Language Models and introduces opportunities to improve retrieval performance through advanced RAG architectures.

## 12. References

1. PUSL 3205 – Large Language Models and Generative AI, Lab Sheet 5 – RAG Architecture.
2. LangChain Documentation – https://python.langchain.com/
3. Ollama Documentation – https://ollama.com/
4. ChromaDB Documentation – https://docs.trychroma.com/
5. Streamlit Documentation – https://docs.streamlit.io/
6. Python Documentation – https://docs.python.org/3/
