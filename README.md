# LangChain Learning Repository

A hands-on learning project covering core LangChain concepts — document loading, text splitting, embeddings, vector stores, retrievers, and a full RAG application.

---

## Project Structure

```
Langchain/
├── noteboooks/              # Core concept notebooks
│   ├── app.ipynb
│   ├── text-splitter.ipynb
│   ├── embeddings.ipynb
│   ├── vectorstores.ipynb
│   ├── ollama.ipynb
│   └── retriever.ipynb
├── youtube/
│   └── youtube_chatbot.ipynb
├── data/                    # Sample PDFs for loaders
│   ├── drylab.pdf
│   ├── sample-local-pdf.pdf
│   └── sample-tables.pdf
├── faiss,indexes/           # Persisted FAISS vector store
├── test.py
├── requirements.txt
└── .env
```

---

## Notebooks

### [noteboooks/app.ipynb](noteboooks/app.ipynb) — Document Loaders
Covers all major LangChain document loaders:
- `TextLoader` — loads plain text files (used with `Goblet-of-Fire.txt`)
- `PyPDFLoader` — loads standard PDFs (used with `sample-local-pdf.pdf`)
- `PDFPlumberLoader` — extracts tables from PDFs (used with `sample-tables.pdf`)
- `PyMuPDFLoader` — loads image-heavy PDFs (used with `drylab.pdf`)
- `DirectoryLoader` — bulk loads all PDFs from a folder (`data/`)
- `WebBaseLoader` — scrapes web pages (tested with an Amazon product page)
- `CSVLoader` — loads CSV rows as documents (used with `support-tickets-100.csv`)

---

### [noteboooks/text-splitter.ipynb](noteboooks/text-splitter.ipynb) — Text Splitters
Covers how to chunk documents for LLM processing:
- `CharacterTextSplitter` — splits on a fixed character, with chunk size and overlap
- `RecursiveCharacterTextSplitter` — smarter splitting that tries to keep logical units intact
- `RecursiveCharacterTextSplitter.from_language` — language-aware splitting (Python code example)
- Also shows combining `PyPDFLoader` + splitter into a pipeline

---

### [noteboooks/embeddings.ipynb](noteboooks/embeddings.ipynb) — Embeddings
Covers converting text into vector embeddings for semantic search:
- `OpenAIEmbeddings` — using `text-embedding-3-large` from OpenAI
- `HuggingFaceEmbeddings` — using `sentence-transformers/all-MiniLM-L6-v2` locally (384 dimensions)
- Manual cosine similarity with `sklearn` to rank documents by relevance to a query

---

### [noteboooks/ollama.ipynb](noteboooks/ollama.ipynb) — Ollama Embeddings
Covers using local models via Ollama for embeddings:
- `OllamaEmbeddings` with `nomic-embed-text` model
- Semantic search over long documents using cosine similarity
- Ranks 5 documents about Python memory management by relevance

---

### [noteboooks/vectorstores.ipynb](noteboooks/vectorstores.ipynb) — Vector Stores
Covers two vector store implementations for storing and querying embeddings:
- **Chroma** — persistent vector store with `add_documents`, `similarity_search`, `update_document`, `delete`
- **FAISS** — fast in-memory vector store with `similarity_search_with_score`
- Saving and loading FAISS indexes to/from disk (`faiss,indexes/`)

---

### [noteboooks/retriever.ipynb](noteboooks/retriever.ipynb) — Retrievers
Covers retrieval strategies for RAG pipelines:
- `WikipediaRetriever` — fetches relevant Wikipedia articles for a query
- **Vector store retriever** (Chroma) — standard top-k semantic search (`k=3`)
- **MMR retriever** (FAISS) — Maximal Marginal Relevance search for diverse results (`lambda_mult` controls diversity)

---

### [youtube/youtube_chatbot.ipynb](youtube/youtube_chatbot.ipynb) — YouTube RAG Chatbot
A full end-to-end RAG application that answers questions about any YouTube video:

**Pipeline:**
1. `YouTubeTranscriptApi` — fetches the video transcript by video ID
2. `RecursiveCharacterTextSplitter` — splits transcript into 500-char chunks (50 overlap)
3. `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) — embeds all chunks
4. `FAISS` — stores chunk embeddings for semantic retrieval
5. Vector store retriever — retrieves top-4 relevant chunks for a query
6. `PromptTemplate` — formats context + question into a prompt
7. `OllamaLLM` (`llama3`) — generates the final answer locally

> Also supports `ChatBedrock` (AWS Bedrock) as a drop-in replacement for the LLM step.

---

## Other Files

| File | Description |
|------|-------------|
| [test.py](test.py) | Workflow classes for a medical draft generation system using Claude on Bedrock |
| [requirements.txt](requirements.txt) | All Python dependencies for the project |
| [support-tickets-100.csv](support-tickets-100.csv) | Sample CSV with 100 support tickets, used in `app.ipynb` |
| [Goblet-of-Fire.txt](Goblet-of-Fire.txt) | Harry Potter text, used as sample document in `app.ipynb` |
| `data/` | PDFs used as sample data for document loader examples |
| `faiss,indexes/` | Persisted FAISS index from `vectorstores.ipynb` |

---

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add API keys to .env
OPENAI_API_KEY=your_key_here

# For Ollama (local LLMs)
# Install from https://ollama.ai
ollama pull llama3
ollama pull nomic-embed-text
```

## Dependencies

Key packages used:
- `langchain`, `langchain-community`, `langchain-core`
- `langchain-huggingface` — HuggingFace embeddings and models
- `langchain-aws` — AWS Bedrock (Claude, Llama via Bedrock)
- `langchain-ollama` — local LLMs via Ollama
- `langchain-openai` — OpenAI GPT models and embeddings
- `langchain-chroma`, `chromadb` — Chroma vector store
- `faiss-cpu` — FAISS vector store
- `youtube-transcript-api` — YouTube transcript fetching
- `langchain-text-splitters` — text chunking utilities
