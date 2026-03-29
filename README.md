# 📄 Chat with your PDF

A lightweight RAG (Retrieval-Augmented Generation) app that lets you upload a PDF and ask questions about it in plain English. Built with LangChain, Groq, HuggingFace embeddings, FAISS, and Streamlit.

---

## How it works

1. **Upload** — Drop a PDF via the Streamlit UI
2. **Parse** — `PyPDFLoader` extracts text page by page
3. **Chunk** — `RecursiveCharacterTextSplitter` breaks it into 500-token chunks (50-token overlap)
4. **Embed** — `all-MiniLM-L6-v2` (HuggingFace) converts chunks to vectors, stored in a local FAISS index
5. **Retrieve** — On each question, the most relevant chunks are pulled from FAISS
6. **Generate** — `llama-3.3-70b-versatile` via Groq answers using only the retrieved context

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| Vector Store | FAISS (in-memory) |
| PDF Parsing | PyPDFLoader |

---

## Project Structure

```
pdf-rag-app/
├── app.py              # Full RAG pipeline + Streamlit UI
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/kanikajadhav/pdf-rag-app.git
cd pdf-rag-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com/).

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Dependencies

```
streamlit
langchain
langchain-groq
langchain-huggingface
langchain-community
langchain-core
langchain-text-splitters
faiss-cpu
pypdf
sentence-transformers
python-dotenv
```

---

## Limitations

- FAISS index is in-memory — resets when you upload a new PDF or refresh the page
- Works best with text-based PDFs; scanned/image-only PDFs won't parse correctly
- Answers are intentionally restricted to document context only (`Answer based only on the context below`)

---
