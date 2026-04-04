# 📄 Chat With Documents

A Streamlit-based RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents, ask questions, and get answers grounded in the document content using vector search + hybrid search (BM25) + reranking.

## ✨ Features
- Upload PDFs – automatic chunking, embedding, and storage
- Hybrid search – vector similarity + BM25 keyword search
- Reranking with Cross-Encoder for better accuracy
- Conversation history per user
- Section‑specific queries (e.g., "What is Section 62?")
- Follow‑up question support (summarize, elaborate, etc.)

## 🧠 Tech Stack
- **Frontend**: Streamlit
- **Embeddings**: Ollama (`nomic-embed-text`)
- **LLM**: Ollama (`llama3.2` or any fine‑tuned model)
- **Vector DB**: PostgreSQL + `pgvector`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-2-v2`
- **BM25**: `rank_bm25`


---


## 🚀 Setup Guide

### 1. Install System Dependencies

#### PostgreSQL with pgvector
- **Ubuntu/Debian**:
  ```bash
  sudo apt update
  sudo apt install postgresql postgresql-contrib
  sudo -u postgres psql -c "CREATE EXTENSION vector;"
  ```

- **MacOS**:
    ```bash
    brew install postgresql
    brew install pgvector
    ```
- **Windows (Recommended to use WSL or Docker)**:
    
#### Ollama: 
- **Linux / macOS / WSL2**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
- **Windows (native)** – download from https://ollama.com/download
### 2. Pull Required Models

```
# Embedding model (768 dimensions)

ollama pull nomic-embed-text

# LLM (change to any model you prefer)

ollama pull llama3.2

# Optional: verify models are downloaded

ollama list
```

### 3. Create PostgreSQL Database

    sudo -u postgres psql


#### Inside PostgreSQL :

```
CREATE DATABASE document_chat;
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE document_chat TO your_user;
\c document_chat
CREATE EXTENSION vector;
\q

```

### 4. Clone & Setup Python Environment

```
git clone https://github.com/Alayasokan/PostgreSQL-RAG.git

cd PostgreSQL-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or .\venv\Scripts\activate  (Windows)

# Install dependencies
pip install -r requirements.txt

```

### 5. Configure Environment Variables

Remove `.env.example` and set up the `.env` file using the database credentials (password, port, host, etc.) provided in `.env.example`.

Example: 

```
POSTGRES_DB_NAME=document_chat
POSTGRES_DB_HOST=localhost
POSTGRES_DB_PORT=5432
POSTGRES_DB_USER=your_user
POSTGRES_DB_PASSWORD=your_password

OPENAI_API_KEY=ollama          # keep as 'ollama' for local
OLLAMA_BASE_URL=http://localhost:11434/v1

FINE_TUNED_MODEL=llama3.2      # or your custom model name
USE_HYBRID_SEARCH=true

```

### 6. Run the Application

```
streamlit run app.py

```

# 📂 Project Structure

```
.
├── app.py                     # Main entry point
├── pages/
│   ├── chat_document.py       # Chat interface
│   ├── Manage_Document.py     # Upload & delete documents
│   └── Manage_tag.py          # Tag management
├── db.py                      # PostgreSQL + peewee models
├── ollama_client.py           # OpenAI‑compatible client
├── constants.py               # System prompts
├── reranker.py                # Cross-encoder reranking
├── bm25_index.py              # BM25 index builder
├── utils.py                   # Helper functions
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

```