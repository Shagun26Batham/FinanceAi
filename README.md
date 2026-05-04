# FinanceAI 💹

> **Your private, offline financial advisor — powered by Phi-3 & RAG.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-purple?logo=ollama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

FinanceAI is a fully local, privacy-first financial advisor chatbot built with a **Retrieval-Augmented Generation (RAG)** pipeline. It runs 100% on your machine — no cloud APIs, no data leaks.

---

## ✨ Features

- **100% Local Execution** — All inference stays on your device. Zero data sent externally.
- **RAG Pipeline** — Retrieves relevant context from your own `.csv` financial data via ChromaDB.
- **Streaming Responses** — Real-time token streaming via Server-Sent Events (SSE) for a fluid, ChatGPT-like experience.
- **Conversation Memory** — Maintains the last 5 turns of chat history for contextual replies.
- **Modern UI** — Clean, responsive dark/light SaaS interface with auto-scroll, typing indicator, and welcome chips.
- **Markdown Rendering** — Bot replies are rendered as rich Markdown via `marked.js`.

---

## 🧱 Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | HTML, Vanilla JS, CSS               |
| Backend     | Python, Flask, Flask-CORS           |
| LLM         | Ollama (`phi3:latest`)              |
| Embeddings  | Ollama (`nomic-embed-text`)         |
| Vector DB   | ChromaDB (persistent local storage) |

---

## 📁 Repository Structure

```
FinanceAI/
├── app.py              # Flask app — routes, RAG logic, SSE streaming
├── ingest.py           # Data ingestion — CSV → embeddings → ChromaDB
├── requirements.txt    # Python dependencies
├── data/               # Place your .csv financial data files here
├── db/
│   └── chroma_db/      # Auto-generated vector index (created after ingest)
├── static/
│   ├── css/
│   │   └── style.css   # UI styling (dark/light themes)
│   └── js/
│       └── app.js      # Frontend chat logic, SSE handler, theme toggle
└── templates/
    └── index.html      # Main HTML template
```

---

## 🔌 API Reference

### `GET /`
Serves the main chat UI.

### `POST /chat`
Sends a user message and returns a **streaming SSE response**.

**Request body:**
```json
{ "message": "What is dollar-cost averaging?" }
```

**SSE Events emitted:**

| Event   | Payload             | Description                        |
|---------|---------------------|------------------------------------|
| `token` | `{ "data": "..." }` | One streamed token of the reply    |
| `done`  | `{ "data": "" }`    | Signals end of stream              |
| `error` | `{ "data": "..." }` | Error message string               |

### `POST /reset`
Clears the server-side conversation history.

**Response:**
```json
{ "status": "ok" }
```

---

## ⚙️ Prerequisites

1. **Python 3.8+**
2. **Ollama** — Install from [ollama.com](https://ollama.com/) and ensure it's running.

### Pull Required Models

```bash
ollama pull phi3:latest
ollama pull nomic-embed-text
```

---

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/FinanceAI.git
cd FinanceAI
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your financial data**

Create a `data/` directory and place your `.csv` files inside. Each row is parsed into a readable text chunk. Supported column names get special handling:

| Column name | Handling               |
|-------------|------------------------|
| `text`      | Used as-is             |
| `question`  | Prefixed with `Q: `    |
| `answer`    | Prefixed with `A: `    |
| *(other)*   | Prefixed with `key: `  |

**4. Ingest your data into ChromaDB**
```bash
python ingest.py
```
This generates embeddings and stores them in `db/chroma_db/`. Subsequent runs are incremental — only new chunks are embedded.

**5. Start the Flask server**
```bash
python app.py
```

**6. Open in your browser**
```
http://127.0.0.1:5000
```

---

## 🛠️ Configuration

Key constants are defined at the top of each file for easy tuning:

| Constant       | File        | Default             | Description                          |
|----------------|-------------|---------------------|--------------------------------------|
| `LLM_MODEL`    | `app.py`    | `phi3:latest`       | Ollama model used for chat           |
| `EMBED_MODEL`  | `app.py`    | `nomic-embed-text`  | Ollama model used for embeddings     |
| `TOP_K`        | `app.py`    | `3`                 | Number of RAG context chunks fetched |
| `MAX_HISTORY`  | `app.py`    | `5`                 | Conversation turns kept in memory    |
| `CHUNK_SIZE`   | `ingest.py` | `300`               | Words per chunk                      |
| `CHUNK_OVERLAP`| `ingest.py` | `75`                | Overlap between consecutive chunks  |
| `BATCH_SIZE`   | `ingest.py` | `32`                | Embedding batch size                 |

---

## 🔍 Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on `/chat` | Make sure `python app.py` is running and Ollama is active |
| `model not found` error | Run `ollama pull phi3:latest` and `ollama pull nomic-embed-text` |
| Empty responses from RAG | Check that `python ingest.py` completed without errors |
| No `.csv` data found | Ensure files are placed inside the `data/` directory |

---

## 👤 Author

**Shagun Batham**  
Full-Stack Developer  | Lovely Professional University

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).