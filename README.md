# 🩺 MediBot — AI-Powered Medical Chatbot

> An intelligent conversational medical assistant with **RAG (Retrieval-Augmented Generation)**, full **conversation memory**, built with Groq, Pinecone, LangChain, and Streamlit.

---

## demo - <img width="1299" height="994" alt="Screenshot 2026-03-18 203220" src="https://github.com/user-attachments/assets/90aaae0d-5c50-40c8-88fb-ecaf5aa7362c" />


## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────┐
│         Streamlit UI                │
│   st.chat_input / st.chat_message   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│           History-Aware Retriever                       │
│  Reformulates follow-up questions using chat history    │
│  using Groq LLM + contextualize prompt                  │
└──────────────┬──────────────────────────────────────────┘
               │ standalone question
               ▼
┌─────────────────────────────────────┐
│         Pinecone Vector Store       │
│  medical-chatbot index (384-dim)    │
│  Similarity search → Top-3 chunks  │
└──────────────┬──────────────────────┘
               │ retrieved context chunks
               ▼
┌─────────────────────────────────────────────────────────┐
│             Groq LLM (llama-3.1-8b-instant)             │
│  system_prompt + chat_history + context + question      │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
         Bot Answer (displayed in chat box)
```

### Data Ingestion Flow (one-time, `store_index.py`)
```
Medical_book.pdf
      │
      ▼  PyPDFLoader
Raw Pages (loaded via LangChain DirectoryLoader)
      │
      ▼  RecursiveCharacterTextSplitter (chunk_size=500, overlap=20)
Text Chunks
      │
      ▼  HuggingFace all-MiniLM-L6-v2 (384-dim embeddings)
Embeddings
      │
      ▼  Pinecone Serverless 
Vector Index → "medical-chatbot"
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | [Groq](https://groq.com) — `llama-3.1-8b-instant` (ultra-fast inference) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim, free) |
| **Vector Store** | [Pinecone](https://pinecone.io) Serverless (cosine similarity) |
| **RAG Framework** | [LangChain](https://langchain.com) — retrieval chain + history-aware retriever |
| **Conversation Memory** | `create_history_aware_retriever` + `MessagesPlaceholder` |
| **UI** | [Streamlit](https://streamlit.io) — native `st.chat_message`, `st.container(height=)` |

---

## ⚙️ How It Works

### 1. Knowledge Base Creation (one-time)
Run `store_index.py` once to:
- Load `Medical_book.pdf` from the `data/` folder
- Split it into 500-character overlapping chunks
- Embed using HuggingFace's free `all-MiniLM-L6-v2` model (returns 384-dim vectors)
- Upsert all vectors into a Pinecone serverless index named `medical-chatbot`

### 2. Answering a Question (each turn)
1. **History-aware reformulation** — the LLM (Groq) rewrites follow-up questions into self-contained questions. E.g. *"What causes it?"* becomes *"What causes diabetes?"*
2. **Vector retrieval** — the reformulated question is embedded and top-3 similar chunks are fetched from Pinecone
3. **Answer generation** — Groq's LLaMA 3.1 model is called with:
   - `system_prompt` (you are a medical assistant…)
   - Full `chat_history` (all prior turns as `HumanMessage`/`AIMessage`)
   - Retrieved `context` chunks
   - Current `input`

### 3. Chat Memory
Every turn, `st.session_state.messages` stores the full conversation. On each invoke, all previous messages are converted to LangChain `HumanMessage`/`AIMessage` objects and passed to the chain — giving the bot full memory of the conversation.

---

## 🚧 Problems I Faced

### 1. ❌ OpenAI → Groq Migration
The original project used **OpenAI GPT-4o**. I didn't have an OpenAI key, so I migrated to **Groq** (`llama-3.1-8b-instant`). Issues:
- `langchain-openai` replaced with `langchain-groq`
- The `llama3-8b-8192` model was **decommissioned** on Groq mid-project — had to switch to `llama-3.1-8b-instant`

### 2. ❌ Flask → Streamlit Migration
The original codebase used **Flask** with custom HTML/CSS templates. I migrated to Streamlit for:
- Simpler deployment (no WSGI server needed)
- Native chat UI (`st.chat_message`)
- Built-in session state for memory

### 3. ❌ Raw HTML Rendered as Code
When using `st.markdown(unsafe_allow_html=True)` with large nested `<div>` blocks + `<script>` tags, Streamlit rendered the entire block as raw `<pre>` code instead of HTML. Fixed by switching to native `st.chat_message()` + `st.container(height=...)` instead of custom HTML.

### 4. ❌ New Messages Appearing Outside Scroll Box
After switching to `st.container(height=500)` for scrolling, new messages rendered separately outside the box (in the input handler block). Fixed by removing inline rendering and using `st.rerun()` — ensuring all messages always render inside the bounded container.

### 5. ❌ Chatbot Not Remembering Conversations
The first RAG implementation passed only the current question — the bot had no memory of prior turns. Fixed by:
- Adding `create_history_aware_retriever` with a contextualize prompt
- Passing full `chat_history` as `HumanMessage`/`AIMessage` objects to the chain

### 6. ❌ Deprecated LangChain Imports
`langchain.document_loaders` and `langchain.embeddings` are deprecated. Updated to:
- `langchain_community.document_loaders`
- `langchain_huggingface.HuggingFaceEmbeddings`

### 7. ❌ `store_index.py` Crashing
The original `store_index.py` tried to set `OPENAI_API_KEY=None` (since we don't have OpenAI), causing `TypeError: str expected, not NoneType` in `os.environ`. Fixed by removing the OpenAI reference.

---

## 📁 Project Structure

```
medi_bot/
├── .streamlit/
│   └── config.toml        # headless mode + dark theme
├── src/
│   ├── __init__.py
│   ├── helper.py          # PDF loader, text splitter, embeddings
│   └── prompt.py          # System prompt for the medical assistant
├── data/                  # Place Medical_book.pdf here (gitignored)
├── .env                   # API keys (gitignored)
├── .gitignore
├── README.md
├── render.yaml            # Render deployment blueprint
├── requirements.txt       # All dependencies pinned
├── setup.py
├── store_index.py         # One-time indexing script
└── streamlit_app.py       # Main Streamlit application
```

---

## 💻 Local Setup

### Prerequisites
- Python 3.10+
- A [Pinecone](https://pinecone.io) account (free tier works)
- A [Groq](https://groq.com) API key (free)

### Steps

```bash
# 1. Clone
git clone https://github.com/Pawantripathi2606/medi_bot
cd medi_bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env
echo "PINECONE_API_KEY=your_pinecone_key" >> .env
echo "GROQ_API_KEY=your_groq_key" >> .env

# 4. Place Medical_book.pdf in data/ folder

# 5. Index the PDF into Pinecone (one-time)
python store_index.py

# 6. Run the app
streamlit run streamlit_app.py
```

Open `http://localhost:8501` 🎉

---

## ☁️ Deploy on Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New → Blueprint**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Set environment variables in Render dashboard:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
5. Click **Deploy** ✅
.
> **Note:** Run `store_index.py` locally first to populate your Pinecone index. The deployed app only queries the existing index.

---


