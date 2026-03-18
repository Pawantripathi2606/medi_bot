import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.prompt import system_prompt
from dotenv import load_dotenv
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MedBot – Medical Assistant",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 MedBot")
st.caption("Your intelligent medical assistant — powered by Groq & Pinecone")
st.divider()


# ── Backend (load once) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_chain():
    load_dotenv()
    pinecone_key = os.environ.get("PINECONE_API_KEY") or ""
    groq_key     = os.environ.get("GROQ_API_KEY") or ""
    os.environ["PINECONE_API_KEY"] = pinecone_key

    embeddings = download_hugging_face_embeddings()
    docsearch  = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot", embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)

    # ── Prompt 1: reformulate user question using chat history ──
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate a standalone question that can be understood "
         "without the chat history. Do NOT answer — just reformulate if needed, "
         "otherwise return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # ── Prompt 2: answer with context + history ──────────────────
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


rag_chain = load_chain()

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history (scrollable) ─────────────────────────────────
chat_area = st.container(height=500)
with chat_area:
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🩺"):
            st.write("👋 Hello! I'm MedBot, your medical assistant. "
                     "Ask me anything about symptoms, conditions, or medications.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🩺" if m["role"] == "assistant" else "👤"):
            st.write(m["content"])

# ── Input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build LangChain chat_history from session messages (exclude last user msg)
    chat_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    with st.spinner("Thinking…"):
        response = rag_chain.invoke({
            "input": prompt,
            "chat_history": chat_history,
        })
        answer = response["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
