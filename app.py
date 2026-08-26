import streamlit as st
import hashlib
import os
from io import BytesIO
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from supabase import create_client

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

load_dotenv()

st.set_page_config(page_title="Document Archive", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@600;700&family=Inter:wght@400;500&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Newsreader', serif !important; letter-spacing: -0.01em; }

.doc-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    background-color: #1A2E2A;
    color: #C9A227;
    border-left: 3px solid #C9A227;
    padding: 0.3rem 0.8rem;
    border-radius: 3px;
    margin-bottom: 0.8rem;
}
.source-card {
    background-color: #1A2E2A;
    border-left: 3px solid #6B8F71;
    border-radius: 4px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #EDE6D6;
}

/* Folder-tab navigation */
[data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent;
    border-bottom: 2px solid #C9A227;
    margin-bottom: 0.5rem;
}

[data-baseweb="tab-highlight"] {
    display: none;
}

[data-baseweb="tab"] {
    background-color: #16281F;
    border: 1px solid #2A3F37;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    padding: 1rem 2rem !important;
    transition: all 0.15s ease;
}

[data-baseweb="tab"] p {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8A9D97 !important;
    margin: 0 !important;
}

[data-baseweb="tab"]:hover {
    background-color: #1A2E2A;
}
[data-baseweb="tab"]:hover p {
    color: #EDE6D6 !important;
}

[aria-selected="true"] {
    background-color: #C9A227 !important;
    border-color: #C9A227 !important;
    transform: translateY(-2px);
}
[aria-selected="true"] p {
    color: #0F1F1D !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=get_secret("GROQ_API_KEY"), model="openai/gpt-oss-20b")

@st.cache_resource
def get_supabase_client():
    return create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY"))

embeddings = get_embeddings()
llm = get_llm()
supabase = get_supabase_client()

vectorstore = SupabaseVectorStore(
    client=supabase, embedding=embeddings,
    table_name="document_chunks", query_name="match_document_chunks"
)

prompt_template = """You are a helpful assistant answering questions about a document.
Use only the following context to answer the question.
If the context does not fully answer the question, say what the context does cover, and explicitly state that the rest is not available in the provided context — do not fill in missing details from your own general knowledge, even if you know them.

Context:
{context}

Question:
{question}

Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

st.markdown("# 📖 Document Archive")
st.caption("Upload a PDF and ask questions grounded only in its content.")

if "active_hash" not in st.session_state:
    st.session_state.active_hash = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = None
if "my_documents" not in st.session_state:
    st.session_state.my_documents = []
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

def remember_document(filename, file_hash):
    if not any(d["file_hash"] == file_hash for d in st.session_state.my_documents):
        st.session_state.my_documents.append({"filename": filename, "file_hash": file_hash})

tab1, tab2 = st.tabs(["📤 Upload New", "📚 My Library"])

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = supabase.table("documents").select("*").eq("file_hash", file_hash).execute()

        if len(existing.data) > 0:
            st.info(f"'{uploaded_file.name}' was already indexed — skipping reprocessing.")
        else:
            with st.spinner("Reading and indexing your PDF..."):
                reader = PdfReader(BytesIO(file_bytes))
                text = "".join(page.extract_text() for page in reader.pages)
                text = text.replace("\x00", "")
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(text)
                metadatas = [{"file_hash": file_hash} for _ in chunks]
                vectorstore.add_texts(chunks, metadatas=metadatas)
                supabase.table("documents").insert({
                    "filename": uploaded_file.name, "file_hash": file_hash, "chunk_count": len(chunks)
                }).execute()
            st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

        remember_document(uploaded_file.name, file_hash)
        if st.button("Start chatting about this document", key="start_new"):
            st.session_state.active_hash = file_hash
            st.session_state.active_filename = uploaded_file.name

with tab2:
    if len(st.session_state.my_documents) == 0:
        st.info("You haven't uploaded anything yet this session.")
    else:
        for doc in st.session_state.my_documents:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{doc['filename']}**")
            if col2.button("Open", key=f"open_{doc['file_hash']}"):
                st.session_state.active_hash = doc["file_hash"]
                st.session_state.active_filename = doc["filename"]

if st.session_state.active_hash:
    st.divider()
    st.markdown(f'<div class="doc-badge">ACTIVE: {st.session_state.active_filename}</div>', unsafe_allow_html=True)

    hash_key = st.session_state.active_hash
    if hash_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[hash_key] = []

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8, "filter": {"file_hash": hash_key}})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    for msg in st.session_state.chat_histories[hash_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("View sources"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f'<div class="source-card">Chunk {i+1}: {src[:200]}...</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about this document...")
    if question:
        st.session_state.chat_histories[hash_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.invoke(question)
                    answer = response["result"]
                    sources = [d.page_content for d in response.get("source_documents", [])]
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        answer = "This app has hit its free-tier request limit for now — please try again in a bit."
                        sources = []
                    else:
                        raise
            st.write(answer)
            if sources:
                with st.expander("View sources"):
                    for i, src in enumerate(sources):
                        st.markdown(f'<div class="source-card">Chunk {i+1}: {src[:200]}...</div>', unsafe_allow_html=True)

        st.session_state.chat_histories[hash_key].append({"role": "assistant", "content": answer, "sources": sources})