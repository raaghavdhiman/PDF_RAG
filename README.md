# PDF Q&A Chatbot (RAG)

Ask questions about any PDF and get answers grounded only in that document's actual content — not the model's general knowledge. Built as a hands-on exploration of Retrieval-Augmented Generation (RAG): chunking, embeddings, vector search, and grounded generation, wired into a persistent, multi-document Streamlit app.

## How it works

1. **Upload** — a PDF is read, split into overlapping ~500-character chunks, and each chunk is embedded locally using a HuggingFace sentence-transformer (`all-MiniLM-L6-v2`, 384-dimensional vectors)
2. **Dedup** — the file's raw bytes are hashed (SHA-256) before processing; if that exact file has already been indexed, reprocessing is skipped entirely
3. **Store** — chunks and their embeddings are stored in **Supabase Postgres with the `pgvector` extension**, so indexed documents persist across app restarts, not just within one session
4. **Retrieve** — a question is embedded and matched against only that document's chunks (filtered by file hash, so multiple stored PDFs never bleed into each other's answers) via a custom `match_document_chunks` Postgres function
5. **Generate** — retrieved chunks + the question are passed to a Groq-hosted LLM (`openai/gpt-oss-20b`) through a strict prompt template that explicitly forbids answering from the model's own general knowledge — if the context doesn't cover it, the model says so

## Why this is harder than it looks

- **Data leakage discipline**: chunks are tagged and filtered by document hash at retrieval time — without this, a shared vector store serving multiple PDFs could silently answer questions using the *wrong* document's content
- **Grounding vs. hallucination**: RAG doesn't automatically stop a model from blending in outside knowledge. The prompt explicitly forbids filling gaps with the model's own training data, even when it "knows" the answer (tested directly — the app correctly refuses to answer general-knowledge questions unrelated to the uploaded document)
- **Broad vs. narrow questions**: similarity search retrieves a fixed number of chunks, which works well for specific factual questions but can miss the full picture on broad, enumerative ones (e.g. "list every unit in this document") — a known, real limitation of vector retrieval, not something a prompt can fully paper over
- **Privacy by design**: while chunk *storage* is shared and deduplicated for efficiency, each visitor's document *browsing history* is session-scoped — no one can see what anyone else has uploaded

## Tech stack

Python, Streamlit, LangChain (`langchain-classic`, `langchain-huggingface`, `langchain-groq`, `langchain-text-splitters`), Supabase (Postgres + pgvector), Groq API, HuggingFace `sentence-transformers`, `pypdf`

## Running locally

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_service_role_key
```

You'll also need a Supabase project with the `pgvector` extension enabled and a `documents` / `document_chunks` table pair (see `PDF_project.ipynb` for the exact schema and setup SQL).

```bash
streamlit run app.py
```

## Project structure

```
├── app.py                  # Streamlit app — upload, chat, retrieval
├── PDF_project.ipynb       # development notebook — pipeline built and tested step by step
├── .streamlit/config.toml  # app theming
└── requirements.txt
```
