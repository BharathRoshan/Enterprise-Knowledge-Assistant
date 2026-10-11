# Enterprise Knowledge Assistant (RAG)

A small, deployable Retrieval-Augmented Generation (RAG) demo for asking questions over your own documents. Built with **Gradio, LangChain, FAISS, Sentence Transformers, and Hugging Face Transformers**.

## What it does

- Accepts PDF, DOCX, TXT, and Markdown documents.
- Splits extracted text into overlapping chunks and embeds them for semantic retrieval.
- Retrieves the most relevant chunks for a question.
- Generates an answer with a text-to-text model and displays the retrieved source snippets.
- Keeps the vector index in memory for the current app session; uploaded indexes are not deserialized.

## Architecture

```text
Documents -> Text extraction -> Chunking -> Embeddings -> FAISS
                                                       |
Question -> Similarity search -> Context + question -> FLAN-T5 -> Answer + sources
```

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/BharathRoshan/Enterprise-Knowledge-Assistant.git
cd Enterprise-Knowledge-Assistant
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal. The first launch downloads the embedding and generation models, so allow time and disk space for model caching.

## Use the app

1. Upload documents you trust and have permission to process.
2. Click **Build knowledge base** and wait for the status message.
3. Ask a question in the text box.
4. Review the answer and the source snippets together.

The app currently supports text-based PDFs. Scanned PDFs need OCR before they can be searched. DOCX support is provided through `python-docx`.

## Configuration

Set these environment variables to change the models:

- `EMBED_MODEL` (default: `BAAI/bge-small-en-v1.5`)
- `GEN_MODEL` (default: `google/flan-t5-small`)

Chunk size, overlap, and retrieval count are defined near the top of `app.py`. Tune them against your own evaluation questions rather than assuming one setting works for every document collection.

## Evaluation checklist

Before using this beyond a demo, create a small set of questions with known answers and track:

- Retrieval hit rate / Recall@k for relevant source chunks.
- Answer correctness and groundedness.
- Citation/source correctness.
- Abstention quality when the documents do not contain the answer.
- Latency and memory usage.

A source snippet being displayed does not guarantee the generated answer is correct. Validate answers against the source material.

## Limitations and safety

- This is a portfolio/demo application, not a production enterprise service.
- The default model is small and may miss nuance or generate incorrect answers.
- The index is rebuilt in memory each time the knowledge base is built; there is no persistent user isolation, authentication, audit logging, or access-control layer.
- Do not upload confidential or regulated data to a public demo.
- Treat document contents as untrusted input. Prompt instructions inside a document must not override application rules.

## Repository layout

- `app.py` — Gradio UI, ingestion, retrieval, and answer generation.
- `requirements.txt` — Python dependencies.
