---
title: Enterprise Knowledge Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.2"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Enterprise Knowledge Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application built with:

- Gradio
- LangChain
- FAISS
- Hugging Face Transformers
- Sentence Transformers embeddings

## Features
- Semantic search over enterprise documents
- Grounded response generation
- Source citation display
- Lightweight deployment for demo purposes

## Usage
Ask questions about the uploaded enterprise knowledge base and the app will:
1. Retrieve relevant document chunks using semantic search
2. Generate a grounded answer using a Hugging Face model
3. Show source snippets used for the answer
