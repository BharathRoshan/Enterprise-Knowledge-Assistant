import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


# -----------------------------
# Load Embeddings + FAISS Index
# -----------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_PATH = "faiss_index"
GEN_MODEL = "google/flan-t5-small"   # free-friendly for Spaces


def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )


print("Loading embedding model...")
embedding_model = load_embedding_model()

print("Loading FAISS index...")
vector_store = FAISS.load_local(
    FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 6}
)

print("Loading generation model...")
generator = pipeline(
    "text-generation",
    model=GEN_MODEL
)

print("App ready!")


# -----------------------------
# Helper Functions
# -----------------------------
def retrieve_docs(query):
    try:
        return retriever.invoke(query)
    except AttributeError:
        return retriever.get_relevant_documents(query)


def format_context(docs):
    context_parts = []
    for i, doc in enumerate(docs[:2], start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:350]
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{snippet}"
        )
    return "\n\n".join(context_parts)


def ask_rag(query):
    if not query.strip():
        return "Please enter a question.", "No sources found."

    retrieved_docs = retrieve_docs(query)
    context = format_context(retrieved_docs)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer ONLY using the provided context.
If the answer is not found in the context, say:
I could not find the answer in the provided enterprise knowledge base.

Be concise and business-friendly.

Context:
{context}

Question:
{query}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    answer = response[0]["generated_text"].strip()

    sources = []
    for i, doc in enumerate(retrieved_docs[:2], start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        preview = doc.page_content[:250].replace("\n", " ")
        sources.append(f"**Source {i}:** {source} (Page: {page})\n\n{preview}...")

    return answer, "\n\n---\n\n".join(sources)


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Enterprise Knowledge Assistant (RAG)") as demo:
    gr.Markdown(
        """
        # 📚 Enterprise Knowledge Assistant (RAG)
        ### Ask questions across enterprise documents and get grounded answers with source snippets.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                lines=4,
                placeholder="Ask a question about your enterprise knowledge base...\n\nExample: Based on the uploaded documents, how should an AML analyst investigate a suspicious transaction alert?",
                label="Your Question"
            )

            with gr.Row():
                ask_btn = gr.Button("🔍 Ask", variant="primary")
                clear_btn = gr.Button("🧹 Clear")

            gr.Examples(
                examples=[
                    ["Summarize this knowledge base in 2 lines."],
                    ["Based only on the uploaded documents, how should an AML analyst investigate a suspicious transaction alert?"],
                    ["What does the knowledge base say about customer due diligence (CDD)?"],
                    ["According to the uploaded documents, when should a suspicious case be escalated?"]
                ],
                inputs=query_input,
                label="Try these example questions"
            )

        with gr.Column(scale=2):
            gr.Markdown(
                """
                ### 💡 Tips
                - Ask document-specific questions for best results  
                - The app uses semantic retrieval + grounded generation  
                - Source snippets are shown for transparency  
                """
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="📌 Answer",
                lines=12,
                show_copy_button=True
            )

        with gr.Column(scale=3):
            sources_output = gr.Markdown(
                label="📄 Retrieved Sources"
            )

    ask_btn.click(
        fn=ask_rag,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )

    query_input.submit(
        fn=ask_rag,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[query_input, answer_output, sources_output]
    )

    gr.Markdown(
        """
        ---
        Built with **Gradio + LangChain + FAISS + Hugging Face Transformers**
        """
    )

if __name__ == "__main__":
    demo.launch(share = True)