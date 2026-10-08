"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend:

  chunker.chunk_files(file_paths: list[Path]) -> list[DocumentChunk]
  store.ingest(chunks: list[DocumentChunk]) -> IngestionResult
  store.list_documents() -> list[dict]
  store.get_document_chunks(source: str) -> list[DocumentChunk]
  store.query(query: str, k: int, topic_filter, difficulty_filter)
      -> list[RetrievedChunk]

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.agent.prompts import NO_CONTEXT_RESPONSE, SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager

TOPIC_OPTIONS = [
    "All topics",
    "ANN",
    "CNN",
    "RNN",
    "LSTM",
    "Seq2Seq",
    "Autoencoder",
    "SOM",
    "BoltzmannMachine",
    "GAN",
]

DIFFICULTY_OPTIONS = ["All levels", "beginner", "intermediate", "advanced"]


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_llm():
    """
    Return the singleton chat model built from LLM_PROVIDER in .env.

    Cached so the client (and its connection pool) is created once rather
    than on every message.
    """
    return LLMFactory().create()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def refresh_document_list(store: VectorStoreManager) -> None:
    """Reload the cached list of ingested documents from the vector store."""
    st.session_state["ingested_documents"] = store.list_documents()


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of Markdown files. Displays ingestion results
    (chunks added, duplicates skipped, errors) and updates the ingested
    documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["md", "pdf"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "Ingest Documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:
        with st.spinner("Chunking and embedding documents..."):
            # Uploaded files live in memory; the chunker works on paths, so
            # they are written to a temp directory for the duration of the call.
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for uploaded in uploaded_files:
                    destination = Path(temp_dir) / uploaded.name
                    destination.write_bytes(uploaded.getvalue())
                    file_paths.append(destination)

                chunks = chunker.chunk_files(file_paths)

            if not chunks:
                st.sidebar.error("No readable content found in the uploaded files.")
            else:
                result = store.ingest(chunks)
                st.session_state["last_ingestion_result"] = result
                refresh_document_list(store)

                message = (
                    f"{result.ingested} chunks added, "
                    f"{result.skipped} duplicates skipped"
                )
                if result.errors:
                    st.sidebar.error(f"{message} — {len(result.errors)} errors")
                    for error in result.errors:
                        st.sidebar.caption(error)
                elif result.ingested:
                    st.sidebar.success(message)
                else:
                    st.sidebar.warning(message)

    # Ingested documents list with per-document removal.
    documents = st.session_state.get("ingested_documents") or []
    if documents:
        st.sidebar.subheader("Ingested documents")

    for document in documents:
        row_label, row_button = st.sidebar.columns([4, 1])
        with row_label:
            st.caption(
                f"**{document['source']}** — {document['topic']} "
                f"({document['chunk_count']} chunks)"
            )
        with row_button:
            if st.button("🗑", key=f"delete-{document['source']}"):
                removed = store.delete_document(document["source"])
                refresh_document_list(store)
                st.sidebar.info(f"Removed {removed} chunks.")
                st.rerun()

    if not documents:
        st.sidebar.info("Upload .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()

    st.sidebar.divider()
    st.sidebar.metric("Total Chunks in ChromaDB", stats["total_chunks"])

    if stats["topics"]:
        st.sidebar.write("**Topics:** " + ", ".join(stats["topics"]))

    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    documents = st.session_state.get("ingested_documents") or []

    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [document["source"] for document in documents]
    selected = st.selectbox("Select document", options=sources)
    st.session_state["selected_document"] = selected

    chunks = store.get_document_chunks(selected)

    with st.container(height=400):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata
            st.caption(
                f"Chunk {index} — `{metadata.topic}` | "
                f"`{metadata.difficulty}` | `{metadata.type}`"
            )
            st.write(chunk.chunk_text)
            st.divider()

    st.caption(f"{len(chunks)} chunks from {selected}")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk carries its citation header so the model can attribute
    every claim back to a specific source.
    """
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            f"--- CONTEXT {index} "
            f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}] "
            f"(similarity {chunk.score:.2f}) ---\n"
            f"{chunk.chunk_text}"
        )
    return "\n\n".join(sections)


def answer_question(
    query: str,
    store: VectorStoreManager,
    llm,
) -> AgentResponse:
    """
    Run one retrieve-then-generate cycle and return a structured response.

    Retrieval happens first; the LLM is only called when chunks above the
    similarity threshold exist. That ordering is the hallucination guard —
    with no grounded context, the system says so instead of guessing.

    Parameters
    ----------
    query : str
        The user's question.
    store : VectorStoreManager
    llm : BaseChatModel

    Returns
    -------
    AgentResponse
        Answer text, citations, confidence, and the no-context flag.
    """
    settings = get_settings()

    chunks = store.query(
        query_text=query,
        k=settings.retrieval_k,
        topic_filter=st.session_state.get("topic_filter"),
        difficulty_filter=st.session_state.get("difficulty_filter"),
    )

    if not chunks:
        return AgentResponse(
            answer=NO_CONTEXT_RESPONSE,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=query,
        )

    context_block = build_context_block(chunks)

    user_prompt = (
        f"CONTEXT FROM THE STUDY CORPUS:\n{context_block}\n\n"
        f"STUDENT QUESTION: {query}\n\n"
        "Answer the question using only the context above. "
        "Cite each factual claim with [SOURCE: topic | filename]. "
        "If the context is insufficient, say so explicitly."
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    confidence = sum(chunk.score for chunk in chunks) / len(chunks)

    return AgentResponse(
        answer=response.content,
        sources=[
            f"{chunk.to_citation()} — similarity {chunk.score:.2f}"
            for chunk in chunks
        ],
        confidence=confidence,
        no_context_found=False,
        rewritten_query=query,
    )


def render_chat_interface(store: VectorStoreManager, llm) -> None:
    """
    Render the RAG chat interface in the right column.

    Supports multi-turn conversation, displays source citations with every
    response, and shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    store : VectorStoreManager
    llm : BaseChatModel
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic = st.selectbox("Topic filter", options=TOPIC_OPTIONS)
        st.session_state["topic_filter"] = (
            None if topic == "All topics" else topic
        )
    with col_diff:
        difficulty = st.selectbox("Difficulty filter", options=DIFFICULTY_OPTIONS)
        st.session_state["difficulty_filter"] = (
            None if difficulty == "All levels" else difficulty
        )

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    try:
                        response = answer_question(query, store, llm)
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")
                        return

                st.markdown(response.answer)

                if response.sources:
                    with st.expander("📎 Sources"):
                        for source in response.sources:
                            st.caption(source)
                    st.caption(f"Confidence: {response.confidence:.2f}")

                if response.no_context_found:
                    st.warning("⚠️ No relevant content found in corpus.")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "no_context_found": response.no_context_found,
            }
        )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, ChromaDB, and Groq"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    llm = get_llm()

    if not st.session_state["ingested_documents"]:
        refresh_document_list(store)

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(store, llm)


if __name__ == "__main__":
    main()
