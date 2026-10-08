"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    #: Valid difficulty levels recognised in the <topic>_<difficulty> convention.
    DIFFICULTY_LEVELS = {"beginner", "intermediate", "advanced"}

    #: Canonical topic names keyed by their lowercase filename form.
    CANONICAL_TOPICS = {
        "ann": "ANN",
        "cnn": "CNN",
        "rnn": "RNN",
        "lstm": "LSTM",
        "seq2seq": "Seq2Seq",
        "autoencoder": "Autoencoder",
        "som": "SOM",
        "boltzmann": "BoltzmannMachine",
        "boltzmannmachine": "BoltzmannMachine",
        "gan": "GAN",
    }

    #: Topics treated as bonus material by the UI.
    BONUS_TOPICS = {"SOM", "BoltzmannMachine", "GAN"}

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"No such file: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        elif suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported types: .md, .pdf"
            )

        # A top-level Markdown header is a better topic label than a generic
        # filename, so it is used as the fallback when the filename does not
        # match a known topic.
        fallback_topic = None
        for raw in raw_chunks:
            header = raw.get("header")
            if header:
                fallback_topic = header
                break

        base_metadata = self._infer_metadata(
            file_path,
            overrides=metadata_overrides,
            fallback_topic=fallback_topic,
        )

        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            text = (raw.get("text") or "").strip()
            if not text:
                continue

            metadata = ChunkMetadata(
                topic=base_metadata.topic,
                difficulty=base_metadata.difficulty,
                type=base_metadata.type,
                source=base_metadata.source,
                related_topics=list(base_metadata.related_topics),
                is_bonus=base_metadata.is_bonus,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=VectorStoreManager.generate_chunk_id(
                        metadata.source, text
                    ),
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info("Chunked '{}' into {} chunks", file_path.name, len(chunks))
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        all_chunks: list[DocumentChunk] = []

        for file_path in file_paths:
            try:
                all_chunks.extend(
                    self.chunk_file(
                        Path(file_path),
                        metadata_overrides=metadata_overrides,
                    )
                )
            except Exception as exc:
                # One bad file must not abort the whole upload batch.
                logger.error("Skipping '{}': {}", file_path, exc)

        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pages = PyPDFLoader(str(file_path)).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[dict] = []
        for document in splitter.split_documents(pages):
            chunks.append(
                {
                    "text": document.page_content,
                    "page": document.metadata.get("page"),
                    "header": None,
                }
            )

        return chunks

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        text = file_path.read_text(encoding="utf-8")

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        sections = header_splitter.split_text(text)

        # A file with no headers at all yields no sections — fall back to the
        # whole document so plain Markdown notes still ingest.
        if not sections:
            raw_sections = [{"text": text, "header": None}]
        else:
            raw_sections = [
                {
                    "text": section.page_content,
                    "header": (
                        section.metadata.get("h1")
                        or section.metadata.get("h2")
                        or section.metadata.get("h3")
                    ),
                }
                for section in sections
            ]

        character_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[dict] = []
        for section in raw_sections:
            body = section["text"]
            if len(body) <= chunk_size:
                chunks.append(section)
                continue

            for piece in character_splitter.split_text(body):
                chunks.append({"text": piece, "header": section["header"]})

        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
        fallback_topic: str | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.
        fallback_topic : str, optional
            Topic label to use when the filename yields no known topic —
            typically the document's first Markdown header.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        stem = file_path.stem.lower()
        parts = [part for part in stem.split("_") if part]

        # Difficulty: trailing token if it names a known level.
        if parts and parts[-1] in self.DIFFICULTY_LEVELS:
            difficulty = parts[-1]
            topic_tokens = parts[:-1]
        else:
            difficulty = "intermediate"
            topic_tokens = parts

        # Topic: first token matching a canonical deep learning topic,
        # else the document header, else a tidied filename.
        topic = None
        for token in topic_tokens:
            if token in self.CANONICAL_TOPICS:
                topic = self.CANONICAL_TOPICS[token]
                break

        if topic is None:
            topic = fallback_topic or " ".join(topic_tokens).title() or "General"

        metadata = ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type="concept_explanation",
            source=file_path.name,
            related_topics=[],
            is_bonus=topic in self.BONUS_TOPICS,
        )

        for key, value in (overrides or {}).items():
            if hasattr(metadata, key) and value is not None:
                setattr(metadata, key, value)

        # Keep is_bonus consistent if topic was overridden explicitly.
        if overrides and "topic" in overrides and "is_bonus" not in overrides:
            metadata.is_bonus = metadata.topic in self.BONUS_TOPICS

        return metadata
