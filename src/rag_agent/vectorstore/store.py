"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    #: Chunks are embedded in batches of this size to bound peak memory use.
    EMBED_BATCH_SIZE = 100

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        import chromadb

        db_path = Path(self._settings.chroma_db_path)

        try:
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                # Cosine distance matches normalised sentence-transformer
                # embeddings; the default (L2) would rank differently.
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Could not initialise ChromaDB at '{db_path}': {exc}"
            ) from exc

        logger.info(
            "ChromaDB ready — collection '{}' at '{}' ({} chunks)",
            self._settings.chroma_collection_name,
            db_path,
            self._collection.count(),
        )

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        try:
            existing = self._collection.get(ids=[chunk_id])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Duplicate check failed for {}: {}", chunk_id, exc)
            return False

        return bool(existing and existing.get("ids"))

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()

        # Filter duplicates first so embedding is only paid for new content.
        pending: list[DocumentChunk] = []
        seen_ids: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_id in seen_ids or self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
                continue
            seen_ids.add(chunk.chunk_id)
            pending.append(chunk)

        for start in range(0, len(pending), self.EMBED_BATCH_SIZE):
            batch = pending[start : start + self.EMBED_BATCH_SIZE]
            texts = [chunk.chunk_text for chunk in batch]

            try:
                embeddings = self._embeddings.embed_documents(texts)

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[chunk.metadata.to_dict() for chunk in batch],
                )

                result.ingested += len(batch)
                for chunk in batch:
                    source = chunk.metadata.source
                    if source not in result.document_ids:
                        result.document_ids.append(source)

            except Exception as exc:
                message = (
                    f"Failed to ingest batch starting at index {start}: {exc}"
                )
                logger.error(message)
                result.errors.append(message)

        logger.info(
            "Ingestion complete — {} added, {} duplicates skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        if not query_text or not query_text.strip():
            return []

        k = k or self._settings.retrieval_k

        if self._collection.count() == 0:
            logger.info("Query attempted against an empty collection.")
            return []

        where_filter = self._build_where_filter(topic_filter, difficulty_filter)

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            # Cosine distance → similarity. Clamped because ChromaDB can
            # return tiny negative distances from floating point error.
            score = max(0.0, min(1.0, 1.0 - float(distance)))

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        retrieved.sort(key=lambda chunk: chunk.score, reverse=True)

        logger.info(
            "Retrieved {}/{} chunks above threshold {:.2f}",
            len(retrieved),
            len(documents),
            self._settings.similarity_threshold,
        )
        return retrieved

    @staticmethod
    def _build_where_filter(
        topic_filter: str | None,
        difficulty_filter: str | None,
    ) -> dict | None:
        """
        Build a ChromaDB where-clause from optional metadata filters.

        ChromaDB requires an explicit $and operator when more than one
        field is constrained, and None when nothing is constrained.
        """
        conditions = []
        if topic_filter:
            conditions.append({"topic": topic_filter})
        if difficulty_filter:
            conditions.append({"difficulty": difficulty_filter})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        grouped: dict[str, dict] = {}
        for metadata in metadatas:
            source = metadata.get("source", "unknown")
            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "topic": metadata.get("topic", "unknown"),
                    "difficulty": metadata.get("difficulty", "unknown"),
                    "chunk_count": 0,
                }
            grouped[source]["chunk_count"] += 1

        return sorted(grouped.values(), key=lambda doc: doc["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        return [
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=text,
                metadata=ChunkMetadata.from_dict(metadata),
            )
            for chunk_id, text, metadata in zip(ids, documents, metadatas)
        ]

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        topics = sorted({m.get("topic", "unknown") for m in metadatas})
        sources = sorted({m.get("source", "unknown") for m in metadatas})
        bonus_present = any(
            str(m.get("is_bonus", "false")).lower() == "true" for m in metadatas
        )

        return {
            "total_chunks": self._collection.count(),
            "topics": topics,
            "sources": sources,
            "bonus_topics_present": bonus_present,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        existing = self._collection.get(where={"source": source})
        count = len(existing.get("ids") or [])

        if count:
            self._collection.delete(where={"source": source})
            logger.info("Deleted {} chunks from source '{}'", count, source)

        return count
