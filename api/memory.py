"""
api/memory.py — Persistent vector memory store for opensense.

Uses ChromaDB with sentence-transformers embeddings to store and retrieve
facts/knowledge that the model should remember across sessions.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"  # ~80 MB, fast, good quality for facts
_COLLECTION_NAME = "opensense_memory"


class MemoryStore:
    """Persistent vector memory backed by ChromaDB."""

    def __init__(self, persist_dir: str = "memory/db", embed_model: str = _EMBED_MODEL):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._embedder = SentenceTransformer(embed_model)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        source: str = "user",
        extra: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a new fact in memory.

        Returns the unique ID of the stored entry, or the existing ID if an
        identical fact was already stored (deduplication by content hash).
        """
        doc_id = _content_id(text)

        # Avoid duplicates
        existing = self._col.get(ids=[doc_id])
        if existing["ids"]:
            return doc_id

        embedding = self._embedder.encode(text).tolist()
        metadata: dict[str, Any] = {
            "source": source,
            "stored_at": int(time.time()),
        }
        if extra:
            # ChromaDB metadata values must be str | int | float | bool
            for k, v in extra.items():
                metadata[k] = v if isinstance(v, (str, int, float, bool)) else json.dumps(v)

        self._col.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return doc_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return the top-k most semantically similar facts to *query*.

        Each result is a dict with keys: id, text, score, metadata.
        """
        if self._col.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._col.count()),
            include=["documents", "distances", "metadatas"],
        )

        hits = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "score": round(1.0 - dist, 4),  # cosine similarity
                    "metadata": meta,
                }
            )
        # Sort by descending similarity
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    def list_all(self) -> list[dict[str, Any]]:
        """Return every stored fact (no vector search)."""
        if self._col.count() == 0:
            return []
        results = self._col.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def delete(self, doc_id: str) -> bool:
        """Delete a fact by ID. Returns True if it existed."""
        existing = self._col.get(ids=[doc_id])
        if not existing["ids"]:
            return False
        self._col.delete(ids=[doc_id])
        return True

    def count(self) -> int:
        return self._col.count()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _content_id(text: str) -> str:
    """Deterministic ID based on normalised text content."""
    normalised = text.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]
