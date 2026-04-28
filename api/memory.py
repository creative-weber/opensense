"""
api/memory.py — Persistent vector memory store for opensense.

Two-tier memory architecture that mirrors human long-term memory:

  Warm (active) memory — ChromaDB collection for recently-accessed facts.
    Scores are decay-weighted by recency: things accessed recently score
    higher than things not touched in weeks.

  Cold (dumped) memory — A second ChromaDB collection for facts that have
    not been accessed for ``dump_after_days`` (default 180 days / 6 months).
    Cold retrieval carries a configurable score penalty AND the caller must
    inject an async delay to simulate the slower recall humans experience
    when recognising something they haven't seen in months.  When a cold
    memory is successfully recalled it is automatically restored to warm
    storage ("re-learning through recognition").

Recency decay formula (warm tier):
    recency_factor = 2 ^ (−age_days / half_life_days)
    adjusted_score = semantic_score × recency_factor

  half_life_days = 30  →  a fact last seen 30 days ago scores at 50 % of its
  raw semantic similarity; one seen 3 days ago scores at ~93 %.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"  # ~80 MB, fast, good quality for facts
_COLLECTION_NAME = "opensense_memory"
_COLD_COLLECTION_NAME = "opensense_memory_cold"
_SECONDS_IN_DAY = 86_400

# Defaults (overridden by constructor args which come from config.yaml)
_DEFAULT_DUMP_AFTER_DAYS = 180      # move to cold after 6 months of no access
_DEFAULT_HALF_LIFE_DAYS = 30        # score halved every 30 days without access
_DEFAULT_COLD_SCORE_PENALTY = 0.6   # multiply cold semantic score by this factor


class MemoryStore:
    """
    Persistent two-tier vector memory backed by ChromaDB.

    Parameters
    ----------
    persist_dir:
        Root directory for ChromaDB storage.  Two sub-clients are created:
        ``<persist_dir>/warm`` and ``<persist_dir>/cold``.
    embed_model:
        SentenceTransformer model name for embedding.
    dump_after_days:
        Facts not accessed for this many days are moved to cold storage.
    recency_half_life_days:
        Warm score halves every this many days since last access.
    cold_score_penalty:
        Multiplier applied to semantic scores from cold storage (0 < x ≤ 1).
    """

    def __init__(
        self,
        persist_dir: str = "memory/db",
        embed_model: str = _EMBED_MODEL,
        dump_after_days: int = _DEFAULT_DUMP_AFTER_DAYS,
        recency_half_life_days: int = _DEFAULT_HALF_LIFE_DAYS,
        cold_score_penalty: float = _DEFAULT_COLD_SCORE_PENALTY,
    ):
        warm_dir = str(Path(persist_dir) / "warm")
        cold_dir = str(Path(persist_dir) / "cold")
        Path(warm_dir).mkdir(parents=True, exist_ok=True)
        Path(cold_dir).mkdir(parents=True, exist_ok=True)

        self._embedder = SentenceTransformer(embed_model)

        self._warm_client = chromadb.PersistentClient(
            path=warm_dir, settings=Settings(anonymized_telemetry=False)
        )
        self._cold_client = chromadb.PersistentClient(
            path=cold_dir, settings=Settings(anonymized_telemetry=False)
        )

        self._col = self._warm_client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._cold = self._cold_client.get_or_create_collection(
            name=_COLD_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        self._dump_after_seconds = dump_after_days * _SECONDS_IN_DAY
        self._half_life_seconds = recency_half_life_days * _SECONDS_IN_DAY
        self._cold_score_penalty = cold_score_penalty

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
        Store a new fact in warm memory.

        Returns the unique ID of the stored entry, or the existing ID if an
        identical fact was already stored (deduplication by content hash).
        """
        doc_id = _content_id(text)

        # Avoid duplicates in warm
        if self._col.get(ids=[doc_id])["ids"]:
            return doc_id
        # Also skip if already sitting in cold storage
        if self._cold.get(ids=[doc_id])["ids"]:
            return doc_id

        now = int(time.time())
        embedding = self._embedder.encode(text).tolist()
        metadata: dict[str, Any] = {
            "source": source,
            "stored_at": now,
            "last_accessed_at": now,  # initialise access clock
            "access_count": 1,
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
    # Read — warm tier
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search warm memory.  Scores are recency-adjusted:
            adjusted_score = semantic_score × recency_factor

        Each returned dict has keys:
            id, text, score, semantic_score, recency_factor, source, metadata
        ``source`` is always ``"warm"`` for results from this method.

        ``last_accessed_at`` is bumped for every returned hit so that
        frequently-used facts stay in warm storage longer.
        """
        if self._col.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._col.count()),
            include=["documents", "distances", "metadatas"],
        )

        now = int(time.time())
        hits: list[dict[str, Any]] = []

        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            semantic_score = round(1.0 - dist, 4)
            last_accessed = int(meta.get("last_accessed_at", meta.get("stored_at", now)))
            rf = self._recency_factor(last_accessed, now)
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "score": round(semantic_score * rf, 4),
                    "semantic_score": semantic_score,
                    "recency_factor": round(rf, 4),
                    "source": "warm",
                    "metadata": meta,
                }
            )
            # Touch access timestamp so this fact ages from now, not from last use
            self._touch(doc_id, meta, now)

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Read — cold tier
    # ------------------------------------------------------------------

    def search_cold(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search cold (dumped) memory.

        Scores carry ``cold_score_penalty`` as a multiplier to reflect that
        old, unused memories are harder to surface confidently.  The caller
        is responsible for injecting an async delay before calling this
        method to simulate the real-world latency of slow recall.

        ``source`` is ``"cold"`` for all results.
        """
        if self._cold.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()
        results = self._cold.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._cold.count()),
            include=["documents", "distances", "metadatas"],
        )

        hits: list[dict[str, Any]] = []
        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            semantic_score = round(1.0 - dist, 4)
            adjusted = round(semantic_score * self._cold_score_penalty, 4)
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "score": adjusted,
                    "semantic_score": semantic_score,
                    "recency_factor": self._cold_score_penalty,
                    "source": "cold",
                    "metadata": meta,
                }
            )

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Dump / restore
    # ------------------------------------------------------------------

    def dump_stale(self) -> int:
        """
        Move all warm memories not accessed for ``dump_after_days`` to the
        cold collection.

        Returns the number of facts moved.
        """
        if self._col.count() == 0:
            return 0

        now = int(time.time())
        cutoff = now - self._dump_after_seconds

        all_warm = self._col.get(include=["documents", "metadatas", "embeddings"])
        stale_ids: list[str] = []

        for doc_id, doc, meta, emb in zip(
            all_warm["ids"],
            all_warm["documents"],
            all_warm["metadatas"],
            all_warm["embeddings"],
        ):
            last_accessed = int(meta.get("last_accessed_at", meta.get("stored_at", now)))
            if last_accessed < cutoff:
                # Skip if already in cold (shouldn't happen, but be defensive)
                if not self._cold.get(ids=[doc_id])["ids"]:
                    cold_meta = dict(meta)
                    cold_meta["dumped_at"] = now
                    self._cold.add(
                        ids=[doc_id],
                        documents=[doc],
                        embeddings=[emb],
                        metadatas=[cold_meta],
                    )
                stale_ids.append(doc_id)

        if stale_ids:
            self._col.delete(ids=stale_ids)

        return len(stale_ids)

    def restore_from_cold(self, doc_id: str) -> bool:
        """
        Promote a cold memory back to warm storage.

        Called automatically when a cold memory is successfully recalled
        during a chat — mirrors how humans re-learn forgotten things when
        they encounter them again.

        Returns True if the memory was found and restored.
        """
        result = self._cold.get(
            ids=[doc_id], include=["documents", "metadatas", "embeddings"]
        )
        if not result["ids"]:
            return False

        doc = result["documents"][0]
        emb = result["embeddings"][0]
        now = int(time.time())
        meta = dict(result["metadatas"][0])
        meta.pop("dumped_at", None)
        meta["last_accessed_at"] = now
        meta["restored_at"] = now
        meta["access_count"] = int(meta.get("access_count", 1)) + 1

        if not self._col.get(ids=[doc_id])["ids"]:
            self._col.add(
                ids=[doc_id],
                documents=[doc],
                embeddings=[emb],
                metadatas=[meta],
            )
        self._cold.delete(ids=[doc_id])
        return True

    # ------------------------------------------------------------------
    # Listing / management
    # ------------------------------------------------------------------

    def list_all(self) -> list[dict[str, Any]]:
        """Return every fact in warm storage (no vector search)."""
        if self._col.count() == 0:
            return []
        results = self._col.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def list_cold(self) -> list[dict[str, Any]]:
        """Return every fact currently in cold (dumped) storage."""
        if self._cold.count() == 0:
            return []
        results = self._cold.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def delete(self, doc_id: str) -> bool:
        """Delete a fact by ID from either tier. Returns True if it existed."""
        found = False
        if self._col.get(ids=[doc_id])["ids"]:
            self._col.delete(ids=[doc_id])
            found = True
        if self._cold.get(ids=[doc_id])["ids"]:
            self._cold.delete(ids=[doc_id])
            found = True
        return found

    def count(self) -> int:
        """Number of facts in warm storage."""
        return self._col.count()

    def count_cold(self) -> int:
        """Number of facts in cold storage."""
        return self._cold.count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recency_factor(self, last_accessed_at: int, now: int) -> float:
        """
        Exponential decay weight based on time elapsed since last access.

        Returns a value in (0, 1]:
            factor = 2 ^ (−age_seconds / half_life_seconds)

        A fact accessed 0 s ago → 1.0
        A fact accessed half_life seconds ago → 0.5
        A fact accessed 2×half_life seconds ago → 0.25
        """
        age = max(0, now - last_accessed_at)
        return math.pow(2.0, -age / self._half_life_seconds)

    def _touch(self, doc_id: str, current_meta: dict[str, Any], now: int) -> None:
        """Bump last_accessed_at and access_count without re-embedding."""
        updated = dict(current_meta)
        updated["last_accessed_at"] = now
        updated["access_count"] = int(updated.get("access_count", 1)) + 1
        self._col.update(ids=[doc_id], metadatas=[updated])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _content_id(text: str) -> str:
    """Deterministic ID based on normalised text content."""
    normalised = text.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]
