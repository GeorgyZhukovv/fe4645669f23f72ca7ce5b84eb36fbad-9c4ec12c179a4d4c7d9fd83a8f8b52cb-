"""Semantic (long-term) memory via chromadb with a JSON-fallback store."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent.types import Memory


@dataclass
class _LocalRecord:
    """Internal record used by the simple TF-IDF fallback."""

    id: str
    kind: str
    content: str
    metadata: dict[str, Any]
    vector: dict[str, float]
    ts: float


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + punctuation tokenizer used by the fallback store."""
    import re

    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)]


def _tfidf(text: str, df: dict[str, int], n_docs: int) -> dict[str, float]:
    """Return a sparse TF-IDF vector keyed by token."""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    out: dict[str, float] = {}
    for t, c in tf.items():
        idf = math.log((1 + n_docs) / (1 + df.get(t, 0))) + 1.0
        out[t] = (c / len(tokens)) * idf
    return out


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Sparse cosine similarity."""
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class _FallbackStore:
    """Pure-Python JSON-backed semantic store used when chromadb is unavailable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            data = json.loads(self.path.read_text())
        else:
            data = {"docs": [], "df": {}}
        self.docs = [
            _LocalRecord(
                id=d["id"], kind=d["kind"], content=d["content"],
                metadata=d.get("metadata", {}), vector=d.get("vector", {}), ts=d.get("ts", 0.0),
            )
            for d in data["docs"]
        ]
        self.df: dict[str, int] = data.get("df", {})

    def _persist(self) -> None:
        data = {
            "docs": [
                {
                    "id": d.id, "kind": d.kind, "content": d.content,
                    "metadata": d.metadata, "vector": d.vector, "ts": d.ts,
                }
                for d in self.docs
            ],
            "df": self.df,
        }
        self.path.write_text(json.dumps(data))

    def add(self, kind: str, content: str, metadata: dict[str, Any]) -> str:
        for t in set(_tokenize(content)):
            self.df[t] = self.df.get(t, 0) + 1
        vector = _tfidf(content, self.df, len(self.docs) + 1)
        rec_id = hashlib.sha1(f"{kind}:{content}:{time.time_ns()}".encode()).hexdigest()
        self.docs.append(_LocalRecord(rec_id, kind, content, metadata, vector, time.time()))
        self._persist()
        return rec_id

    def search(self, query: str, top_k: int) -> list[Memory]:
        if not self.docs:
            return []
        qv = _tfidf(query, self.df, len(self.docs))
        scored = [(d, _cosine(qv, d.vector)) for d in self.docs]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        out: list[Memory] = []
        for d, s in scored[:top_k]:
            if s <= 0:
                continue
            out.append(
                Memory(id=d.id, kind=d.kind, content=d.content, metadata=d.metadata, score=s)
            )
        return out


class SemanticMemory:
    """Long-term semantic store; chromadb-backed if installed, else local TF-IDF."""

    def __init__(self, dir_path: Path | str, collection: str = "agent_default") -> None:
        """Open the semantic store at ``dir_path`` for the given collection."""
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self._fallback: _FallbackStore | None = None
        self._chroma_collection = None
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(self.dir / "chroma"),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._chroma_collection = client.get_or_create_collection(name=collection)
        except Exception:
            self._fallback = _FallbackStore(self.dir / f"{collection}.json")

    @property
    def backend(self) -> str:
        return "chroma" if self._chroma_collection is not None else "fallback"

    def add(
        self,
        kind: Literal["file", "decision", "error", "summary", "fact"],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a new memory and return its id."""
        meta = dict(metadata or {})
        meta["kind"] = kind
        meta.setdefault("ts", time.time())
        if self._chroma_collection is not None:
            rec_id = str(uuid.uuid4())
            self._chroma_collection.add(ids=[rec_id], documents=[content], metadatas=[meta])
            return rec_id
        assert self._fallback is not None
        return self._fallback.add(kind, content, meta)

    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> list[Memory]:
        """Return the top-k most relevant memories for ``query``."""
        if self._chroma_collection is not None:
            try:
                res = self._chroma_collection.query(query_texts=[query], n_results=top_k)
            except Exception:
                return []
            out: list[Memory] = []
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0] if "distances" in res else [0.0] * len(ids)
            for i, doc_id in enumerate(ids):
                m = metas[i] or {}
                out.append(
                    Memory(
                        id=doc_id,
                        kind=m.get("kind", "fact"),
                        content=docs[i],
                        metadata=m,
                        score=1.0 - distances[i] if distances else 0.0,
                    )
                )
            return out
        assert self._fallback is not None
        return self._fallback.search(query, top_k)
