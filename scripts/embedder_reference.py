"""Embed ATL judgment paragraphs once, cache them, and reuse them everywhere.

Usage (from the atl-search folder):
    python scripts/embedder.py                      # embed data/clean/atl_chunks.json
    python scripts/embedder.py --query "tenant did not pay rent"
    python scripts/embedder.py --model Qwen/Qwen3-Embedding-0.6B

Other scripts import it:
    from embedder import Embedder, load_chunks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger("atl.embed")


@dataclass(frozen=True)
class Config:
    model_name: str = "BAAI/bge-m3"
    chunks_file: Path = Path("data/clean/atl_chunks.json")
    cache_dir: Path = Path("data/vectors")
    max_seq_length: int = 1024
    batch_size: int | None = None          # None = pick from the device
    query_prompt_name: str | None = None   # e.g. "query" for Qwen3-Embedding

    @property
    def cache_file(self) -> Path:
        return self.cache_dir / f"{self.model_name.replace('/', '__')}.npz"


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def text_hash(model_name: str, text: str) -> str:
    return hashlib.blake2b(f"{model_name}\x00{text}".encode(), digest_size=12).hexdigest()


class Embedder:
    """Wraps one embedding model with a content-addressed cache on disk."""

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer
        device = pick_device()
        model = SentenceTransformer(self.cfg.model_name, device=device)
        model.max_seq_length = self.cfg.max_seq_length
        if device == "cuda":
            model.half()
        log.info("model=%s device=%s dim=%d", self.cfg.model_name, device, self.dim_of(model))
        return model

    @staticmethod
    def dim_of(model) -> int:
        get = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return get()

    @property
    def batch_size(self) -> int:
        if self.cfg.batch_size:
            return self.cfg.batch_size
        return {"cuda": 64, "mps": 16, "cpu": 8}[str(self.model.device).split(":")[0]]

    def _encode(self, texts: list[str], **kw) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 32,
            **kw,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cfg.cache_file.exists():
            return {}
        with np.load(self.cfg.cache_file) as z:
            return dict(zip(z["hashes"].tolist(), z["vectors"]))

    def _save_cache(self, cache: dict[str, np.ndarray]) -> None:
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.cache_file.with_suffix(".tmp.npz")
        np.savez(tmp, hashes=np.array(list(cache)), vectors=np.stack(list(cache.values())))
        tmp.replace(self.cfg.cache_file)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return one vector per text, in order. Only new or changed texts hit the model."""
        hashes = [text_hash(self.cfg.model_name, t) for t in texts]
        cache = self._load_cache()
        todo = {h: t for h, t in zip(hashes, texts) if h not in cache}
        if todo:
            t0 = time.perf_counter()
            new = self._encode(list(todo.values()))
            cache.update(zip(todo, new))
            self._save_cache({h: cache[h] for h in dict.fromkeys(hashes)})
            log.info("embedded %d new, reused %d, %.1fs", len(todo), len(set(hashes)) - len(todo),
                     time.perf_counter() - t0)
        else:
            log.info("all %d paragraphs served from cache", len(texts))
        return np.stack([cache[h] for h in hashes])

    @lru_cache(maxsize=1024)
    def embed_query(self, question: str) -> np.ndarray:
        kw = {"prompt_name": self.cfg.query_prompt_name} if self.cfg.query_prompt_name else {}
        return self._encode([question], **kw)[0]


def top_k(vectors: np.ndarray, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Exact cosine search (vectors are normalized, so a dot product is enough)."""
    scores = vectors @ query
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--chunks", type=Path, default=Config.chunks_file)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--query-prompt", help='prompt name for questions, e.g. "query"')
    ap.add_argument("--query", help="run one test search after embedding")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    cfg = Config(model_name=args.model, chunks_file=args.chunks, batch_size=args.batch_size,
                 query_prompt_name=args.query_prompt)
    chunks = load_chunks(cfg.chunks_file)
    emb = Embedder(cfg)
    vectors = emb.embed_documents([c["paragraph"] for c in chunks])
    log.info("vectors %s -> %s", vectors.shape, cfg.cache_file)

    if args.query:
        for rank, (i, score) in enumerate(top_k(vectors, emb.embed_query(args.query), args.k), 1):
            c = chunks[i]
            print(f"\n{rank}. {c['case_name']} (para {c['para_no']})  {score:.3f}\n   {c['paragraph'][:220]}...")


if __name__ == "__main__":
    main()
