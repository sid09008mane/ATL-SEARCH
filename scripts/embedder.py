import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/clean/atl_chunks.json")
VECTORS_FILE = Path("data/vectors/bge_m3.npy")
MODEL_NAME = "BAAI/bge-m3"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def pick_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

_model = None

def get_model():
    global _model
    if _model is None:
        device = pick_device()
        print(f"Loading model {MODEL_NAME} on {device}...")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def embed_documents(texts, batch_size=8):
    if VECTORS_FILE.exists():
        vectors = np.load(VECTORS_FILE)
        if len(vectors) == len(texts):
            print(f"Loaded {len(vectors)} vectors from {VECTORS_FILE}")
            return vectors
        print("Saved vectors do not match the paragraphs, embedding again")

    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    VECTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_FILE, vectors)
    print(f"Saved {vectors.shape} to {VECTORS_FILE}")
    return vectors

def embed_query(question):
    vector = get_model().encode(question, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def top_k(vectors, query_vector, k=5):
    scores = vectors @ query_vector
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best]


if __name__ == "__main__":
    chunks = load_chunks()
    texts = [c["paragraph"] for c in chunks]
    print(f"{len(texts)} paragraphs")

    vectors = embed_documents(texts)

    question = sys.argv[1] if len(sys.argv) > 1 else "tenant did not pay rent"
    print(f"\nQuestion: {question}")
    for rank, (i, score) in enumerate(top_k(vectors, embed_query(question)), 1):
        c = chunks[i]
        print(f"\n{rank}. {c['case_name']} (para {c['para_no']})  score={score:.3f}")
        print("   " + c["paragraph"][:200] + "...")