"""Settings shared by every script, so the loaders, search and compare always agree.

Override the database addresses with environment variables if needed:
    ATL_PG_DSN       Postgres connection string
    ATL_QDRANT_URL   Qdrant server address
    ATL_QDRANT_PATH  folder for Qdrant's built-in local mode (no Docker); used instead of the URL when set
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = ROOT / "data" / "clean" / "atl_chunks.json"
VECTORS_FILE = ROOT / "data" / "vectors" / "bge_m3.npy"
QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_FILE = ROOT / "eval" / "results.csv"

MODEL_NAME = "BAAI/bge-m3"
VECTOR_SIZE = 1024

PG_DSN = os.getenv("ATL_PG_DSN", "postgresql://postgres:atl@localhost:5432/atl")
PG_TABLE = "chunks"

QDRANT_URL = os.getenv("ATL_QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("ATL_QDRANT_PATH")
QDRANT_COLLECTION = "atl_chunks"


def connect_postgres():
    """Open a Postgres connection that can send and receive pgvector values."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def connect_qdrant():
    """Connect to the Qdrant server, or to local mode when ATL_QDRANT_PATH is set."""
    from qdrant_client import QdrantClient

    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(url=QDRANT_URL, timeout=30)
