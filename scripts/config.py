"""Settings shared by every script, so the loaders, search, answer and compare always agree.

Values can be overridden with environment variables or a .env file in the project folder:
    GROQ_API_KEY             API key for the answer engine (the "brain")
    ATL_LLM_MODEL            model name (default openai/gpt-oss-20b)
    ATL_LLM_BASE_URL         any OpenAI-compatible API (default Groq)
    ATL_LLM_REASONING_EFFORT low, medium or high (default low, the fastest)
    ATL_PG_DSN               Postgres connection string
    ATL_QDRANT_URL           Qdrant server address
    ATL_QDRANT_PATH          folder for Qdrant's built-in local mode (no Docker); used instead of the URL when set
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path=ROOT / ".env"):
    """Read KEY=value lines from .env without overriding variables that are already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


_load_dotenv()

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

LLM_BASE_URL = os.getenv("ATL_LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("ATL_LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("ATL_LLM_API_KEY")
LLM_REASONING_EFFORT = os.getenv("ATL_LLM_REASONING_EFFORT", "low")


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
