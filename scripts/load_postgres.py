"""Load every paragraph and its vector into Postgres + pgvector.

    python scripts/load_postgres.py

Safe to re-run: the table is dropped and rebuilt each time.

What gets built:
  - one row per paragraph, with the case details and its 1024-number vector
  - an HNSW index on the vectors        -> meaning search
  - a full-text (tsvector) index        -> ranked keyword search
  - a trigram index on the paragraph    -> fast exact-phrase search such as "Section 12(1)(a)"
"""
import time

from config import PG_TABLE, VECTOR_SIZE, connect_postgres
from embedder import embed_documents, load_chunks

CREATE_TABLE = f"""
CREATE TABLE {PG_TABLE} (
    id          integer PRIMARY KEY,
    chunk_id    text    NOT NULL UNIQUE,
    case_id     text    NOT NULL,
    case_name   text    NOT NULL,
    court       text,
    year        integer,
    category    text,
    para_no     integer,
    part        integer,
    parts_in_para integer,
    paragraph   text    NOT NULL,
    source_url  text,
    embedding   vector({VECTOR_SIZE}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', paragraph)) STORED
)
"""

INDEXES = [
    f"CREATE INDEX {PG_TABLE}_embedding_idx ON {PG_TABLE} USING hnsw (embedding vector_cosine_ops)",
    f"CREATE INDEX {PG_TABLE}_tsv_idx ON {PG_TABLE} USING gin (tsv)",
    f"CREATE INDEX {PG_TABLE}_paragraph_trgm_idx ON {PG_TABLE} USING gin (paragraph gin_trgm_ops)",
    f"CREATE INDEX {PG_TABLE}_category_idx ON {PG_TABLE} (category)",
]

INSERT = f"""
INSERT INTO {PG_TABLE}
    (id, chunk_id, case_id, case_name, court, year, category, para_no, part, parts_in_para, paragraph, source_url, embedding)
VALUES
    (%(id)s, %(chunk_id)s, %(case_id)s, %(case_name)s, %(court)s, %(year)s, %(category)s,
     %(para_no)s, %(part)s, %(parts_in_para)s, %(paragraph)s, %(source_url)s, %(embedding)s)
"""


def recreate_table(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    conn.execute(f"DROP TABLE IF EXISTS {PG_TABLE}")
    conn.execute(CREATE_TABLE)


def insert_rows(conn, chunks, vectors):
    rows = [
        {
            "id": chunk["id"],
            "chunk_id": chunk["chunk_id"],
            "case_id": chunk["case_id"],
            "case_name": chunk["case_name"],
            "court": chunk["court"],
            "year": chunk["year"],
            "category": chunk["category"],
            "para_no": chunk["para_no"],
            "part": chunk["part"],
            "parts_in_para": chunk["parts_in_para"],
            "paragraph": chunk["paragraph"],
            "source_url": chunk["source_url"],
            "embedding": vectors[position],
        }
        for position, chunk in enumerate(chunks)
    ]
    with conn.cursor() as cur:
        cur.executemany(INSERT, rows)


def create_indexes(conn):
    for statement in INDEXES:
        conn.execute(statement)
    conn.execute(f"ANALYZE {PG_TABLE}")


def main():
    chunks = load_chunks()
    vectors = embed_documents([chunk["paragraph"] for chunk in chunks])
    if vectors.shape[1] != VECTOR_SIZE:
        raise SystemExit(f"Vectors have {vectors.shape[1]} numbers each, expected {VECTOR_SIZE}.")

    try:
        conn = connect_postgres()
    except Exception as exc:
        raise SystemExit(f"Cannot connect to Postgres ({exc.__class__.__name__}). Is Docker running? Try: docker compose up -d")

    started = time.perf_counter()
    recreate_table(conn)
    insert_rows(conn, chunks, vectors)
    inserted = time.perf_counter()
    create_indexes(conn)
    indexed = time.perf_counter()

    count = conn.execute(f"SELECT count(*) FROM {PG_TABLE}").fetchone()[0]
    size = conn.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{PG_TABLE}'))").fetchone()[0]
    print(f"Loaded {count} rows into Postgres table '{PG_TABLE}' ({size} with indexes)")
    print(f"  insert: {inserted - started:.2f}s   indexes: {indexed - inserted:.2f}s")
    conn.close()


if __name__ == "__main__":
    main()
