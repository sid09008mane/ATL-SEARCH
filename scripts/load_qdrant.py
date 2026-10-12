"""Load every paragraph and its vector into Qdrant.

    python scripts/load_qdrant.py

Safe to re-run: the collection is deleted and rebuilt each time.

What gets built:
  - one point per paragraph: its 1024-number vector plus the case details (payload)
  - a full-text index on the paragraph with phrase matching -> exact phrases such as "Section 12(1)(a)"
  - a keyword index on category                              -> filtering by property / tenant
"""
import time

from qdrant_client import models

from config import QDRANT_COLLECTION, VECTOR_SIZE, connect_qdrant
from embedder import embed_documents, load_chunks

COLLECTION_NAME = QDRANT_COLLECTION
UPLOAD_BATCH_SIZE = 64
PAYLOAD_FIELDS = [
    "chunk_id", "case_id", "case_name", "court", "year", "category",
    "para_no", "part", "parts_in_para", "paragraph", "source_url",
]


def build_points(chunks, vectors):
    """One Qdrant point per paragraph: the vector plus its case details."""
    points = []
    for position, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=chunk["id"],
                vector=vectors[position].tolist(),
                payload={field: chunk[field] for field in PAYLOAD_FIELDS},
            )
        )
    return points


def recreate_collection(client, vector_size):
    """Delete any old collection so reruns never create duplicates."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def upload_points(client, points):
    for start in range(0, len(points), UPLOAD_BATCH_SIZE):
        batch = points[start:start + UPLOAD_BATCH_SIZE]
        client.upsert(COLLECTION_NAME, points=batch)


def create_payload_indexes(client):
    """Indexes that let Qdrant filter on words and phrases inside the paragraph text."""
    client.create_payload_index(
        COLLECTION_NAME,
        field_name="paragraph",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            min_token_len=1,
            phrase_matching=True,
        ),
    )
    client.create_payload_index(
        COLLECTION_NAME,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def main():
    chunks = load_chunks()
    paragraph_texts = [chunk["paragraph"] for chunk in chunks]
    vectors = embed_documents(paragraph_texts)
    if vectors.shape[1] != VECTOR_SIZE:
        raise SystemExit(f"Vectors have {vectors.shape[1]} numbers each, expected {VECTOR_SIZE}.")

    client = connect_qdrant()
    try:
        client.get_collections()
    except Exception as exc:
        raise SystemExit(f"Cannot connect to Qdrant ({exc.__class__.__name__}). Is Docker running? Try: docker compose up -d")

    started = time.perf_counter()
    recreate_collection(client, vector_size=vectors.shape[1])
    points = build_points(chunks, vectors)
    upload_points(client, points)
    uploaded = time.perf_counter()
    create_payload_indexes(client)
    indexed = time.perf_counter()

    uploaded_count = client.count(COLLECTION_NAME).count
    print(f"Uploaded {uploaded_count} points to '{COLLECTION_NAME}'")
    print(f"  upload: {uploaded - started:.2f}s   indexes: {indexed - uploaded:.2f}s")
    client.close()


if __name__ == "__main__":
    main()
