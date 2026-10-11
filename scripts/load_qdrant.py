from qdrant_client import QdrantClient, models

from embedder_reference import Config, Embedder, load_chunks

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "atl_chunks"
UPLOAD_BATCH_SIZE = 64


def build_points(chunks, vectors):
    """One Qdrant point per paragraph: the vector plus its case details."""
    points = []
    for position, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=position,
                vector=vectors[position].tolist(),
                payload={
                    "case_name": chunk["case_name"],
                    "para_no": chunk["para_no"],
                    "paragraph": chunk["paragraph"],
                },
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


def main():
    config = Config()
    chunks = load_chunks(config.chunks_file)
    paragraph_texts = [chunk["paragraph"] for chunk in chunks]

    embedder = Embedder(config)
    vectors = embedder.embed_documents(paragraph_texts)

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    recreate_collection(client, vector_size=vectors.shape[1])

    points = build_points(chunks, vectors)
    upload_points(client, points)

    uploaded_count = client.count(COLLECTION_NAME).count
    print(f"Uploaded {uploaded_count} points to '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()