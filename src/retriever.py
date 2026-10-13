import numpy as np

from faiss_store import index
from chunking import chunks
from embeddings import create_embeddings


def retrieve_documents(query, top_k=2, distance_threshold=1.0):
    """
    Retrieve relevant chunks using FAISS
    and filter out weak matches.
    """

    query_embedding = create_embeddings([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k=top_k)

    print("\nFAISS distances:", distances[0])

    retrieved_chunks = []

    for distance, index_id in zip(distances[0], indices[0]):

        if distance <= distance_threshold:
            retrieved_chunks.append(chunks[index_id])

    return retrieved_chunks