import faiss
import numpy as np

from chunking import chunks
from embeddings import create_embeddings


# Create embeddings for all chunks
embeddings = create_embeddings(chunks)

embeddings = np.array(embeddings).astype("float32")


# Create FAISS index
index = faiss.IndexFlatL2(384)

index.add(embeddings)


print("Embedding shape:", embeddings.shape)
print("Number of vectors in FAISS:", index.ntotal)