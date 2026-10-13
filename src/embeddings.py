from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    """
    Convert text into numerical embeddings.
    """

    embeddings = model.encode(texts)

    return embeddings