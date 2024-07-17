from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Initialize the model
text_similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


def text_similarity(
    text1: str,
    text2: str,
) -> float:
    """Calculate the text similarity (0 to 100)
    Use ChatGPT's embedding to calculate the similarity
    """

    # If the texts are the same, return 1
    if text1 == text2:
        return 100.0

    # Compute the embeddings
    text1_embedding = text_similarity_model.encode([text1])[0]
    text2_embedding = text_similarity_model.encode([text2])[0]

    # Compute the cosine similarity
    similarity = cosine(text1_embedding, text2_embedding)

    return similarity * 100
