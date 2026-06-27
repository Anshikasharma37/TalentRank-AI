
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMBEDDING_MODEL

_model = None   # initialized here so global reference in get_model() works

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of text strings.
   
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalized → dot product = cosine sim
    )
    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Embed a single string. """
    return embed_texts([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Since we normalize during embedding, this is just dot product.
    """
    return float(np.dot(a, b))


def build_jd_embedding(jd: dict) -> np.ndarray:
    """
    Build a single JD embedding by averaging weighted section embeddings.

    Weights:
      requirements -> 40% (must-haves)
      ideal_profile -> 35% (our hand-crafted signal-rich description)
      responsibilities -> 25% (what the role does)
    """
    from src.jd_parser import get_jd_embedding_text

    
    jd_text = get_jd_embedding_text(jd)
    return embed_single(jd_text)


if __name__ == "__main__":
    print("Testing embedder...")
    texts = [
        "Machine learning engineer with experience in retrieval systems and vector databases.",
        "Marketing manager with AI keywords in skills section but no production ML experience.",
    ]
    embeddings = embed_texts(texts)
    print(f"Embedding shape: {embeddings.shape}")

    jd_query = embed_single(
        "Senior AI engineer production embeddings retrieval ranking recommendation system"
    )
    for i, text in enumerate(texts):
        sim = cosine_similarity(embeddings[i], jd_query)
        print(f"  Similarity {i+1}: {sim:.4f} — {text[:60]}")
