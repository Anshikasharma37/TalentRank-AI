"""
Why FAISS?
  - No server needed, pure numpy/CPU
  - Handles 100K vectors comfortably
  - Returns top-K by cosine similarity in milliseconds
"""

import numpy as np
import faiss
import os
import pickle


def build_index(embeddings: np.ndarray) -> faiss.Index:
    
    dim = embeddings.shape[1]

   
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def search_index(index: faiss.Index, query: np.ndarray, top_k: int = 500):
    # FAISS expects 2D input
    query_2d = query.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_2d, top_k)

    return distances[0], indices[0]


def save_index(index: faiss.Index, embeddings: np.ndarray,
               candidate_ids: list, cache_dir: str):
    """Save FAISS index + metadata to disk so we don't re-embed every run."""
    os.makedirs(cache_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(cache_dir, "faiss.index"))

    with open(os.path.join(cache_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({"candidate_ids": candidate_ids}, f)

    np.save(os.path.join(cache_dir, "embeddings.npy"), embeddings)
    print(f"  Cache saved to {cache_dir}/")


def load_index(cache_dir: str):
    
    index_path = os.path.join(cache_dir, "faiss.index")
    meta_path  = os.path.join(cache_dir, "metadata.pkl")
    emb_path   = os.path.join(cache_dir, "embeddings.npy")

    if not all(os.path.exists(p) for p in [index_path, meta_path, emb_path]):
        return None, None, None

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    embeddings = np.load(emb_path)

    print(f"  Loaded index from cache: {index.ntotal} vectors")
    return index, embeddings, meta["candidate_ids"]


if __name__ == "__main__":
    # Quick sanity test
    print("Testing vector_store...")
    N, DIM = 1000, 384

    # Fake normalized embeddings
    rng = np.random.RandomState(42)
    embeddings = rng.randn(N, DIM).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    idx = build_index(embeddings)

    query = embeddings[0].copy()   # query for itself → should be rank 1
    dists, inds = search_index(idx, query, top_k=5)

    print(f"  Top-5 for query=embeddings[0]: indices={inds}, sims={dists.round(4)}")
    assert inds[0] == 0, "Exact match should be rank 1"
    print("  Test passed!")
