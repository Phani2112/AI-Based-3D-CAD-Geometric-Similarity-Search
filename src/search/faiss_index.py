import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

def build_faiss_index(embeddings):
    if faiss is None:
        return None
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.detach().cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_similar(index, query_embedding, k=5):
    if index is None:
        return [], []
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.detach().cpu().numpy().astype('float32')
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    scores, indices = index.search(query_embedding, k)
    return indices[0], scores[0]