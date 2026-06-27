import torch

def test_faiss_index():
    try:
        import faiss
    except ImportError:
        return
    
    from src.search.faiss_index import build_faiss_index, search_similar
    
    embeddings = torch.randn(100, 256).numpy().astype('float32')
    index = build_faiss_index(embeddings)
    assert index.ntotal == 100
    
    query = embeddings[0:1]
    indices, scores = search_similar(index, query, k=5)
    assert len(indices) == 5