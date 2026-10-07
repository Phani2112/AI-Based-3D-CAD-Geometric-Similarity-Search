import torch

def test_predictor(tmp_path):
    try:
        import faiss
    except ImportError:
        return
    
    from src.search.predictor import CADGCLPredictor
    
    embeddings_path = tmp_path / 'test_embeddings.pt'
    model_path = tmp_path / 'test_model.pt'
    embeddings = torch.randn(10, 256)
    torch.save(embeddings, embeddings_path)
    torch.save({
        'conv.f_delta.weight': torch.randn(16, 11), 
        'conv.f_delta.bias': torch.randn(16),
        'conv.linear.weight': torch.randn(256, 32),
        'conv.linear.bias': torch.randn(256)
    }, model_path)
    
    predictor = CADGCLPredictor(
        model_path=model_path,
        embeddings_path=embeddings_path
    )
    
    predictor.load()
    results = predictor.search(query_model_id=0, k=5)
    assert len(results) == 5


def test_predictor_search_uses_cosine_similarity(tmp_path):
    try:
        import faiss
    except ImportError:
        return

    from src.search.predictor import CADGCLPredictor

    embeddings_path = tmp_path / 'test_embeddings.pt'
    model_path = tmp_path / 'test_model.pt'
    embeddings = torch.zeros(3, 256)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 0.9
    embeddings[1, 1] = 0.1
    embeddings[2, 0] = 100.0
    embeddings[2, 1] = 100.0
    torch.save(embeddings, embeddings_path)
    torch.save({}, model_path)

    predictor = CADGCLPredictor(
        model_path=model_path,
        embeddings_path=embeddings_path
    )

    predictor.load()
    results = [r for r in predictor.search(query_model_id=0, k=3) if r != 0]
    assert results[0] == 1


def test_predictor_search_with_scores_returns_ranked_cosine_scores(tmp_path):
    try:
        import faiss
    except ImportError:
        return

    from src.search.predictor import CADGCLPredictor

    embeddings_path = tmp_path / 'test_embeddings.pt'
    model_path = tmp_path / 'test_model.pt'
    embeddings = torch.zeros(3, 256)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 0.8
    embeddings[1, 1] = 0.6
    embeddings[2, 1] = 1.0
    torch.save(embeddings, embeddings_path)
    torch.save({}, model_path)

    predictor = CADGCLPredictor(
        model_path=model_path,
        embeddings_path=embeddings_path
    )

    predictor.load()
    results = predictor.search_with_scores(query_model_id=0, k=3)

    assert results[0] == (0, 1.0)
    assert results[1] == (1, 0.800000011920929)
    assert results[2] == (2, 0.0)
