import torch

def test_predictor():
    try:
        import faiss
    except ImportError:
        return
    
    from src.search.predictor import CADGCLPredictor
    
    import os
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    embeddings = torch.randn(10, 256)
    torch.save(embeddings, 'embeddings/test.pt')
    torch.save({
        'conv.f_delta.weight': torch.randn(16, 11), 
        'conv.f_delta.bias': torch.randn(16),
        'conv.linear.weight': torch.randn(256, 32),
        'conv.linear.bias': torch.randn(256)
    }, 'checkpoints/cadgcl_model.pt')
    
    predictor = CADGCLPredictor(
        model_path='checkpoints/cadgcl_model.pt',
        embeddings_path='embeddings/test.pt'
    )
    
    predictor.load()
    results = predictor.search(query_model_id=0, k=5)
    assert len(results) == 5