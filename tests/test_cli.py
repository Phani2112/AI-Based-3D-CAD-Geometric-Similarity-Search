import torch
from src.cli import search_cmd

def test_cli_search_command():
    try:
        import faiss
        import typer
    except ImportError:
        return
    
    # Create mock files
    import os
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(torch.randn(10, 256), 'embeddings/fabwave.pt')
    torch.save({
        'conv.f_delta.weight': torch.randn(16, 11), 
        'conv.f_delta.bias': torch.randn(16),
        'conv.linear.weight': torch.randn(256, 32),
        'conv.linear.bias': torch.randn(256)
    }, 'checkpoints/cadgcl_model.pt')
    
    # Test function directly
    search_cmd(query=0, k=5)