def test_cli_search_command():
    try:
        import faiss
        from typer.testing import CliRunner
    except ImportError:
        return
    
    from src.cli import search
    import torch
    import os
    
    # Create mock files
    torch.save(torch.randn(10, 256), 'embeddings/fabwave.pt')
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'conv.f_delta.weight': torch.randn(16, 11), 
        'conv.f_delta.bias': torch.randn(16),
        'conv.linear.weight': torch.randn(256, 32),
        'conv.linear.bias': torch.randn(256)
    }, 'checkpoints/cadgcl_model.pt')
    
    # Test function directly
    search(query=0, k=5)
    
    runner = CliRunner()
    from src.cli import app
    result = runner.invoke(app, ['search', '--query', '0'])
    assert result.exit_code == 0