def test_ui_app():
    import torch
    from src.ui.app import CADGCLUI

    ui = CADGCLUI(datasets=['fabwave'])
    assert ui.datasets == ['fabwave']

    # Mock embeddings for testing
    ui.embeddings['fabwave'] = torch.randn(100, 256)  # 100 models, 256-dim embeddings
    results = ui.search(query_model_id=0)
    assert len(results) == 5
    assert all(0 <= r < 100 for r in results)  # Valid model indices