import torch
from typer.testing import CliRunner
from src.cli import app

def test_cli_search_command():
    try:
        import faiss
        import typer
    except ImportError:
        return
    
    class FakePredictor:
        def __init__(self, model_path, embeddings_path):
            pass

        def load(self):
            pass

        def search(self, query, k):
            return [1, 2, 3, 4, 5]

    import os
    from src.search import predictor

    original_exists = os.path.exists
    original_predictor = predictor.CADGCLPredictor
    os.path.exists = lambda path: True
    predictor.CADGCLPredictor = FakePredictor
    runner = CliRunner()
    result = runner.invoke(app, ['search', '--query', '0', '--k', '5'])
    os.path.exists = original_exists
    predictor.CADGCLPredictor = original_predictor
    assert result.exit_code == 0
    assert "Top-5 similar models" in result.output


def test_cli_train_passes_dataset(monkeypatch):
    from src import cli

    calls = []

    def fake_run(cmd, check, cwd):
        calls.append(cmd)

    monkeypatch.setattr("subprocess.run", fake_run)
    cli.train(epochs=1, quick=False, batch_size=8, limit=2, skip_embeddings=True, dataset="NewDataset")

    assert "--dataset" in calls[0]
    assert "NewDataset" in calls[0]


def test_cli_search_accepts_dataset(monkeypatch):
    from src import cli

    called = {}

    class FakePredictor:
        def __init__(self, model_path, embeddings_path):
            called["model_path"] = model_path
            called["embeddings_path"] = embeddings_path

        def load(self):
            pass

        def search(self, query, k):
            return [1, 2, 3]

    monkeypatch.setattr("src.search.predictor.CADGCLPredictor", FakePredictor)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    cli.search(query=0, k=3, dataset="NewDataset")

    assert called["model_path"] == "dataset/NewDataset/checkpoints/cadgcl_model.pt"
    assert called["embeddings_path"] == "dataset/NewDataset/embeddings/embeddings.pt"
