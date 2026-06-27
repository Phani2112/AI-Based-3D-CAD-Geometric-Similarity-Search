import torch
from typer.testing import CliRunner
from src.cli import app

def test_cli_search_command():
    try:
        import faiss
        import typer
    except ImportError:
        return
    
    runner = CliRunner()
    # Mock files already exist
    result = runner.invoke(app, ['search', '--query', '0', '--k', '5'])
    assert result.exit_code == 0
    assert "Top-5 similar models" in result.output