try:
    import typer
except ImportError:
    typer = None
import torch
import os
import sys

sys.path.insert(0, os.getcwd())

def search(query: int = typer.Option(..., help="Query model ID"), 
           k: int = typer.Option(5, help="Number of results"),
           dataset: str = typer.Option("FabWave", help="Dataset folder name under dataset")):
    """Search for similar CAD models by query ID."""
    from src.data.dataset_registry import dataset_artifacts
    from src.search.predictor import CADGCLPredictor
    
    artifacts = dataset_artifacts(dataset)
    model_path = artifacts.model_path
    embeddings_path = artifacts.embeddings_path
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found. Run 'python3 -m src.cli train --quick' first.")
        return
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings not found.")
        return
    
    predictor = CADGCLPredictor(model_path=model_path, embeddings_path=embeddings_path)
    predictor.load()
    results = predictor.search(query, k)
    print(f"Top-{k} similar models for query {query}: {results}")

def train(epochs: int = typer.Option(1, help="Training epochs"), 
          quick: bool = typer.Option(False, help="Quick demo with synthetic data"),
          batch_size: int = typer.Option(128, help="Batch size"),
          limit: int = typer.Option(0, help="Limit real dataset training to N models"),
          skip_embeddings: bool = typer.Option(False, help="Skip embedding generation after training"),
          dataset: str = typer.Option("FabWave", help="Dataset folder name under dataset")):
    """Train a CADGCL model."""
    import subprocess

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--limit",
        str(limit),
        "--dataset",
        dataset,
    ]
    if quick:
        cmd.append("--quick")
    if skip_embeddings:
        cmd.append("--skip-embeddings")
    subprocess.run(cmd, check=True, cwd=os.getcwd())

def ui(dataset: str = typer.Option("FabWave", help="Dataset folder name under dataset")):
    """Launch interactive similarity search UI."""
    try:
        import faiss
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        print("Install optional dependencies: pip install faiss-cpu rich")
        return
    
    from src.data.dataset_registry import dataset_artifacts
    artifacts = dataset_artifacts(dataset)
    model_path = artifacts.model_path
    embeddings_path = artifacts.embeddings_path
    
    if not os.path.exists(model_path) or not os.path.exists(embeddings_path):
        print("Error: Missing model or embeddings. Run 'python3 -m src.cli train --quick' first.")
        return
    
    from src.search.predictor import CADGCLPredictor
    predictor = CADGCLPredictor(model_path=model_path, embeddings_path=embeddings_path)
    predictor.load()
    
    console = Console()
    console.print("[bold blue]CADGCL Similarity Search[/bold blue]")
    console.print("Enter 'q' to quit\n")
    
    while True:
        try:
            query = Prompt.ask("Query model ID (0-4570)")
            if query.lower() == 'q':
                break
            query_id = int(query)
            results = predictor.search(query_id, k=5)
            
            table = Table(title=f"Top-5 similar to model {query_id}")
            table.add_column("Rank", style="cyan")
            table.add_column("Model ID", style="magenta")
            for i, r in enumerate(results, 1):
                table.add_row(str(i), str(r))
            console.print(table)
        except (ValueError, EOFError):
            break

def web():
    """Launch browser-based similarity search UI."""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"])

if typer:
    app = typer.Typer()
    app.command()(search)
    app.command()(train)
    app.command()(ui)
    app.command()(web)
    
    if __name__ == '__main__':
        app()
