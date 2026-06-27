try:
    import typer
except ImportError:
    typer = None
import torch
import os

def search_cmd(query: int, k: int = 5, model: str = "checkpoints/cadgcl_model.pt", embeddings: str = "embeddings/fabwave.pt"):
    """Search for similar CAD models by query ID."""
    from src.search.predictor import CADGCLPredictor
    
    if not os.path.exists(model):
        print(f"Error: Model not found at {model}. Run 'python -m src.cli train --quick' first.")
        return
    if not os.path.exists(embeddings):
        print(f"Error: Embeddings not found at {embeddings}. Generate embeddings first.")
        return
    
    predictor = CADGCLPredictor(model_path=model, embeddings_path=embeddings)
    predictor.load()
    results = predictor.search(query, k)
    print(f"Top-{k} similar models for query {query}: {results}")

def train_cmd(dataset: str = "dataset/FabWave", epochs: int = 1, quick: bool = False):
    """Train a CADGCL model on a dataset."""
    import sys
    cmd = [sys.executable, "scripts/train.py", "--dataset", dataset, "--epochs", str(epochs)]
    if quick:
        cmd.append("--quick")
    import subprocess
    subprocess.run(cmd, check=True)

def ui_cmd():
    """Launch simple similarity search UI."""
    try:
        import faiss
        import typer
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        print("Install optional dependencies: pip install faiss-cpu typer rich")
        return
    
    model = "checkpoints/cadgcl_model.pt"
    embeddings_path = "embeddings/fabwave.pt"
    
    if not os.path.exists(model) or not os.path.exists(embeddings_path):
        print(f"Error: Missing model or embeddings. Run 'python -m src.cli train --quick' first.")
        return
    
    from src.search.predictor import CADGCLPredictor
    predictor = CADGCLPredictor(model_path=model, embeddings_path=embeddings_path)
    predictor.load()
    
    console = Console()
    console.print("[bold blue]CADGCL Similarity Search[/bold blue]")
    
    while True:
        try:
            query = Prompt.ask("Enter query model ID (0-4570, or 'q' to quit)")
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

if typer:
    app = typer.Typer()
    app.command()(search_cmd)
    app.command()(train_cmd)
    app.command()(ui_cmd)
    
    if __name__ == '__main__':
        app()
else:
    # Function interface for use without typer
    def search(query, k=5, model="checkpoints/cadgcl_model.pt", embeddings="embeddings/fabwave.pt"):
        search_cmd(query, k, model, embeddings)
    
    def train(dataset="dataset/FabWave", epochs=1, quick=False):
        train_cmd(dataset, epochs, quick)