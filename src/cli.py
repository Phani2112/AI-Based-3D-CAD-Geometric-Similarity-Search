try:
    import typer
except ImportError:
    typer = None
import torch
import os

def search(query: int = typer.Option(..., help="Query model ID"), 
           k: int = typer.Option(5, help="Number of results")):
    """Search for similar CAD models by query ID."""
    from src.search.predictor import CADGCLPredictor
    
    model_path = "checkpoints/cadgcl_model.pt"
    embeddings_path = "embeddings/fabwave.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found. Run 'python -m src.cli train --quick' first.")
        return
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings not found.")
        return
    
    predictor = CADGCLPredictor(model_path=model_path, embeddings_path=embeddings_path)
    predictor.load()
    results = predictor.search(query, k)
    print(f"Top-{k} similar models for query {query}: {results}")

def train(epochs: int = typer.Option(1, help="Training epochs"), 
          quick: bool = typer.Option(False, help="Quick demo with synthetic data")):
    """Train a CADGCL model."""
    import sys
    import subprocess
    cmd = [sys.executable, "scripts/train.py", "--epochs", str(epochs)]
    if quick:
        cmd.append("--quick")
    subprocess.run(cmd, check=True)

def ui():
    """Launch interactive similarity search UI."""
    try:
        import faiss
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        print("Install optional dependencies: pip install faiss-cpu rich")
        return
    
    model_path = "checkpoints/cadgcl_model.pt"
    embeddings_path = "embeddings/fabwave.pt"
    
    if not os.path.exists(model_path) or not os.path.exists(embeddings_path):
        print("Error: Missing model or embeddings. Run 'python -m src.cli train --quick' first.")
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

if typer:
    app = typer.Typer()
    app.command()(search)
    app.command()(train)
    app.command()(ui)
    
    if __name__ == '__main__':
        app()
else:
    def search_fallback(query, k=5):
        search_cmd = lambda query=query, k=k: None
        search_cmd(query, k)
    
    def train_fallback(epochs=1, quick=False):
        pass