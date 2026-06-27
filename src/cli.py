try:
    import typer
except ImportError:
    typer = None
import torch

if typer:
    app = typer.Typer()

    @app.command()
    def search(query: int = typer.Option(..., help="Query model ID"), 
               k: int = typer.Option(5, help="Number of results"),
               model: str = typer.Option("checkpoints/cadgcl_model.pt"),
               embeddings: str = typer.Option("embeddings/fabwave.pt")):
        """Search for similar CAD models by query ID."""
        from src.search.predictor import CADGCLPredictor
        predictor = CADGCLPredictor(model_path=model, embeddings_path=embeddings)
        predictor.load()
        results = predictor.search(query, k)
        print(f"Top-{k} similar models: {results}")

    @app.command()
    def train(dataset: str = typer.Option("dataset/FabWave", help="Dataset path"),
              epochs: int = typer.Option(20, help="Training epochs")):
        """Train a new model on a dataset."""
        import subprocess
        import sys
        cmd = [sys.executable, "scripts/train.py", "--dataset", dataset, "--epochs", str(epochs)]
        subprocess.run(cmd, check=True)

    def main():
        if typer:
            app()

    if __name__ == '__main__':
        main()
else:
    # Fallback: simple function interface
    def search(query, k=5, model="checkpoints/cadgcl_model.pt", embeddings="embeddings/fabwave.pt"):
        from src.search.predictor import CADGCLPredictor
        predictor = CADGCLPredictor(model_path=model, embeddings_path=embeddings)
        predictor.load()
        results = predictor.search(query, k)
        print(f"Top-{k} similar models: {results}")