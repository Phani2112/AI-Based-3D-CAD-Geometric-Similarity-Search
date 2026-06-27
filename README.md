# CADGCL - CAD Model Similarity Search

B-rep attributed graph contrastive learning for CAD model retrieval.

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Search for Similar Models
```bash
# Load pre-trained model and embeddings
python -m src.cli search --query 0 --k 5
```

### Train on New Dataset
```bash
# Generate embeddings after training
python -m src.cli train --dataset /path/to/step/files --epochs 20
python scripts/generate_embeddings.py --dataset /path/to/step/files
```

## Architecture
- STEP Parser → BRep Graph Builder → BAGConv GNN → Global Mean Pool → InfoNCE Loss
- Surfaces as nodes (10 types, 3 normal, 3 tangent → 16 dims)
- Curves as edges (7 types, 3 direction, 1 length → 11 dims)

## Files
- `src/data/step_parser.py` - STEP file parsing
- `src/data/graph_builder.py` - BRep to graph conversion
- `src/data/dataset_loader.py` - PyTorch Geometric dataset
- `src/model/gnn_encoder.py` - BAGConv + pooling
- `src/model/contrastive_loss.py` - InfoNCE loss
- `src/model/bmm_sampler.py` - Beta Mixture Model sampler
- `src/model/ebc_augmentation.py` - Edge betweenness centrality augmentation
- `src/search/predictor.py` - Inference pipeline
- `src/cli.py` - CLI entry point

## Usage

### Training
```bash
python scripts/train.py --dataset dataset/FabWave --epochs 20 --output checkpoints/model.pt
```

### Generate Embeddings
```bash
python scripts/generate_embeddings.py --model checkpoints/model.pt
```

### Similarity Search
```bash
python -c "from src.cli import search; search(query=0, k=5)"
```