# CADGCL - CAD Model Similarity Search

B-rep attributed graph contrastive learning for CAD model retrieval.

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch Browser-based Similarity Search
```bash
python -m src.cli web
```

### Run CLI Search
```bash
python -m src.cli search --query 0 --k 5
```

### Train on New Dataset (demo)
```bash
python -m src.cli train --quick --epochs 20
```

## Architecture

## Files

## Usage

### Launch Browser-based UI
```bash
python -m src.cli web
```

### Launch Terminal UI
```bash
python -m src.cli ui
```

### Training (full)
```bash
python scripts/train.py --dataset dataset/FabWave --epochs 20
```

## Performance vs Paper

| Metric | Paper (mAP@50) | Our Implementation (mAP@10) |
|--------|----------------|---------------------------|
| Target | ≥ 89.35% | ~89% (estimated) |
| Mock model | N/A | 19.44% |
| Full training | N/A | Run `scripts/evaluate.py` |

**Evaluate trained model:**
```bash
python scripts/evaluate.py
```

### Generate Embeddings
```bash
python scripts/generate_embeddings.py --model checkpoints/cadgcl_model.pt
```
 # CADGCL

 CADGCL is a research prototype for similarity search over 3D CAD models. It parses STEP boundary representations into attributed graphs, learns graph embeddings with contrastive learning, and searches the embedding space with FAISS.

 ## Project Status

This repository contains the source code, tests, research documentation, and the approved open-source FabWave dataset. The Gühring dataset and any company-derived files must remain private. Trained weights, embeddings, and processed tensors are generated artifacts and should be published only when their data provenance and redistribution rights are clear.

 ## Setup

 Use Python 3.10 or newer in a virtual environment:

 ```bash
 python -m venv .venv
 # Windows PowerShell
 .\.venv\Scripts\Activate.ps1
 python -m pip install --upgrade pip
 python -m pip install -e ".[dev]"
 ```

 The `OCP`, PyTorch, PyTorch Geometric, FAISS, and Streamlit dependencies can require platform-specific installation choices. If a platform wheel is unavailable, install the compatible PyTorch stack first and then install this project.

 ## Commands

 After installing the package, the CLI is available as `cadgcl` or through `python -m src.cli`.

 ```bash
 # Run the unit tests
 python -m pytest

 # Train a small synthetic smoke-test model
 cadgcl train --quick --epochs 1 --skip-embeddings

 # Train against a dataset folder named under dataset/
 cadgcl train --dataset FabWave --epochs 20

 # Generate embeddings for a trained dataset model
 python scripts/generate_embeddings.py --dataset FabWave

 # Evaluate retrieval quality
 python scripts/evaluate.py --dataset FabWave --k 10

# Launch the browser UI and choose a ready dataset in the app
cadgcl web
 ```

 Dataset arguments are folder names, not paths. Dataset-derived outputs are stored together:

 ```text
 dataset/<name>/
 ├── processed/graphs.pt
 ├── processed/metadata.pt
 ├── checkpoints/cadgcl_model.pt
 └── embeddings/embeddings.pt
 ```

 ## Architecture

 ```text
 STEP file
	 -> STEP parser
	 -> B-rep graph builder
	 -> PyTorch Geometric graph dataset
	 -> BAGConv encoder and global pooling
	 -> contrastive training
	 -> embedding generation
	 -> FAISS similarity search
 ```

 Important modules:

 | Area | Module |
 | --- | --- |
 | STEP parsing | `src/data/step_parser.py` |
 | Graph construction | `src/data/graph_builder.py` |
 | Dataset and artifact paths | `src/data/dataset_loader.py`, `src/data/dataset_registry.py` |
 | Encoder and loss | `src/model/gnn_encoder.py`, `src/model/contrastive_loss.py` |
 | Retrieval | `src/search/predictor.py`, `src/search/faiss_index.py` |
 | CLI and UI | `src/cli.py`, `src/ui/streamlit_app.py` |

 ## Repository Layout

 ```text
 src/       installable application and ML code
 tests/     unit and contract tests
 scripts/   training, embedding, and evaluation workflows
 docs/      architecture and development notes
dataset/   approved FabWave data plus private Gühring data and generated artifacts
 ```

 ## Development Direction

 The next cleanup steps are to consolidate training into one implementation, remove legacy path manipulation from scripts, add a small public STEP fixture for end-to-end testing, and separate approved presentation material from confidential project records. Large datasets and trained artifacts should be distributed separately through approved storage or release assets.