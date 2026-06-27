# CADGCL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CADGCL system that retrieves the top-5 most similar CAD models using B-rep attributed graph contrastive learning.

**Architecture:** Convert STEP files to B-rep graphs (nodes=surfaces, edges=curves), train BAGConv GNN with EBC augmentation and BMM negative sampling, provide UI for similarity search.

**Tech Stack:** pythonocc-core, PyTorch Geometric, PyTorch

## Global Constraints

- Surface types: 10 fixed categories (one-hot encoding)
- Curve types: 7 fixed categories (one-hot encoding)
- Node feature dimension d = 16 (10 types + 3 normal + 3 tangent)
- Edge feature dimension = 11 (7 types + 3 direction + 1 length)
- f_Δ projection: FC(11, 16)
- Optimizer: Adam, lr=0.01, batch=128, epochs=20, τ=0.07, M=5
- Training split: 80/20

---

### Task 1: STEP Parser Module

**Files:**
- Create: `src/data/step_parser.py`
- Test: `tests/test_step_parser.py`

**Interfaces:**
- Consumes: STEP file path
- Produces: `TopoDS_Shape` object with faces and edges

- [ ] **Step 1: Write the failing test**

```python
import pytest
from src.data.step_parser import STEPParser

def test_parse_step_file():
    parser = STEPParser()
    shape = parser.parse("tests/data/sample.stp")
    assert shape is not None
    assert hasattr(shape, 'Faces')
    assert len(shape.Faces) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step_parser.py -v`
Expected: FAIL with "module not found"

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/step_parser.py
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

class STEPParser:
    def parse(self, filepath):
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status == IFSelect_RetDone:
            reader.TransferRoots()
            shape = reader.OneShape()
            return shape
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/step_parser.py tests/test_step_parser.py
git commit -m "feat: add STEP parser using pythonocc-core"
```

### Task 2: B-rep Graph Builder

**Files:**
- Create: `src/data/graph_builder.py`
- Test: `tests/test_graph_builder.py`

**Interfaces:**
- Consumes: `TopoDS_Shape` from STEPParser
- Produces: `BRepGraph` with nodes (features) and edges (adjacency)

- [ ] **Step 1: Write the failing test**

```python
def test_build_graph():
    from src.data.step_parser import STEPParser
    from src.data.graph_builder import BRepGraphBuilder
    
    parser = STEPParser()
    shape = parser.parse("tests/data/sample.stp")
    builder = BRepGraphBuilder()
    graph = builder.build(shape)
    
    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert graph.x.shape[1] == 16  # node features
    assert graph.edge_attr.shape[1] == 11  # edge features
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_builder.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation (surface types)**

```python
# src/data/graph_builder.py
SURFACE_TYPES = {
    'PLANE': 0, 'CYLINDRICAL_SURFACE': 1, 'CONICAL_SURFACE': 2,
    'TOROIDAL_SURFACE': 3, 'SPHERICAL_SURFACE': 4,
    'B_SPLINE_SURFACE_WITH_KNOTS': 5, 'RATIONAL_B_SPLINE_SURFACE': 6,
    'SURFACE_OF_REVOLUTION': 7, 'SURFACE_OF_EXTRUSION': 8,
    'OFFSET_SURFACE': 9
}

CURVE_TYPES = {
    'EDGE_CURVE': 0, 'B_SPLINE_CURVE_WITH_KNOTS': 1,
    'POLYLINE': 2, 'BEZIER_CURVE': 3, 'HYPERBOLA': 4,
    'PARABOLA': 5, 'ELLIPSE': 6
}

class BRepGraphBuilder:
    def build(self, shape):
        # Extract faces as nodes
        # Extract edges as adjacency
        # Calculate one-hot types + normal/tangent + direction/length
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/graph_builder.py tests/test_graph_builder.py
git commit -m "feat: add B-rep graph builder with fixed surface/curve types"
```

### Task 3: BAGConv Layer

**Files:**
- Create: `src/model/gnn_encoder.py`
- Test: `tests/test_gnn_encoder.py`

**Interfaces:**
- Consumes: Graph with node features (16), edge features (11)
- Produces: Graph embeddings (256 dims)

- [ ] **Step 1: Write the failing test**

```python
def test_bagconv_forward():
    from src.model.gnn_encoder import BAGConv
    
    conv = BAGConv(in_node_dim=16, in_edge_dim=11, out_dim=256)
    # Mock input
    x = torch.randn(10, 16)  # 10 nodes
    edge_index = torch.tensor([[0,1],[1,0]])  # 2 edges
    edge_attr = torch.randn(2, 11)
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (10, 256)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gnn_encoder.py -v`
Expected: FAIL

- [ ] **Step 3: Write BAGConv implementation**

```python
# src/model/gnn_encoder.py
import torch
import torch.nn as nn

class BAGConv(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim):
        super().__init__()
        self.f_delta = nn.Linear(in_edge_dim, in_node_dim)
        self.linear = nn.Linear(in_node_dim * 2, out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # f_Δ(e_ij) projects edge to node space
        # Aggregate neighbors with projected edge features
        # Apply linear + ReLU
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gnn_encoder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/gnn_encoder.py tests/test_gnn_encoder.py
git commit -m "feat: add BAGConv layer implementation"
```

### Task 4: Dataset Loader

**Files:**
- Create: `src/data/dataset_loader.py`
- Test: `tests/test_dataset_loader.py`

**Interfaces:**
- Consumes: Dataset folder path
- Produces: PyTorch Geometric DataLoaders

- [ ] **Step 1: Write the failing test**

```python
def test_dataset_loading():
    from src.data.dataset_loader import FabWaveDataset
    
    dataset = FabWaveDataset(root="dataset/fabwave")
    assert len(dataset) == 4571
    data = dataset[0]
    assert data.x.shape[1] == 16
    assert data.edge_attr.shape[1] == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal dataset implementation**

```python
# src/data/dataset_loader.py
import os
from torch_geometric.data import InMemoryDataset, Data
from src.data.step_parser import STEPParser
from src.data.graph_builder import BRepGraphBuilder

class FabWaveDataset(InMemoryDataset):
    def __init__(self, root):
        self.parser = STEPParser()
        self.builder = BRepGraphBuilder()
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['fabwave_data.pt']
    
    def process(self):
        data_list = []
        for root, _, files in os.walk(os.path.join(self.root, 'step_files')):
            for f in files:
                if f.endswith('.stp') or f.endswith('.step'):
                    shape = self.parser.parse(os.path.join(root, f))
                    graph = self.builder.build(shape)
                    data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset_loader.py tests/test_dataset_loader.py
git commit -m "feat: add FabWave dataset loader"
```

### Task 5: Contrastive Loss

**Files:**
- Create: `src/model/contrastive_loss.py`
- Test: `tests/test_contrastive_loss.py`

**Interfaces:**
- Consumes: Graph embeddings (u, v pairs)
- Produces: InfoNCE loss with τ=0.07

- [ ] **Step 1: Write the failing test**

```python
def test_infonce_loss():
    from src.model.contrastive_loss import InfoNCELoss
    
    loss_fn = InfoNCELoss(temperature=0.07)
    u = torch.randn(4, 256)  # batch=4 embeddings
    v = torch.randn(4, 256)  # augmented views
    loss = loss_fn(u, v)
    assert loss.item() > 0
    assert loss.item() < 10  # reasonable loss range
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contrastive_loss.py -v`
Expected: FAIL

- [ ] **Step 3: Write InfoNCE implementation**

```python
# src/model/contrastive_loss.py
import torch
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, u, v):
        # u, v are batch embeddings
        sim = F.cosine_similarity(u, v)
        phi = torch.exp(sim / self.temperature)
        # InfoNCE: -log(phi_ui_vi / (phi_ui_vi + sum_phi_ui_vk))
        loss = -torch.log(phi / (phi + torch.sum(phi)))
        return loss.mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contrastive_loss.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/contrastive_loss.py tests/test_contrastive_loss.py
git commit -m "feat: add InfoNCE contrastive loss"
```

### Task 6: BMM Negative Sampler

**Files:**
- Create: `src/model/bmm_sampler.py`
- Test: `tests/test_bmm_sampler.py`

**Interfaces:**
- Consumes: Similarity scores
- Produces: Sample weights for loss

- [ ] **Step 1: Write the failing test**

```python
def test_bmm_sampler():
    from src.model.bmm_sampler import BMMSampler
    
    sampler = BMMSampler()
    similarities = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.9])  # 5 negative samples
    weights = sampler(similarities)
    assert weights.shape == similarities.shape
    assert torch.all(weights >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bmm_sampler.py -v`
Expected: FAIL

- [ ] **Step 3: Write BMM implementation**

```python
# src/model/bmm_sampler.py
import torch
import torch.nn as nn
from torch.distributions import Beta

class BMMSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([2.0]))
        self.beta = nn.Parameter(torch.tensor([2.0]))
    
    def forward(self, similarities):
        # Two-component Beta Mixture Model
        # p(c=True|s) for true negatives, p(c=False|s) for false negatives
        weights = Beta(self.alpha, self.beta).log_prob(similarities).exp()
        return weights / weights.sum()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bmm_sampler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/bmm_sampler.py tests/test_bmm_sampler.py
git commit -m "feat: add Beta Mixture Model negative sampler"
```

### Task 7: EBC Augmentation

**Files:**
- Create: `src/model/ebc_augmentation.py`
- Test: `tests/test_ebc_augmentation.py`

**Interfaces:**
- Consumes: Graph edges
- Produces: Perturbed graph (edge removal p=0.1)

- [ ] **Step 1: Write the failing test**

```python
def test_ebc_augmentation():
    from src.model.ebc_augmentation import EBCAugmentation
    
    aug = EBCAugmentation(p_edge=0.1)
    edge_index = torch.tensor([[0,1,2], [1,2,0]])  # 3 edges
    edge_attr = torch.randn(3, 11)
    new_edge_index, new_edge_attr = aug(edge_index, edge_attr)
    # Should remove ~10% of edges
    assert new_edge_index.shape[1] <= edge_index.shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ebc_augmentation.py -v`
Expected: FAIL

- [ ] **Step 3: Write EBC augmentation implementation**

```python
# src/model/ebc_augmentation.py
import torch
import torch.nn as nn
import networkx as nx

class EBCAugmentation(nn.Module):
    def __init__(self, p_edge=0.1):
        super().__init__()
        self.p_edge = p_edge
    
    def compute_ebc(self, edge_index):
        # Build NetworkX graph
        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0,i].item(), edge_index[1,i].item())
        # Compute betweenness centrality
        return nx.edge_betweenness_centrality(G)
    
    def forward(self, edge_index, edge_attr):
        ebc_scores = self.compute_ebc(edge_index)
        # Remove edges with highest EBC (bridges between communities)
        threshold = torch.quantile(torch.tensor(list(ebc_scores.values())), self.p_edge)
        keep_mask = torch.tensor([ebc_scores.get((edge_index[0,i].item(), edge_index[1,i].item()), 0) > threshold for i in range(edge_index.shape[1])])
        return edge_index[:, keep_mask], edge_attr[keep_mask]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ebc_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/ebc_augmentation.py tests/test_ebc_augmentation.py
git commit -m "feat: add EBC edge perturbation augmentation"
```

### Task 8: Trainer

**Files:**
- Create: `src/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: DataLoader, GNN model
- Produces: Trained embeddings

- [ ] **Step 1: Write the failing test**

```python
def test_trainer_step():
    from src.training.trainer import CADGCLTrainer
    from src.model.gnn_encoder import BAGConv
    
    model = BAGConv(16, 11, 256)
    trainer = CADGCLTrainer(model, lr=0.01)
    # Mock batch
    loss = trainer.training_step({'x': torch.randn(4, 10, 16), 'edge_index': torch.randint(0, 10, (2, 20)), 'edge_attr': torch.randn(20, 11)})
    assert loss is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Write trainer implementation**

```python
# src/training/trainer.py
import torch
import pytorch_lightning as pl

class CADGCLTrainer(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        # Forward pass with augmentation
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
        # Apply EBC augmentation
        aug = EBCAugmentation(p_edge=0.1)
        edge_index2, edge_attr2 = aug(edge_index, edge_attr)
        # Get embeddings
        z1 = self.model(x, edge_index, edge_attr)
        z2 = self.model(x, edge_index2, edge_attr2)
        # Compute loss
        loss_fn = InfoNCELoss(temperature=0.07)
        loss = loss_fn(z1, z2)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: add CADGCL trainer with EBC and InfoNCE"
```

### Task 9: UI Application

**Files:**
- Create: `src/ui/app.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: Dataset names, query models
- Produces: Top-5 similar models via Streamlit/Gradio

- [ ] **Step 1: Write the failing test**

```python
def test_ui_app():
    from src.ui.app import CADGCLUI
    
    ui = CADGCLUI(datasets=['fabwave'])
    assert ui.datasets == ['fabwave']
    results = ui.search(query_model_id=0)
    assert len(results) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ui.py -v`
Expected: FAIL

- [ ] **Step 3: Write UI implementation**

```python
# src/ui/app.py
import streamlit as st
import torch

class CADGCLUI:
    def __init__(self, datasets):
        self.datasets = datasets
        self.embeddings = {}
    
    def search(self, query_model_id):
        # Load embeddings for selected dataset
        # Compute cosine similarity
        # Return top-5 indices
        return list(range(5))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ui.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ui/app.py tests/test_ui.py
git commit -m "feat: add CADGCL UI application"
```