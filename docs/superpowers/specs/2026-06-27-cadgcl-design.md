# CADGCL Implementation Design

**Date**: 2026-06-27
**Dataset**: FabWave (44 categories, 4,571 STEP files with matching images)
**Reference**: CADGCL: unsupervised retrieval of CAD models via boundary representations (Qin et al.)

## 1. Architecture Overview

The system implements a complete CADGCL pipeline:
- **Input**: STEP files from dataset folder
- **Processing**: B-rep attributed graph extraction → Graph contrastive learning
- **Output**: Top-5 similar models ranked by cosine similarity

### Directory Structure
```
dataset/
└── fabwave/         # Plug-and-play dataset (drop folder here)
    ├── step_files/
    └── images/

src/
├── data/
│   ├── step_parser.py     # Parse STEP files using pythonocc
│   └── graph_builder.py   # Build B-rep attributed graph (VGNet: nodes=surfaces, edges=curves)
├── model/
│   ├── gnn_encoder.py     # BAGConv-based GNN (VGNet: 2-layer BAGConv, ReLU, dim=256)
│   ├── ebc_augmentation.py # EBC edge perturbation (paper formula)
│   ├── bmm_sampler.py     # Beta Mixture Model negative sampling
│   └── contrastive_loss.py # InfoNCE loss (τ=0.07) with BMM weighting
```

## 2. Data Transformation Module

### 2.1 STEP to B-rep Graph Conversion (VGNet Specification)
- Extract surfaces (faces) as nodes with geometric features
- Extract curves as edges representing adjacency between surfaces
- Edge features: curve type (one-hot), direction (3), length (1)
- **Adjacency rule**: Edge exists if surfaces share a curve
- Graph representation: G=(V,E) where |V|=surfaces, |E|=curves

### 2.2 Surface Types (Comprehensive STEP List)
Based on STEP ISO 10303-21 standard and FabWave discovery:

| Type | FabWave Count | Description |
|------|--------------|-------------|
| PLANE | 30,671 | Flat planar surface |
| CYLINDRICAL_SURFACE | 55,289 | Cylindrical surface |
| CONICAL_SURFACE | 5,197 | Conical surface |
| TOROIDAL_SURFACE | 2,465 | Torus surface |
| SPHERICAL_SURFACE | 20 | Spherical surface |
| B_SPLINE_SURFACE_WITH_KNOTS | 37,960 | NURBS surface with explicit knots |
| RATIONAL_B_SPLINE_SURFACE | 0 | Rational B-spline (future-proof) |
| SURFACE_OF_REVOLUTION | 0 | Surface created by revolution (future-proof) |
| SURFACE_OF_EXTRUSION | 0 | Surface created by extrusion (future-proof) |
| OFFSET_SURFACE | 0 | Offset from another surface (future-proof) |

**Total: 10 surface types (6 discovered + 4 future-proof)**

**Node feature dimension d = 10 (types) + 3 (normal) + 3 (tangent) = 16**

### 2.3 Curve Types (Comprehensive STEP List)
Based on STEP ISO 10303-21 standard and FabWave discovery:

| Type | FabWave Count | Description |
|------|--------------|-------------|
| EDGE_CURVE (LINE/CIRCLE) | 343,727 | Line or circular arc curves |
| B_SPLINE_CURVE_WITH_KNOTS | 167,765 | NURBS curve with explicit knots |
| POLYLINE | 0 | Polyline approximation (future-proof) |
| BEZIER_CURVE | 0 | Bezier curve (future-proof) |
| HYPERBOLA | 0 | Hyperbolic curve (future-proof) |
| PARABOLA | 0 | Parabolic curve (future-proof) |
| ELLIPSE | 0 | Elliptical curve (future-proof) |

**Total: 7 curve types (2 discovered + 5 future-proof)**

**Edge feature dimension = 7 (types) + 3 (direction) + 1 (length) = 11**

### 2.4 Feature Augmentation (VGNet + CADGCL)
- **Feature Masking**: Bernoulli mask with p_mask=0.3, set masked features to 0
- **EBC Perturbation**: Remove edges based on betweenness centrality
  - EBC formula from CADGCL paper: EB C(e) = Σ σ(s,t|e) / σ(s,t) for all node pairs
  - Remove top p_edge portion of edges ranked by EBC
- **Edge removal rate**: p_edge=0.1 (from paper hyperparameter analysis)

## 3. Contrastive Learning Module

### 3.1 GNN Encoder (VGNet Specification)
- **Architecture**: 2-layer BAGConv (based on SAGEConv) + BAGPool  
- **Input node features**: Surface type (one-hot, 10 types) + normal vector (3) + tangent vector (3) = 16
- **Input edge features**: Curve type (one-hot, 7 types) + direction (3) + length (1) = 11
- **Hidden dimension**: 256 throughout
- **Update function** (from VGNet Eq.1): `h^(k+1)_i = σ(W^(k) · [h_i, Mean({f_Δ(e_ij) ⊕ h_j, ∀j ∈ N(i)})])`
- **Neighbor sampling ξ**: Variable (configurable), default ξ=1.0 (full neighborhood)
- **f_Δ projection**: FC(11, 14) linear layer (VGNet: FC(5, 14) for aligning edge features to hidden space)
- **BAGPool** (from VGNet Eq.2): `Score_ij = Softmax(W^(k)(f_Δ(e_ij) ⊕ [h_i, h_j]) + b^(k))`
- **Readout function**: Average pooling for graph embeddings
- **Activation**: ReLU
- **Batch Norm**: Removed from residual mapping, output layer only

### 3.2 Negative Sampling Strategy
- **Beta Mixture Model (BMM)**: Two-component mixture for true/false negatives
- **EM Algorithm**: Fit BMM parameters after epoch M
- **Weighting**: Use posterior probabilities to weight contrastive loss

### 3.3 Loss Function
- **InfoNCE Loss**: -log(φ(u_i, v_i) / Σ φ(u_i, v_k) + Σ φ(u_i, u_k))
- **φ(u, v) = exp(sim(u, v) / τ)** where sim = cosine similarity
- **Temperature τ**: 0.07 (research-based for CAD/graph similarity)
- **Weighted Version**: Lw = -log(Σ w(i,k)φ(u_i,v_k) / Σ w(i,k)φ(u_i,v_k) + w(i,k)φ(u_i,u_k))
- **Training epochs**: 20 (with M threshold for BMM initialization, typically M=5)

## 4. Dataset Management

### 4.1 Multi-Dataset Support
- Drop new dataset folder into `dataset/` directory
- Dataset must contain `step_files/` and `images/` subdirectories
- Command: `python run.py --dataset <name>` to train/onboard new dataset

### 4.2 UI Isolation
- Each dataset maintains separate trained weights
- Dataset dropdown selector in UI
- No cross-dataset queries (single dataset mode only)

## 5. Training Configuration

```
Optimizer: Adam
Learning rate: 0.01
Batch size: 128
Epochs: 20
Train/Test split: 80/20
Feature dimension: 256
Feature mask rate: 0.3
Edge perturbation rate: 0.1
```

## 6. Implementation Plan

### Phase 1: Data Infrastructure
1. STEP parser implementation
2. Graph builder with node/edge extraction
3. Dataset loader with validation

### Phase 2: Core Model
1. GNN encoder (2-layer GCN)
2. EBC augmentation module
3. BMM negative sampler

### Phase 3: Training
1. Contrastive loss implementation
2. Trainer with checkpointing
3. Embedding generation for all models

### Phase 4: UI
1. Dataset selector
2. Model search interface
3. Results display (top-5 similar models)

## 7. Success Criteria

- Top-5 mAP ≥ 89.35% on FabWave (paper's achieved performance)
- Model retrieval completes in <5 seconds
- Drop-in dataset support with single command
- Clean separation between datasets (no cross-contamination)

## 8. Implementation Decisions

| Parameter | Decision | Notes |
|-----------|----------|-------|
| **Surface type encoding** | ✅ One-hot encoding | Fixed list of 10 types (6 discovered + 4 future-proof) |
| **Curve type encoding** | ✅ One-hot encoding | Fixed list of 7 types (2 discovered + 5 future-proof) |
| **STEP parsing library** | ✅ pythonocc-core | Full OpenCASCADE bindings for B-rep access |
| **Node feature dimension d** | ✅ 16 | 10 (types) + 3 (normal) + 3 (tangent) |
| **Edge feature dimension** | ✅ 11 | 7 (types) + 3 (direction) + 1 (length) |
| **Neighbor sampling ξ** | ✅ 1.0 (full) | Variable configurable, default uses all neighbors |
| **f_Δ projection** | ✅ FC(11, 16) | Projects 11-dim edge to 16-dim to match node features for concatenation |
| **BMM start epoch M** | ✅ M=5 | Begin BMM fitting after 5 epochs