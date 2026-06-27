### Task 3: BAGConv Layer

**Files:**
- Create: `src/model/gnn_encoder.py`
- Test: `tests/test_gnn_encoder.py`

**Interfaces:**
- Consumes: Graph with node features (16), edge features (11)
- Produces: Graph embeddings (256 dims)

**Step 1: Write the failing test**

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

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_gnn_encoder.py -v`
Expected: FAIL

**Step 3: Write BAGConv implementation**

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

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_gnn_encoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/gnn_encoder.py tests/test_gnn_encoder.py
git commit -m "feat: add BAGConv layer implementation"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-3-report.md`