### Task 7: EBC Augmentation

**Files:**
- Create: `src/model/ebc_augmentation.py`
- Test: `tests/test_ebc_augmentation.py`

**Interfaces:**
- Consumes: Graph edges
- Produces: Perturbed graph (edge removal p=0.1)

**Step 1: Write the failing test**

```python
def test_ebc_augmentation():
    from src.model.ebc_augmentation import EBCAugmentation
    
    aug = EBCAugmentation(p_edge=0.1)
    edge_index = torch.tensor([[0,1,2], [1,2,0]])  # 3 edges
    edge_attr = torch.randn(3, 11)
    new_edge_index, new_edge_attr = aug(edge_index, edge_attr)
    # Remove edges with LOWEST EBC (not highest) to remove ~10% bridges
    assert new_edge_index.shape[1] <= edge_index.shape[1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ebc_augmentation.py -v`
Expected: FAIL

**Step 3: Write EBC augmentation implementation**

The EBC (Edge Betweenness Centrality) augmentation removes edges with highest betweenness (bridges), which should be ~10% of edges.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ebc_augmentation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/ebc_augmentation.py tests/test_ebc_augmentation.py
git commit -m "feat: add EBC edge perturbation augmentation"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-7-report.md`