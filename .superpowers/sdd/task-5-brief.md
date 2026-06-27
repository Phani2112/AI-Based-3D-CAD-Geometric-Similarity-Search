### Task 5: Contrastive Loss

**Files:**
- Create: `src/model/contrastive_loss.py`
- Test: `tests/test_contrastive_loss.py`

**Interfaces:**
- Consumes: Graph embeddings (u, v pairs)
- Produces: InfoNCE loss with τ=0.07

**Step 1: Write the failing test**

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

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_contrastive_loss.py -v`
Expected: FAIL

**Step 3: Write InfoNCE implementation**

The InfoNCE loss for contrastive learning:
- Given batch of embedding pairs (u, v) where u is original view and v is augmented view
- Similarity: s(u_i, v_i) = cos(u_i, v_i)
- Loss = -log(exp(s(u_i, v_i)/τ) / Σ_j exp(s(u_i, v_j)/τ))

For each anchor u_i, compute similarity with all v_j in batch (positive + negatives).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_contrastive_loss.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/contrastive_loss.py tests/test_contrastive_loss.py
git commit -m "feat: add InfoNCE contrastive loss"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-5-report.md`