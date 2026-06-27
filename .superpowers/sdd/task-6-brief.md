### Task 6: BMM Negative Sampler

**Files:**
- Create: `src/model/bmm_sampler.py`
- Test: `tests/test_bmm_sampler.py`

**Interfaces:**
- Consumes: Similarity scores
- Produces: Sample weights for loss

**Step 1: Write the failing test**

```python
def test_bmm_sampler():
    from src.model.bmm_sampler import BMMSampler
    
    sampler = BMMSampler()
    similarities = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.9])  # 5 negative samples
    weights = sampler(similarities)
    assert weights.shape == similarities.shape
    assert torch.all(weights >= 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bmm_sampler.py -v`
Expected: FAIL

**Step 3: Write BMM implementation**

The Beta Mixture Model (BMM) for negative sampling:
- Two-component Beta distribution to distinguish true negatives (low similarity) from false negatives (high similarity)
- Learnable alpha, beta parameters
- Returns normalized weights for sampling

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bmm_sampler.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/bmm_sampler.py tests/test_bmm_sampler.py
git commit -m "feat: add Beta Mixture Model negative sampler"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-6-report.md`