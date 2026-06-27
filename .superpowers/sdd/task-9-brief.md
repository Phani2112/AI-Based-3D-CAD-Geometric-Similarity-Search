### Task 9: UI Application

**Files:**
- Create: `src/ui/app.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: Dataset names, query models
- Produces: Top-5 similar models via cosine similarity

**Step 1: Write the failing test**

```python
def test_ui_app():
    import torch
    from src.ui.app import CADGCLUI
    
    ui = CADGCLUI(datasets=['fabwave'])
    assert ui.datasets == ['fabwave']
    
    # Mock embeddings for testing
    ui.embeddings['fabwave'] = torch.randn(100, 256)  # 100 models, 256-dim embeddings
    results = ui.search(query_model_id=0)
    assert len(results) == 5
    assert all(0 <= r < 100 for r in results)  # Valid model indices
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ui.py -v`
Expected: FAIL

**Step 3: Write UI implementation**

```python
# src/ui/app.py
import torch
import torch.nn.functional as F

class CADGCLUI:
    def __init__(self, datasets):
        self.datasets = datasets
        self.embeddings = {}
    
    def load_embeddings(self, dataset_name, embeddings):
        """Load pre-computed graph embeddings for a dataset."""
        self.embeddings[dataset_name] = embeddings
    
    def search(self, query_model_id, dataset='fabwave'):
        """Find top-5 similar models using cosine similarity."""
        if dataset not in self.embeddings:
            return []
        
        query_emb = self.embeddings[dataset][query_model_id:query_model_id+1]
        all_embs = self.embeddings[dataset]
        
        similarities = F.cosine_similarity(query_emb, all_embs)
        top5_idx = similarities.topk(k=5, sorted=True)[1].tolist()
        
        return top5_idx
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ui.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ui/app.py tests/test_ui.py
git commit -m "feat: add CADGCL UI application"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-9-report.md`