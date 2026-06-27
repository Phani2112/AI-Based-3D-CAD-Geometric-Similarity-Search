### Task 8: Trainer

**Files:**
- Create: `src/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: DataLoader, GNN model
- Produces: Trained embeddings

**Step 1: Write the failing test**

```python
def test_trainer_step():
    from src.training.trainer import CADGCLTrainer
    from src.model.gnn_encoder import BAGConv
    
    model = BAGConv(16, 11, 256)
    trainer = CADGCLTrainer(model, lr=0.01)
    # Single graph: x [N, 16], edge_index [2, E], edge_attr [E, 11]
    batch = {
        'x': torch.randn(10, 16),
        'edge_index': torch.randint(0, 10, (2, 20)),
        'edge_attr': torch.randn(20, 11)
    }
    loss = trainer.training_step(batch, 0)
    assert loss is not None
    assert loss.item() > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL

**Step 3: Write trainer implementation**

```python
# src/training/trainer.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model.ebc_augmentation import EBCAugmentation
from src.model.contrastive_loss import InfoNCELoss

class CADGCLTrainer(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.lr = lr
        self.aug = EBCAugmentation(p_edge=0.1)
        self.loss_fn = InfoNCELoss(temperature=0.07)
    
    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
        edge_index2, edge_attr2 = self.aug(edge_index, edge_attr)
        z1 = self.model(x, edge_index, edge_attr)
        z2 = self.model(x, edge_index2, edge_attr2)
        loss = self.loss_fn(z1, z2)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: add CADGCL trainer with EBC and InfoNCE"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-8-report.md`