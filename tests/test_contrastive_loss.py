import pytest
import torch
from src.model.contrastive_loss import InfoNCELoss

def test_infonce_loss():
    loss_fn = InfoNCELoss(temperature=0.07)
    u = torch.randn(4, 256)  # batch=4 embeddings
    v = torch.randn(4, 256)  # augmented views
    loss = loss_fn(u, v)
    assert loss.item() > 0
    assert loss.item() < 10  # reasonable loss range