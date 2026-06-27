import pytest
import torch
from src.training.trainer import CADGCLTrainer
from src.model.gnn_encoder import BAGConv


def test_trainer_step():
    model = BAGConv(16, 11, 256)
    trainer = CADGCLTrainer(model, lr=0.01)
    batch = {
        'x': torch.randn(10, 16),
        'edge_index': torch.randint(0, 10, (2, 20)),
        'edge_attr': torch.randn(20, 11)
    }
    loss = trainer.training_step(batch, 0)
    assert loss is not None
    assert loss.item() > 0