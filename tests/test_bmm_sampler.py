import torch
import pytest


def test_bmm_sampler():
    from src.model.bmm_sampler import BMMSampler
    
    sampler = BMMSampler()
    similarities = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.9])  # 5 negative samples
    weights = sampler(similarities)
    assert weights.shape == similarities.shape
    assert torch.all(weights >= 0)