import torch
from src.model.pooling import GlobalMeanPool

def test_global_mean_pooling():
    pool = GlobalMeanPool()
    x = torch.randn(10, 256)  # 10 nodes
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 3 graphs
    out = pool(x, batch)
    assert out.shape == (3, 256)  # 3 graph embeddings