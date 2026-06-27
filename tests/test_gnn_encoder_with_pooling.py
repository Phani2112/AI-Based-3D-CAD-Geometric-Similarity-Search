import torch
from src.model.gnn_encoder import BAGConvEncoder

def test_gnn_encoder_with_pooling():
    encoder = BAGConvEncoder()
    x = torch.randn(10, 16)  # 10 nodes
    edge_index = torch.tensor([[0,1,2], [1,2,0]])
    edge_attr = torch.randn(3, 11)
    batch = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    out = encoder(x, edge_index, edge_attr, batch)
    assert out.shape == (3, 256)