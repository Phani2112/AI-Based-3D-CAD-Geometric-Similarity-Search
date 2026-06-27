import pytest
import torch
from src.model.gnn_encoder import BAGConv

def test_bagconv_forward():
    conv = BAGConv(in_node_dim=16, in_edge_dim=11, out_dim=256)
    # Mock input
    x = torch.randn(10, 16)  # 10 nodes
    edge_index = torch.tensor([[0, 1], [1, 0]])  # 2 edges
    edge_attr = torch.randn(2, 11)
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (10, 256)