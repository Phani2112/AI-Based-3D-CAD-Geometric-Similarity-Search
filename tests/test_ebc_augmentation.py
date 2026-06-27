import pytest
import torch
from src.model.ebc_augmentation import EBCAugmentation

def test_ebc_augmentation():
    aug = EBCAugmentation(p_edge=0.1)
    edge_index = torch.tensor([[0,1,2], [1,2,0]])  # 3 edges
    edge_attr = torch.randn(3, 11)
    new_edge_index, new_edge_attr = aug(edge_index, edge_attr)
    # Remove edges with highest EBC (bridges) to remove ~10% bridges
    assert new_edge_index.shape[1] <= edge_index.shape[1]