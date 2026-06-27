import torch
import torch.nn as nn

class BAGConv(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim):
        super().__init__()
        self.f_delta = nn.Linear(in_edge_dim, in_node_dim)
        self.linear = nn.Linear(in_node_dim * 2, out_dim)

    def forward(self, x, edge_index, edge_attr):
        # f_Δ(e_ij) projects edge to node space
        # Aggregate neighbors with projected edge features
        # Apply linear + ReLU
        projected = self.f_delta(edge_attr)
        
        # Aggregate neighbor features
        row, col = edge_index
        neighbor_features = x[col] * projected
        
        # Mean aggregation per node
        agg = torch.zeros_like(x)
        counts = torch.zeros(x.shape[0], 1, device=x.device)
        agg = agg.index_add(0, row, neighbor_features)
        counts = counts.index_add(0, row, torch.ones(row.shape[0], 1, device=x.device))
        agg = agg / counts.clamp(min=1)
        
        # Concatenate node features with aggregated neighbors
        h = torch.cat([x, agg], dim=1)
        
        # Apply linear + ReLU
        return torch.relu(self.linear(h))