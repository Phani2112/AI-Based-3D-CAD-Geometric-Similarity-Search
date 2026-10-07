import torch
import torch.nn as nn
from .pooling import GlobalMeanPool

class BAGConv(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim):
        super().__init__()
        self.f_delta = nn.Linear(in_edge_dim, in_node_dim)
        self.linear = nn.Linear(in_node_dim * 2, out_dim)

    def forward(self, x, edge_index, edge_attr):
        projected = self.f_delta(edge_attr)
        row, col = edge_index
        neighbor_features = x[col] * projected
        agg = torch.zeros_like(x)
        counts = torch.zeros(x.shape[0], 1, device=x.device)
        agg = agg.index_add(0, row, neighbor_features)
        counts = counts.index_add(0, row, torch.ones(row.shape[0], 1, device=x.device))
        agg = agg / counts.clamp(min=1)
        h = torch.cat([x, agg], dim=1)
        return torch.relu(self.linear(h))

class BAGConvEncoder(nn.Module):
    def __init__(self, in_node_dim=16, in_edge_dim=11, hidden_dim=256):
        super().__init__()
        self.conv1 = BAGConv(in_node_dim, in_edge_dim, hidden_dim)
        self.conv2 = BAGConv(hidden_dim, in_edge_dim, hidden_dim)
        self.pool = GlobalMeanPool()
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return self.pool(x, batch)

    def project(self, graph_embeddings):
        return self.projection_head(graph_embeddings)
