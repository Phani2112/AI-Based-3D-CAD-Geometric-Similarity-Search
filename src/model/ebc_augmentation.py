import torch
from typing import Tuple


class EBCAugmentation:
    def __init__(self, p_edge: float = 0.1):
        self.p_edge = p_edge

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_edges = edge_index.size(1)
        num_remove = max(1, int(num_edges * self.p_edge))
        
        # Compute edge betweenness centrality using networkx
        import networkx as nx
        
        G = nx.Graph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        
        edge_betweenness = nx.edge_betweenness_centrality(G)
        
        # Get edges sorted by betweenness (highest first - these are bridges)
        sorted_edges = sorted(edge_betweenness.keys(), key=lambda e: edge_betweenness[e], reverse=True)
        
        # Determine which edges to keep
        edges_to_remove = sorted_edges[:num_remove]
        edges_to_keep = sorted_edges[num_remove:]
        
        # Create mapping for edges
        edge_set = set(tuple(e) for e in edges)
        keep_indices = []
        for i, edge in enumerate(edges):
            edge_tuple = tuple(edge)
            # Check if edge is in keep set (either direction)
            if edge_tuple in edges_to_keep or (edge_tuple[1], edge_tuple[0]) in edges_to_keep:
                keep_indices.append(i)
        
        if len(keep_indices) == 0:
            return edge_index, edge_attr
        
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        new_edge_index = edge_index[:, keep_indices]
        new_edge_attr = edge_attr[keep_indices]
        
        return new_edge_index, new_edge_attr

    def __call__(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(edge_index, edge_attr)