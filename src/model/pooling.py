import torch
from torch_geometric.nn import global_mean_pool

class GlobalMeanPool(torch.nn.Module):
    def forward(self, x, batch):
        return global_mean_pool(x, batch)