import torch
import torch.nn as nn


class FeatureMasking(nn.Module):
    def __init__(self, p_mask: float = 0.3):
        super().__init__()
        self.p_mask = p_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_mask <= 0:
            return x
        keep = torch.rand(x.size(0), 1, device=x.device) > self.p_mask
        return x * keep.to(dtype=x.dtype)
