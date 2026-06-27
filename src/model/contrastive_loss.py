import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=2)
        logits = similarity / self.temperature
        labels = torch.arange(u.size(0), device=u.device)
        loss = F.cross_entropy(logits, labels)
        return loss