import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        negative_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        similarity = F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=2)
        logits = similarity / self.temperature
        if negative_weights is not None:
            weights = negative_weights.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-8)
            weights = weights.clone()
            weights.fill_diagonal_(1.0)
            logits = logits + weights.log()
        labels = torch.arange(u.size(0), device=u.device)
        loss = F.cross_entropy(logits, labels)
        return loss
