import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class BMMSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        sim_clamped = similarities.clamp(0.0, 1.0)
        weights = torch.zeros_like(sim_clamped)
        
        for i, s in enumerate(sim_clamped):
            dist1 = Beta(self.alpha1, self.beta1)
            dist2 = Beta(self.alpha2, self.beta2)
            
            mix_w = torch.sigmoid(self.mix_weight)
            log_prob = mix_w * dist1.log_prob(s) + (1 - mix_w) * dist2.log_prob(s)
            weights[i] = log_prob.exp().detach()
        
        return weights