import torch
import torch.nn.functional as F


def sce_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 3) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    cos_sim = (x * y).sum(dim=-1)
    
    loss = (1 - cos_sim).pow_(alpha)
    
    return loss.mean()


def sig_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    cos_sim = (x * y).sum(dim=-1)
    loss = 1 - cos_sim
    
    return loss.mean()


class SCELoss(torch.nn.Module):
    def __init__(self, alpha: float = 2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sce_loss(pred, target, self.alpha)
