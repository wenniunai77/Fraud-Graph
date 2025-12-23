"""
损失函数模块
包含SCE损失和其他损失函数
"""

import torch
import torch.nn.functional as F


def sce_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 3) -> torch.Tensor:
    """
    Scaled Cosine Error (SCE) Loss
    
    GraphMAE使用SCE而非MSE，因为：
    1. 对异常值更鲁棒
    2. 避免了特征尺度的影响
    3. 训练更稳定
    
    Args:
        x: 预测值
        y: 目标值
        alpha: 缩放指数
    
    Returns:
        SCE损失值
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    # 余弦相似度
    cos_sim = (x * y).sum(dim=-1)
    
    # SCE: (1 - cos_sim)^alpha
    loss = (1 - cos_sim).pow_(alpha)
    
    return loss.mean()


def sig_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid损失函数
    
    Args:
        x: 预测值
        y: 目标值
    
    Returns:
        损失值
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    均方误差损失
    
    Args:
        x: 预测值
        y: 目标值
    
    Returns:
        MSE损失值
    """
    return F.mse_loss(x, y)


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    余弦距离损失
    
    Args:
        x: 预测值
        y: 目标值
    
    Returns:
        余弦距离损失值
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    cos_sim = (x * y).sum(dim=-1)
    loss = 1 - cos_sim
    
    return loss.mean()


class SCELoss(torch.nn.Module):
    """
    SCE损失的Module封装
    """
    def __init__(self, alpha: float = 2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sce_loss(pred, target, self.alpha)
