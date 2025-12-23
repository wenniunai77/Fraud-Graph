"""
训练模块
实现GraphMAE的无监督训练流程
"""

import os
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

from config import MainConfig, TrainConfig
from models import GraphMAE, create_optimizer, get_current_lr, set_random_seed

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class Trainer:
    """
    GraphMAE训练器
    支持无监督训练和早停
    """
    
    def __init__(
        self,
        model: GraphMAE,
        config: TrainConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = create_optimizer(
            config.optimizer,
            model,
            config.lr,
            config.weight_decay
        )
        
        # 学习率调度器
        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.lr * 0.01
            )
        else:
            self.scheduler = None
        
        # 训练历史
        self.train_losses = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, data) -> float:
        """
        训练一个epoch
        
        Args:
            data: 图数据
        
        Returns:
            epoch平均损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        loss, loss_dict = self.model(data)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def train(
        self, 
        data,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            data: 图数据
            epochs: 训练轮数（可选，默认使用配置）
            verbose: 是否打印训练信息
        
        Returns:
            训练历史字典
        """
        epochs = epochs or self.config.epochs
        
        # 移动数据到设备
        data = data.to(self.device)
        
        logging.info(f"Starting training for {epochs} epochs...")
        logging.info(f"Device: {self.device}")
        
        epoch_iter = tqdm(range(epochs)) if verbose else range(epochs)
        
        for epoch in epoch_iter:
            # 训练一个epoch
            loss = self.train_epoch(data)
            self.train_losses.append(loss)
            
            # 更新进度条
            if verbose:
                epoch_iter.set_description(
                    f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | "
                    f"LR: {get_current_lr(self.optimizer):.6f}"
                )
            
            # 早停检查
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # 定期打印
            if not verbose and (epoch + 1) % self.config.log_interval == 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | "
                    f"LR: {get_current_lr(self.optimizer):.6f}"
                )
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logging.info(f"Loaded best model with loss: {self.best_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'epochs_trained': len(self.train_losses)
        }
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.best_loss = checkpoint['best_loss']
        logging.info(f"Checkpoint loaded from {path}")


def train_graphmae(
    model: GraphMAE,
    data,
    config: MainConfig,
    device: torch.device
) -> Tuple[GraphMAE, Dict]:
    """
    训练GraphMAE模型的便捷函数
    
    Args:
        model: GraphMAE模型
        data: 图数据
        config: 主配置对象
        device: 计算设备
    
    Returns:
        训练好的模型和训练历史
    """
    trainer = Trainer(model, config.train, device)
    history = trainer.train(data)
    
    # 保存模型
    if config.save_model:
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            'graphmae_checkpoint.pt'
        )
        trainer.save_checkpoint(checkpoint_path)
    
    return model, history
