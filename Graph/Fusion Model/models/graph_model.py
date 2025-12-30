"""
图异常检测模型封装
封装 GraphMAE 用于异常检测
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

import sys
sys.path.append('..')
from config import GraphModelConfig, TrainConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


# ==================== GraphMAE 核心实现 ====================
# 以下代码从 graph_main/models/ 复制并简化

def sce_loss(x, y, alpha=2.0):
    """Scaled Cosine Error Loss"""
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class GraphMAE(nn.Module):
    """Graph Masked Autoencoder"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        mask_rate: float = 0.5,
        replace_rate: float = 0.1,
        alpha_l: float = 2.0
    ):
        super().__init__()
        
        from torch_geometric.nn import GATConv
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - replace_rate
        self.alpha_l = alpha_l
        
        # Mask token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.xavier_uniform_(self.enc_mask_token)
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True)
        )
        for _ in range(num_layers - 2):
            self.encoder_layers.append(
                GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True)
            )
        if num_layers > 1:
            self.encoder_layers.append(
                GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False)
            )
        
        # Encoder to Decoder
        self.encoder_to_decoder = nn.Linear(out_channels, out_channels, bias=False)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encoding_mask_noise(self, x, mask_rate):
        """对节点特征添加 mask 噪声"""
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        out_x = x.clone()
        
        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[:int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        
        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        
        return out_x, (mask_nodes, keep_nodes)
    
    def encode(self, x, edge_index, edge_weight=None):
        """编码"""
        h = x
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, edge_index)
            if i < len(self.encoder_layers) - 1:
                h = torch.nn.functional.elu(h)
                h = self.dropout(h)
        return h
    
    def decode(self, z):
        """解码"""
        z = self.encoder_to_decoder(z)
        return self.decoder(z)
    
    def forward(self, data, x=None):
        """前向传播（训练）"""
        if x is None:
            x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        
        # Mask
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_rate)
        
        # Encode
        enc_rep = self.encode(use_x, edge_index, edge_weight)
        
        # Decode
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(rep)
        
        # 计算 loss（只在 mask 节点上）
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        
        loss = sce_loss(x_rec, x_init, alpha=self.alpha_l)
        
        return loss, {"mask_nodes": mask_nodes, "keep_nodes": keep_nodes}
    
    def get_embeddings(self, data):
        """获取节点嵌入"""
        self.eval()
        with torch.no_grad():
            z = self.encode(data.x, data.edge_index)
        return z
    
    def compute_reconstruction_error(self, data):
        """计算重构误差（用于异常检测）"""
        self.eval()
        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            
            # Encode
            z = self.encode(x, edge_index)
            
            # Decode
            recon = self.decode(z)
            
            # 计算每个节点的重构误差
            error = torch.mean((x - recon) ** 2, dim=1)
        
        return error


class GraphAnomalyDetector:
    """图异常检测器"""
    
    def __init__(
        self,
        config: GraphModelConfig,
        train_config: TrainConfig,
        device: str = "cpu"
    ):
        self.config = config
        self.train_config = train_config
        self.device = device
        
        self.model: Optional[GraphMAE] = None
        self.optimizer = None
        self.scheduler = None
        
        self.node_scores: Optional[np.ndarray] = None
        self.edge_scores: Optional[np.ndarray] = None
    
    def build_model(self, in_channels: int):
        """构建模型"""
        self.model = GraphMAE(
            in_channels=in_channels,
            hidden_channels=self.config.hidden_channels,
            out_channels=self.config.out_channels,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            mask_rate=self.config.mask_rate,
            replace_rate=self.config.replace_rate,
            alpha_l=self.config.alpha_l
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay
        )
        
        if self.train_config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.train_config.scheduler_factor,
                patience=self.train_config.scheduler_patience,
                verbose=True
            )
        
        logging.info(f"GraphMAE 模型已构建. 参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, data):
        """训练模型"""
        if self.model is None:
            self.build_model(data.x.shape[1])
        
        data = data.to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        logging.info(f"开始训练 GraphMAE (epochs={self.train_config.epochs})...")
        
        for epoch in range(self.train_config.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            loss, _ = self.model(data)
            loss.backward()
            
            # 梯度裁剪
            if self.train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.train_config.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % self.train_config.log_interval == 0:
                logging.info(f"  Epoch {epoch+1}/{self.train_config.epochs}, Loss: {loss.item():.6f}")
        
        logging.info(f"GraphMAE 训练完成. Best loss: {best_loss:.6f}")
    
    def compute_node_scores(self, data) -> np.ndarray:
        """计算节点异常分数（重构误差）"""
        data = data.to(self.device)
        self.node_scores = self.model.compute_reconstruction_error(data).cpu().numpy()
        return self.node_scores
    
    def compute_edge_scores(self, data, strategy: str = "max") -> np.ndarray:
        """计算边异常分数"""
        if self.node_scores is None:
            self.compute_node_scores(data)
        
        edge_index = data.original_edge_index.cpu().numpy() if hasattr(data, 'original_edge_index') else data.edge_index.cpu().numpy()
        
        src_indices = edge_index[0]
        dst_indices = edge_index[1]
        
        if strategy == 'max':
            self.edge_scores = np.maximum(
                self.node_scores[src_indices],
                self.node_scores[dst_indices]
            )
        elif strategy == 'mean':
            self.edge_scores = (
                self.node_scores[src_indices] + 
                self.node_scores[dst_indices]
            ) / 2
        elif strategy == 'sum':
            self.edge_scores = (
                self.node_scores[src_indices] + 
                self.node_scores[dst_indices]
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.edge_scores
    
    def predict_scores(self, data, level: str = "edge", strategy: str = "max") -> np.ndarray:
        """预测异常分数"""
        if level == "edge":
            scores = self.compute_edge_scores(data, strategy)
        else:
            scores = self.compute_node_scores(data)
        
        # 归一化
        return self._normalize_scores(scores)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到 [0, 1]"""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s > 1e-8:
            return (scores - min_s) / (max_s - min_s)
        return np.zeros_like(scores)
    
    def get_embeddings(self, data) -> np.ndarray:
        """获取节点嵌入"""
        data = data.to(self.device)
        return self.model.get_embeddings(data).cpu().numpy()
    
    def save(self, path: str):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "train_config": self.train_config
        }, path)
        logging.info(f"图模型已保存: {path}")
    
    def load(self, path: str, in_channels: int):
        """加载模型"""
        state = torch.load(path, map_location=self.device)
        
        self.config = state["config"]
        self.train_config = state["train_config"]
        
        self.build_model(in_channels)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        
        logging.info(f"图模型已加载: {path}")
