"""
图异常检测模型封装
封装 GraphMAE 用于异常检测

对齐 graph_main 实现:
- P0: 重构误差改为 Cosine (与训练 loss 一致) + 多次采样计算异常分数
- P1: 添加 Batch/Layer Norm + 残差连接
- 新增: MLP 解码器选项
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from configs import GraphModelConfig, TrainConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


# ==================== Loss 函数 ====================

def sce_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Scaled Cosine Error Loss"""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def cosine_error(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """计算 Cosine 重构误差 (与 SCE loss 对齐)"""
    x_norm = F.normalize(x, p=2, dim=-1)
    recon_norm = F.normalize(recon, p=2, dim=-1)
    cos_sim = (x_norm * recon_norm).sum(dim=-1)
    return 1 - cos_sim  # 误差 = 1 - 相似度


def drop_edge(edge_index: torch.Tensor, p: float = 0.0) -> torch.Tensor:
    """
    DropEdge: 随机丢弃边以增强模型鲁棒性
    
    Args:
        edge_index: 边索引 [2, num_edges]
        p: 丢弃概率，0 表示不丢弃
        
    Returns:
        处理后的边索引
    """
    if p <= 0.0:
        return edge_index
    
    num_edges = edge_index.size(1)
    # 生成保留掩码
    keep_mask = torch.rand(num_edges, device=edge_index.device) >= p
    
    # 确保至少保留一条边
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    
    return edge_index[:, keep_mask]


# ==================== 工具函数 ====================

def create_activation(activation: str) -> nn.Module:
    """创建激活函数"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.2)
    else:
        return nn.PReLU()


def create_norm(norm: Optional[str]) -> type:
    """创建归一化层类"""
    if norm == "batch" or norm == "batchnorm":
        return nn.BatchNorm1d
    elif norm == "layer" or norm == "layernorm":
        return nn.LayerNorm
    else:
        return nn.Identity


# ==================== MLP 解码器 ====================

class MLPDecoder(nn.Module):
    """MLP 解码器"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "prelu"
    ):
        super().__init__()
        
        if num_layers == 1:
            self.decoder = nn.Linear(in_channels, out_channels)
        else:
            layers = []
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(create_activation(activation))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(create_activation(activation))
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_channels, out_channels))
            self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ==================== GAT Encoder ====================

class GATEncoder(nn.Module):
    """GAT 编码器 (对齐 graph_main，支持 norm + residual)"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        attn_drop: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = False,
        norm: Optional[str] = None,
        activation: str = "prelu"
    ):
        super().__init__()
        
        from torch_geometric.nn import GATConv
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        NormClass = create_norm(norm)
        
        if num_layers == 1:
            # 单层
            self.convs.append(GATConv(
                in_channels, out_channels, heads=1,
                concat=False, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(NormClass(out_channels) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
        else:
            # 第一层
            self.convs.append(GATConv(
                in_channels, hidden_channels, heads=num_heads,
                concat=True, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(NormClass(hidden_channels * num_heads) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
            
            # 中间层
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(
                    hidden_channels * num_heads, hidden_channels, heads=num_heads,
                    concat=True, dropout=attn_drop,
                    negative_slope=negative_slope
                ))
                self.norms.append(NormClass(hidden_channels * num_heads) if norm else nn.Identity())
                self.activations.append(create_activation(activation))
            
            # 最后一层
            self.convs.append(GATConv(
                hidden_channels * num_heads, out_channels, heads=1,
                concat=False, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(NormClass(out_channels) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
        
        # 残差连接
        if residual:
            self.res_fc = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        hidden_list = []
        h = x
        
        for i in range(self.num_layers):
            # dropout 在前 (对齐 graph_main)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = self.activations[i](h)
            hidden_list.append(h)
        
        # 残差连接
        if self.residual:
            h = h + self.res_fc(x)
        
        if return_hidden:
            return h, hidden_list
        return h


# ==================== GraphMAE 核心实现 ====================

class GraphMAE(nn.Module):
    """
    Graph Masked Autoencoder
    
    对齐 graph_main 实现，支持:
    - GAT encoder (固定)
    - MLP/GAT decoder
    - Batch/Layer Normalization
    - 残差连接
    - DropEdge 数据增强
    - Cosine 重构误差 (SCE loss)
    - 多次采样异常分数计算
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        attn_drop: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = False,
        norm: Optional[str] = None,
        activation: str = "prelu",
        decoder_type: str = "mlp",  # "mlp" or "gat"
        mask_rate: float = 0.5,
        replace_rate: float = 0.1,
        drop_edge_rate: float = 0.0,  # DropEdge 概率
        alpha_l: float = 2.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.alpha_l = alpha_l
        self.decoder_type = decoder_type.lower()
        
        # Mask token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.xavier_uniform_(self.enc_mask_token)
        
        # Encoder (GAT with norm + residual)
        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            activation=activation
        )
        
        # Encoder to Decoder projection
        self.encoder_to_decoder = nn.Linear(out_channels, out_channels, bias=False)
        
        # Decoder
        if self.decoder_type == "mlp":
            self.decoder = MLPDecoder(
                in_channels=out_channels,
                hidden_channels=hidden_channels,
                out_channels=in_channels,
                num_layers=decoder_layers,
                dropout=dropout,
                activation=activation
            )
        else:
            # GAT decoder (单层)
            from torch_geometric.nn import GATConv
            self.decoder = GATConv(
                out_channels, in_channels, heads=1,
                concat=False, dropout=attn_drop,
                negative_slope=negative_slope
            )
        
        logging.info(f"GraphMAE 初始化: encoder=GAT, decoder={self.decoder_type}, "
                    f"norm={norm}, residual={residual}, drop_edge_rate={drop_edge_rate}")
    
    def encoding_mask_noise(
        self, 
        x: torch.Tensor, 
        mask_rate: float
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            
            if num_noise_nodes > 0:
                noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
                out_x[token_nodes] = 0.0
                out_x[noise_nodes] = x[noise_to_be_chosen]
            else:
                out_x[token_nodes] = 0.0
        else:
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        
        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        
        return out_x, (mask_nodes, keep_nodes)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(x, edge_index)
    
    def decode(self, z: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码"""
        rep = self.encoder_to_decoder(z)
        
        if self.decoder_type == "mlp":
            return self.decoder(rep)
        else:
            # GAT decoder 需要 edge_index
            return self.decoder(rep, edge_index)
    
    def forward(self, data, x: Optional[torch.Tensor] = None):
        """前向传播（训练）
        
        训练时会应用:
        1. DropEdge: 随机丢弃边以增强鲁棒性
        2. Mask: 随机遮盖节点特征
        """
        if x is None:
            x = data.x
        edge_index = data.edge_index
        
        # DropEdge: 训练时随机丢弃边
        if self.training and self.drop_edge_rate > 0:
            edge_index = drop_edge(edge_index, p=self.drop_edge_rate)
        
        # Mask
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_rate)
        
        # Encode
        enc_rep = self.encode(use_x, edge_index)
        
        # Decode
        recon = self.decode(enc_rep, edge_index)
        
        # 计算 loss（只在 mask 节点上）
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        
        loss = sce_loss(x_rec, x_init, alpha=self.alpha_l)
        
        return loss, {"mask_nodes": mask_nodes, "keep_nodes": keep_nodes}
    
    def get_embeddings(self, data) -> torch.Tensor:
        """获取节点嵌入"""
        self.eval()
        with torch.no_grad():
            z = self.encode(data.x, data.edge_index)
        return z
    
    def compute_reconstruction_error(self, data) -> torch.Tensor:
        """
        计算重构误差（使用 Cosine 误差，与训练 loss 对齐）
        
        P0 修复: 使用 Cosine 误差代替 MSE
        """
        self.eval()
        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            
            # Encode
            z = self.encode(x, edge_index)
            
            # Decode
            recon = self.decode(z, edge_index)
            
            # Cosine 重构误差 (与 SCE loss 对齐)
            error = cosine_error(x, recon)
        
        return error
    
    def compute_node_anomaly_score(
        self, 
        data, 
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        计算节点异常分数（多次采样 mask，对齐 graph_main）
        
        P0 修复: 多次随机 mask → 只统计被 mask 节点的重构误差 → 取平均
        这样更稳定，模拟训练时的 mask 机制
        """
        self.eval()
        
        x = data.x
        edge_index = data.edge_index
        num_nodes = x.shape[0]
        
        # 累积分数和计数
        score_sum = torch.zeros(num_nodes, device=x.device)
        count = torch.zeros(num_nodes, device=x.device)
        
        with torch.no_grad():
            for _ in range(num_samples):
                # 随机 mask
                masked_x, (mask_nodes, _) = self.encoding_mask_noise(x, self.mask_rate)
                
                # Encode
                enc_rep = self.encode(masked_x, edge_index)
                
                # Decode
                recon = self.decode(enc_rep, edge_index)
                
                # 计算 Cosine 误差
                error = cosine_error(x, recon)
                
                # 只累积 mask 节点的误差
                score_sum[mask_nodes] += error[mask_nodes]
                count[mask_nodes] += 1
        
        # 避免除零
        count = torch.clamp(count, min=1)
        avg_scores = score_sum / count
        
        return avg_scores


# ==================== 图异常检测器封装 ====================

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
            decoder_layers=getattr(self.config, 'decoder_layers', 2),
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            attn_drop=getattr(self.config, 'attn_drop', 0.1),
            negative_slope=getattr(self.config, 'negative_slope', 0.2),
            residual=getattr(self.config, 'residual', False),
            norm=getattr(self.config, 'norm', None),
            activation=getattr(self.config, 'activation', 'prelu'),
            decoder_type=getattr(self.config, 'decoder_type', 'mlp'),
            mask_rate=self.config.mask_rate,
            replace_rate=self.config.replace_rate,
            drop_edge_rate=getattr(self.config, 'drop_edge_rate', 0.0),
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
                patience=self.train_config.scheduler_patience
            )
            logging.info("学习率调度器已启用 (ReduceLROnPlateau)")
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logging.info(f"GraphMAE 模型已构建. 参数量: {param_count:,}")
    
    def train(self, data) -> List[float]:
        """
        训练模型
        
        Returns:
            List[float]: 每个epoch的训练损失
        """
        if self.model is None:
            self.build_model(data.x.shape[1])
        
        data = data.to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        
        logging.info(f"开始训练 GraphMAE (epochs={self.train_config.epochs})...")
        
        current_lr = self.optimizer.param_groups[0]['lr']
        logging.info(f"初始学习率: {current_lr:.6f}")
        
        for epoch in range(self.train_config.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            loss, _ = self.model(data)
            loss.backward()
            
            train_losses.append(loss.item())
            
            # 梯度裁剪
            if self.train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    logging.info(f"  Epoch {epoch+1}: 学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
            
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
        
        return train_losses
    
    def compute_node_scores(self, data, use_sampling: bool = True, num_samples: int = 10) -> np.ndarray:
        """
        计算节点异常分数
        
        Args:
            data: 图数据
            use_sampling: 是否使用多次采样 (P0 修复，默认启用)
            num_samples: 采样次数
        """
        data = data.to(self.device)
        
        if use_sampling:
            # P0: 多次采样计算分数 (对齐 graph_main)
            self.node_scores = self.model.compute_node_anomaly_score(
                data, num_samples=num_samples
            ).cpu().numpy()
        else:
            # 传统方式: 单次重构误差
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
    
    def predict_scores(
        self, 
        data, 
        level: str = "edge", 
        strategy: str = "max",
        use_sampling: bool = True,
        num_samples: int = 10
    ) -> np.ndarray:
        """预测异常分数"""
        # 先计算节点分数
        self.compute_node_scores(data, use_sampling=use_sampling, num_samples=num_samples)
        
        if level == "edge":
            scores = self.compute_edge_scores(data, strategy)
        else:
            scores = self.node_scores
        
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
