"""
GraphMAE: Graph Masked Autoencoder 核心模型

核心思想：
1. 特征掩码 (Feature Masking): 随机掩码部分节点的特征
2. 编码器 (Encoder): 使用GNN对掩码后的图进行编码
3. 解码器 (Decoder): 重建被掩码节点的原始特征
4. 异常检测: 使用重建误差作为异常分数

支持两种框架：
- PyTorch Geometric (PyG)
- Deep Graph Library (DGL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from functools import partial

from .encoder import create_encoder, create_decoder, MLPDecoder
from .loss_func import sce_loss, mse_loss


class GraphMAE(nn.Module):
    """
    GraphMAE: Graph Masked Autoencoder
    
    核心流程：
    1. 随机掩码部分节点的特征
    2. 使用可学习的[MASK]令牌替换被掩码的特征
    3. 通过GNN编码器获取节点嵌入
    4. 使用解码器重建被掩码节点的原始特征
    5. 计算重建损失（仅在被掩码节点上）
    
    异常检测原理：
    - 正常节点特征可以从邻居信息中被很好地重建
    - 欺诈节点的行为模式异常，难以被正确重建
    - 使用重建误差作为异常分数
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        encoder_type: str = "gat",
        decoder_type: str = "gat",
        num_layers: int = 2,
        num_heads: int = 4,
        num_out_heads: int = 1,
        dropout: float = 0.2,
        attn_drop: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = False,
        norm: Optional[str] = None,
        activation: str = "prelu",
        mask_rate: float = 0.5,
        replace_rate: float = 0.1,
        drop_edge_rate: float = 0.0,
        loss_fn: str = "sce",
        alpha_l: float = 2.0,
        concat_hidden: bool = False,
        use_dgl: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.concat_hidden = concat_hidden
        self.use_dgl = use_dgl
        
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        
        # 可学习的掩码令牌
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.xavier_uniform_(self.enc_mask_token)
        
        # 构建编码器
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_out_heads=num_out_heads,
            dropout=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            activation=activation,
            encoding=True,
            use_dgl=use_dgl
        )
        
        # 编码器到解码器的映射
        dec_in_dim = out_channels * num_layers if concat_hidden else out_channels
        self.encoder_to_decoder = nn.Linear(dec_in_dim, out_channels, bias=False)
        
        # 构建解码器
        if decoder_type in ("mlp", "linear"):
            self.decoder = MLPDecoder(
                in_channels=out_channels,
                hidden_channels=hidden_channels,
                out_channels=in_channels,
                num_layers=1 if decoder_type == "linear" else 2,
                dropout=dropout,
                activation=activation
            )
        else:
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                in_channels=out_channels,
                hidden_channels=hidden_channels,
                out_channels=in_channels,
                num_layers=1,
                num_heads=num_heads,
                num_out_heads=num_out_heads,
                dropout=dropout,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                activation=activation,
                use_dgl=use_dgl
            )
        
        # 设置损失函数
        if loss_fn == "sce":
            self.criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "mse":
            self.criterion = mse_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def encoding_mask_noise(
        self, 
        x: torch.Tensor, 
        mask_rate: float
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        对节点特征进行掩码
        
        策略：
        - mask_rate比例的节点被掩码
        - replace_rate比例的掩码节点使用随机特征替换
        - 其余掩码节点使用可学习的[MASK]令牌
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            mask_rate: 掩码比例
        
        Returns:
            masked_x: 掩码后的特征
            (mask_nodes, keep_nodes): 掩码节点和保留节点的索引
        """
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        
        # 选择要掩码的节点数量
        num_mask_nodes = int(mask_rate * num_nodes)
        
        # 掩码节点和保留节点索引
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        # 创建掩码后的特征
        out_x = x.clone()
        
        if self.replace_rate > 0:
            # 随机替换
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
        
        # 添加掩码令牌
        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        
        return out_x, (mask_nodes, keep_nodes)
    
    def forward(
        self, 
        data, 
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            data: 图数据（PyG Data或DGL Graph）
            x: 节点特征（可选，如果data中包含特征则不需要）
        
        Returns:
            loss: 损失值
            loss_dict: 损失字典
        """
        # 获取特征和边索引
        if self.use_dgl:
            if x is None:
                x = data.ndata['feat']
            g = data
        else:
            if x is None:
                x = data.x
            edge_index = data.edge_index
        
        # 掩码节点特征
        masked_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_rate)
        
        # 边丢弃（可选）
        if self.drop_edge_rate > 0 and not self.use_dgl:
            from .utils import drop_edge
            edge_index = drop_edge(edge_index, self.drop_edge_rate, x.shape[0])
        
        # 编码
        if self.use_dgl:
            enc_rep, all_hidden = self.encoder(g, masked_x, return_hidden=True)
        else:
            enc_rep, all_hidden = self.encoder(masked_x, edge_index, return_hidden=True)
        
        # 是否拼接所有隐藏层
        if self.concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        
        # 编码器到解码器的映射
        rep = self.encoder_to_decoder(enc_rep)
        
        # 重新掩码（re-masking）- 防止信息泄露
        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0
        
        # 解码
        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            if self.use_dgl:
                recon = self.decoder(g, rep)
            else:
                recon = self.decoder(rep, edge_index)
        
        # 只计算被掩码节点的损失
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        
        loss = self.criterion(x_rec, x_init)
        
        return loss, {"loss": loss.item()}
    
    def get_embeddings(
        self, 
        data, 
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取节点嵌入（推理模式，不进行掩码）
        
        Args:
            data: 图数据
            x: 节点特征（可选）
        
        Returns:
            节点嵌入
        """
        self.eval()
        with torch.no_grad():
            if self.use_dgl:
                if x is None:
                    x = data.ndata['feat']
                enc_rep = self.encoder(data, x)
            else:
                if x is None:
                    x = data.x
                enc_rep = self.encoder(x, data.edge_index)
        
        return enc_rep
    
    def compute_reconstruction_error(
        self, 
        data, 
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算每个节点的重建误差（用于异常检测）
        
        Args:
            data: 图数据
            x: 节点特征（可选）
        
        Returns:
            每个节点的重建误差
        """
        self.eval()
        with torch.no_grad():
            if self.use_dgl:
                if x is None:
                    x = data.ndata['feat']
                g = data
                enc_rep = self.encoder(g, x)
                rep = self.encoder_to_decoder(enc_rep)
                
                if self._decoder_type in ("mlp", "linear"):
                    recon = self.decoder(rep)
                else:
                    recon = self.decoder(g, rep)
            else:
                if x is None:
                    x = data.x
                edge_index = data.edge_index
                
                enc_rep = self.encoder(x, edge_index)
                rep = self.encoder_to_decoder(enc_rep)
                
                if self._decoder_type in ("mlp", "linear"):
                    recon = self.decoder(rep)
                else:
                    recon = self.decoder(rep, edge_index)
            
            # 计算每个节点的重建误差（余弦距离）
            x_norm = F.normalize(x, p=2, dim=1)
            recon_norm = F.normalize(recon, p=2, dim=1)
            
            # 余弦距离 = 1 - 余弦相似度
            cos_sim = (x_norm * recon_norm).sum(dim=1)
            recon_error = 1 - cos_sim
            
            return recon_error
    
    def compute_node_anomaly_score(
        self, 
        data, 
        x: Optional[torch.Tensor] = None,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        计算节点级异常分数
        使用多次采样取平均，提高稳定性
        
        Args:
            data: 图数据
            x: 节点特征（可选）
            num_samples: 采样次数
        
        Returns:
            每个节点的异常分数
        """
        self.eval()
        
        if self.use_dgl:
            if x is None:
                x = data.ndata['feat']
            g = data
        else:
            if x is None:
                x = data.x
            edge_index = data.edge_index
        
        scores_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # 随机掩码
                masked_x, (mask_nodes, _) = self.encoding_mask_noise(x, self.mask_rate)
                
                # 编码和解码
                if self.use_dgl:
                    enc_rep = self.encoder(g, masked_x)
                else:
                    enc_rep = self.encoder(masked_x, edge_index)
                
                rep = self.encoder_to_decoder(enc_rep)
                rep[mask_nodes] = 0
                
                if self._decoder_type in ("mlp", "linear"):
                    recon = self.decoder(rep)
                else:
                    if self.use_dgl:
                        recon = self.decoder(g, rep)
                    else:
                        recon = self.decoder(rep, edge_index)
                
                # 计算掩码节点的重建误差
                x_norm = F.normalize(x, p=2, dim=1)
                recon_norm = F.normalize(recon, p=2, dim=1)
                
                cos_sim = (x_norm * recon_norm).sum(dim=1)
                error = 1 - cos_sim
                
                # 创建完整的分数向量
                scores = torch.zeros(x.shape[0], device=x.device)
                scores[mask_nodes] = error[mask_nodes]
                scores_list.append(scores)
            
            # 取平均
            avg_scores = torch.stack(scores_list).mean(dim=0)
        
        return avg_scores
    
    @property
    def enc_params(self):
        """编码器参数"""
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        """解码器参数"""
        from itertools import chain
        return chain(
            self.encoder_to_decoder.parameters(), 
            self.decoder.parameters()
        )


def build_model(config) -> GraphMAE:
    """
    根据配置构建GraphMAE模型
    
    Args:
        config: 配置对象（包含model和train配置）
    
    Returns:
        GraphMAE模型实例
    """
    model_config = config.model
    train_config = config.train
    
    # 判断使用哪个框架
    try:
        import dgl
        use_dgl = True
    except ImportError:
        use_dgl = False
    
    model = GraphMAE(
        in_channels=config.num_features,
        hidden_channels=model_config.hidden_channels,
        out_channels=model_config.out_channels,
        encoder_type=model_config.encoder_type,
        decoder_type=model_config.decoder_type,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        num_out_heads=model_config.num_out_heads,
        dropout=model_config.dropout,
        attn_drop=model_config.attn_drop,
        negative_slope=model_config.negative_slope,
        residual=model_config.residual,
        norm=model_config.norm,
        activation=model_config.activation,
        mask_rate=train_config.mask_rate,
        replace_rate=train_config.replace_rate,
        drop_edge_rate=train_config.drop_edge_rate,
        loss_fn=train_config.loss_fn,
        alpha_l=train_config.alpha_l,
        concat_hidden=model_config.concat_hidden,
        use_dgl=use_dgl
    )
    
    return model
