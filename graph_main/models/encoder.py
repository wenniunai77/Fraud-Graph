"""
GNN编码器模块
包含GAT和GCN编码器实现

支持两种框架：
- PyTorch Geometric (PyG)
- Deep Graph Library (DGL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .utils import create_activation, create_norm


# ============================================================================
# PyTorch Geometric 实现
# ============================================================================

try:
    from torch_geometric.nn import GATConv, GCNConv, GINConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class PyGGATEncoder(nn.Module):
    """
    基于PyTorch Geometric的GAT编码器
    """
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
        activation: str = "prelu",
        concat_out: bool = False,
        encoding: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoding = encoding
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(
            in_channels, hidden_channels, heads=num_heads,
            concat=True, dropout=attn_drop,
            negative_slope=negative_slope
        ))
        self.norms.append(create_norm(norm)(hidden_channels * num_heads) if norm else nn.Identity())
        self.activations.append(create_activation(activation))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * num_heads, hidden_channels, heads=num_heads,
                concat=True, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(create_norm(norm)(hidden_channels * num_heads) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
        
        # 最后一层
        if num_layers > 1:
            last_activation = create_activation(activation) if encoding else nn.Identity()
            # 最后一层不concat，输出维度为out_channels
            self.convs.append(GATConv(
                hidden_channels * num_heads, out_channels, heads=1,
                concat=False, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        
        # 残差连接
        self.residual = residual
        if residual:
            self.res_fc = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_hidden: bool = False) -> torch.Tensor:
        hidden_list = []
        h = x
        
        for i in range(self.num_layers):
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


class PyGGCNEncoder(nn.Module):
    """
    基于PyTorch Geometric的GCN编码器
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        residual: bool = False,
        norm: Optional[str] = None,
        activation: str = "prelu",
        encoding: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoding = encoding
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # 第一层
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(create_norm(norm)(hidden_channels) if norm else nn.Identity())
        self.activations.append(create_activation(activation))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(create_norm(norm)(hidden_channels) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
        
        # 最后一层
        if num_layers > 1:
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(GCNConv(hidden_channels, out_channels))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        
        # 残差连接
        self.residual = residual
        if residual:
            self.res_fc = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_hidden: bool = False) -> torch.Tensor:
        hidden_list = []
        h = x
        
        for i in range(self.num_layers):
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


# ============================================================================
# DGL 实现
# ============================================================================

try:
    import dgl
    from dgl.nn.pytorch import GATConv as DGLGATConv, GraphConv
    HAS_DGL = True
except ImportError:
    HAS_DGL = False


class DGLGATEncoder(nn.Module):
    """
    基于DGL的GAT编码器
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        num_out_heads: int = 1,
        dropout: float = 0.2,
        attn_drop: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = False,
        norm: Optional[str] = None,
        activation: str = "prelu",
        encoding: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoding = encoding
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # 第一层
        self.convs.append(DGLGATConv(
            in_channels, hidden_channels, num_heads=num_heads,
            feat_drop=dropout, attn_drop=attn_drop,
            negative_slope=negative_slope, residual=residual
        ))
        self.norms.append(create_norm(norm)(hidden_channels * num_heads) if norm else nn.Identity())
        self.activations.append(create_activation(activation))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(DGLGATConv(
                hidden_channels * num_heads, hidden_channels, num_heads=num_heads,
                feat_drop=dropout, attn_drop=attn_drop,
                negative_slope=negative_slope, residual=residual
            ))
            self.norms.append(create_norm(norm)(hidden_channels * num_heads) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
        
        # 最后一层
        if num_layers > 1:
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(DGLGATConv(
                hidden_channels * num_heads, out_channels, num_heads=num_out_heads,
                feat_drop=dropout, attn_drop=attn_drop,
                negative_slope=negative_slope, residual=residual
            ))
            self.norms.append(create_norm(norm)(out_channels * num_out_heads) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        
        self.head = nn.Identity()
    
    def forward(self, g, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        hidden_list = []
        h = x
        
        for i in range(self.num_layers):
            h = self.convs[i](g, h)
            h = h.flatten(1)  # 展平多头输出
            h = self.norms[i](h)
            h = self.activations[i](h)
            hidden_list.append(h)
        
        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)


class DGLGCNEncoder(nn.Module):
    """
    基于DGL的GCN编码器
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        residual: bool = False,
        norm: Optional[str] = None,
        activation: str = "prelu",
        encoding: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoding = encoding
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # 第一层
        self.convs.append(GraphConv(in_channels, hidden_channels))
        self.norms.append(create_norm(norm)(hidden_channels) if norm else nn.Identity())
        self.activations.append(create_activation(activation))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
            self.norms.append(create_norm(norm)(hidden_channels) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
        
        # 最后一层
        if num_layers > 1:
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(GraphConv(hidden_channels, out_channels))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        
        self.head = nn.Identity()
    
    def forward(self, g, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        hidden_list = []
        h = x
        
        for i in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.convs[i](g, h)
            h = self.norms[i](h)
            h = self.activations[i](h)
            hidden_list.append(h)
        
        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)


# ============================================================================
# MLP 解码器
# ============================================================================

class MLPDecoder(nn.Module):
    """
    MLP解码器
    用于重建被掩码节点的原始特征
    GraphMAE使用轻量级解码器，与重型编码器形成非对称架构
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
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


# ============================================================================
# 编码器工厂函数
# ============================================================================

def create_encoder(
    encoder_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 2,
    num_heads: int = 4,
    num_out_heads: int = 1,
    dropout: float = 0.2,
    attn_drop: float = 0.1,
    negative_slope: float = 0.2,
    residual: bool = False,
    norm: Optional[str] = None,
    activation: str = "prelu",
    encoding: bool = True,
    use_dgl: bool = False
):
    """
    创建编码器的工厂函数
    
    Args:
        encoder_type: 编码器类型 ('gat', 'gcn')
        in_channels: 输入特征维度
        hidden_channels: 隐藏层维度
        out_channels: 输出维度
        num_layers: 层数
        num_heads: 注意力头数（仅GAT）
        num_out_heads: 输出层注意力头数（仅GAT）
        dropout: Dropout比例
        attn_drop: 注意力Dropout比例（仅GAT）
        negative_slope: LeakyReLU斜率（仅GAT）
        residual: 是否使用残差连接
        norm: 归一化类型
        activation: 激活函数
        encoding: 是否为编码模式
        use_dgl: 是否使用DGL框架
    
    Returns:
        编码器模块
    """
    encoder_type = encoder_type.lower()
    
    if use_dgl:
        if not HAS_DGL:
            raise ImportError("DGL is not installed!")
        
        if encoder_type == "gat":
            return DGLGATEncoder(
                in_channels, hidden_channels, out_channels,
                num_layers=num_layers, num_heads=num_heads,
                num_out_heads=num_out_heads, dropout=dropout,
                attn_drop=attn_drop, negative_slope=negative_slope,
                residual=residual, norm=norm, activation=activation,
                encoding=encoding
            )
        elif encoder_type == "gcn":
            return DGLGCNEncoder(
                in_channels, hidden_channels, out_channels,
                num_layers=num_layers, dropout=dropout,
                residual=residual, norm=norm, activation=activation,
                encoding=encoding
            )
        else:
            raise ValueError(f"Unknown encoder type for DGL: {encoder_type}")
    else:
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is not installed!")
        
        if encoder_type == "gat":
            return PyGGATEncoder(
                in_channels, hidden_channels, out_channels,
                num_layers=num_layers, num_heads=num_heads,
                dropout=dropout, attn_drop=attn_drop,
                negative_slope=negative_slope, residual=residual,
                norm=norm, activation=activation, encoding=encoding
            )
        elif encoder_type == "gcn":
            return PyGGCNEncoder(
                in_channels, hidden_channels, out_channels,
                num_layers=num_layers, dropout=dropout,
                residual=residual, norm=norm, activation=activation,
                encoding=encoding
            )
        else:
            raise ValueError(f"Unknown encoder type for PyG: {encoder_type}")


def create_decoder(
    decoder_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 1,
    num_heads: int = 4,
    num_out_heads: int = 1,
    dropout: float = 0.2,
    attn_drop: float = 0.1,
    negative_slope: float = 0.2,
    residual: bool = False,
    norm: Optional[str] = None,
    activation: str = "prelu",
    use_dgl: bool = False
):
    """
    创建解码器的工厂函数
    
    Args:
        decoder_type: 解码器类型 ('gat', 'gcn', 'mlp', 'linear')
        其他参数同create_encoder
    
    Returns:
        解码器模块
    """
    decoder_type = decoder_type.lower()
    
    if decoder_type in ("mlp", "linear"):
        return MLPDecoder(
            in_channels, hidden_channels, out_channels,
            num_layers=1 if decoder_type == "linear" else num_layers,
            dropout=dropout, activation=activation
        )
    else:
        return create_encoder(
            decoder_type, in_channels, hidden_channels, out_channels,
            num_layers=1,  # 解码器通常只用1层
            num_heads=num_heads, num_out_heads=num_out_heads,
            dropout=dropout, attn_drop=attn_drop,
            negative_slope=negative_slope, residual=residual,
            norm=norm, activation=activation, encoding=False,
            use_dgl=use_dgl
        )
