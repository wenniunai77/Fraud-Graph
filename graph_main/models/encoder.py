import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .utils import create_activation, create_norm

from torch_geometric.nn import GATConv, GCNConv, GINConv


class PyGGATEncoder(nn.Module):
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
        
        if num_layers == 1:
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(GATConv(
                in_channels, out_channels, heads=1,
                concat=False, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        else:
            self.convs.append(GATConv(
                in_channels, hidden_channels, heads=num_heads,
                concat=True, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(create_norm(norm)(hidden_channels * num_heads) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(
                    hidden_channels * num_heads, hidden_channels, heads=num_heads,
                    concat=True, dropout=attn_drop,
                    negative_slope=negative_slope
                ))
                self.norms.append(create_norm(norm)(hidden_channels * num_heads) if norm else nn.Identity())
                self.activations.append(create_activation(activation))
            
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(GATConv(
                hidden_channels * num_heads, out_channels, heads=1,
                concat=False, dropout=attn_drop,
                negative_slope=negative_slope
            ))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        
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
        
        if self.residual:
            h = h + self.res_fc(x)
        
        if return_hidden:
            return h, hidden_list
        return h


class PyGGCNEncoder(nn.Module):
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
        
        if num_layers == 1:
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(GCNConv(in_channels, out_channels))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.norms.append(create_norm(norm)(hidden_channels) if norm else nn.Identity())
            self.activations.append(create_activation(activation))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.norms.append(create_norm(norm)(hidden_channels) if norm else nn.Identity())
                self.activations.append(create_activation(activation))
            
            last_activation = create_activation(activation) if encoding else nn.Identity()
            self.convs.append(GCNConv(hidden_channels, out_channels))
            self.norms.append(create_norm(norm)(out_channels) if norm and encoding else nn.Identity())
            self.activations.append(last_activation)
        
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
        
        if self.residual:
            h = h + self.res_fc(x)
        
        if return_hidden:
            return h, hidden_list
        return h


class MLPDecoder(nn.Module):
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
    encoding: bool = True
):
    encoder_type = encoder_type.lower()
    
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
        raise ValueError(f"Unknown encoder type: {encoder_type}")


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
    activation: str = "prelu"
):
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
            num_layers=1,
            num_heads=num_heads, num_out_heads=num_out_heads,
            dropout=dropout, attn_drop=attn_drop,
            negative_slope=negative_slope, residual=residual,
            norm=norm, activation=activation, encoding=False
        )
