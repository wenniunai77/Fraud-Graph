import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from functools import partial

from .encoder import create_encoder, create_decoder, MLPDecoder
from .loss_func import sce_loss, mse_loss


class GraphMAE(nn.Module):
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
        concat_hidden: bool = False
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
        
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.xavier_uniform_(self.enc_mask_token)
        
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
            encoding=True
        )
        
        if concat_hidden:
            if encoder_type == "gat" and num_layers > 1:
                dec_in_dim = hidden_channels * num_heads * (num_layers - 1) + out_channels
            elif encoder_type == "gcn":
                dec_in_dim = hidden_channels * (num_layers - 1) + out_channels if num_layers > 1 else out_channels
            else:
                dec_in_dim = out_channels * num_layers
        else:
            dec_in_dim = out_channels
        
        self.encoder_to_decoder = nn.Linear(dec_in_dim, out_channels, bias=False)
        
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
                activation=activation
            )
        
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
    
    def forward(
        self, 
        data, 
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        if x is None:
            x = data.x
        edge_index = data.edge_index
        
        masked_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_rate)
        
        if self.drop_edge_rate > 0:
            from .utils import drop_edge
            edge_index = drop_edge(edge_index, self.drop_edge_rate, x.shape[0])
        
        enc_rep, all_hidden = self.encoder(masked_x, edge_index, return_hidden=True)
        
        if self.concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        
        rep = self.encoder_to_decoder(enc_rep)
        
        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0
        
        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, edge_index)
        
        if recon.shape[1] != x.shape[1]:
            import logging
            logging.error(f"Decoder output dimension mismatch!")
            logging.error(f"  Expected (from x): {x.shape[1]}")
            logging.error(f"  Got (from decoder): {recon.shape[1]}")
            logging.error(f"  Encoder: {self._encoder_type}, Decoder: {self._decoder_type}")
            logging.error(f"  in_channels: {self.in_channels}, out_channels: {self.out_channels}")
            raise RuntimeError(f"Decoder output shape {recon.shape} doesn't match input shape {x.shape}")
        
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        
        if x_rec.shape != x_init.shape:
            import logging
            logging.error(f"Dimension mismatch! x_rec: {x_rec.shape}, x_init: {x_init.shape}")
            logging.error(f"Encoder type: {self._encoder_type}, Decoder type: {self._decoder_type}")
            logging.error(f"in_channels: {self.in_channels}, out_channels: {self.out_channels}")
            raise RuntimeError(f"Dimension mismatch in forward: x_rec {x_rec.shape} vs x_init {x_init.shape}")
        
        loss = self.criterion(x_rec, x_init)
        
        return loss, {"loss": loss.item()}
    
    def get_embeddings(
        self, 
        data, 
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if x is None:
                x = data.x
            enc_rep, all_hidden = self.encoder(x, data.edge_index, return_hidden=True)
            if self.concat_hidden:
                enc_rep = torch.cat(all_hidden, dim=1)
        
        return enc_rep
    
    def compute_reconstruction_error(
        self, 
        data, 
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if x is None:
                x = data.x
            edge_index = data.edge_index
            
            enc_rep, all_hidden = self.encoder(x, edge_index, return_hidden=True)
            if self.concat_hidden:
                enc_rep = torch.cat(all_hidden, dim=1)
            rep = self.encoder_to_decoder(enc_rep)
            
            if self._decoder_type in ("mlp", "linear"):
                recon = self.decoder(rep)
            else:
                recon = self.decoder(rep, edge_index)
            
            x_norm = F.normalize(x, p=2, dim=1)
            recon_norm = F.normalize(recon, p=2, dim=1)
            
            cos_sim = (x_norm * recon_norm).sum(dim=1)
            recon_error = 1 - cos_sim
            
            return recon_error
    
    def compute_node_anomaly_score(
        self, 
        data, 
        x: Optional[torch.Tensor] = None,
        num_samples: int = 10
    ) -> torch.Tensor:
        self.eval()
        
        if x is None:
            x = data.x
        edge_index = data.edge_index
        
        scores_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                masked_x, (mask_nodes, _) = self.encoding_mask_noise(x, self.mask_rate)
                
                enc_rep, all_hidden = self.encoder(masked_x, edge_index, return_hidden=True)
                if self.concat_hidden:
                    enc_rep = torch.cat(all_hidden, dim=1)
                
                rep = self.encoder_to_decoder(enc_rep)
                
                if self._decoder_type not in ("mlp", "linear"):
                    rep[mask_nodes] = 0
                
                if self._decoder_type in ("mlp", "linear"):
                    recon = self.decoder(rep)
                else:
                    recon = self.decoder(rep, edge_index)
                
                x_norm = F.normalize(x, p=2, dim=1)
                recon_norm = F.normalize(recon, p=2, dim=1)
                
                cos_sim = (x_norm * recon_norm).sum(dim=1)
                error = 1 - cos_sim
                
                scores = torch.zeros(x.shape[0], device=x.device)
                scores[mask_nodes] = error[mask_nodes]
                scores_list.append(scores)
            
            avg_scores = torch.stack(scores_list).mean(dim=0)
        
        return avg_scores
    
    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        from itertools import chain
        return chain(
            self.encoder_to_decoder.parameters(), 
            self.decoder.parameters()
        )


def build_model(config) -> GraphMAE:
    model_config = config.model
    
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
        mask_rate=model_config.mask_rate,
        replace_rate=model_config.replace_rate,
        drop_edge_rate=model_config.drop_edge_rate,
        loss_fn=model_config.loss_fn,
        alpha_l=model_config.alpha_l,
        concat_hidden=model_config.concat_hidden
    )
    
    return model
