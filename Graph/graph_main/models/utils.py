import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
from functools import partial


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None or name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError(f"Activation '{name}' is not implemented.")


def create_norm(name: str):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name is None or name == "none":
        return nn.Identity
    else:
        return nn.Identity


def create_optimizer(opt: str, model: nn.Module, lr: float, weight_decay: float):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer: {opt}")
    
    return optimizer


def get_current_lr(optimizer) -> float:
    return optimizer.state_dict()["param_groups"][0]["lr"]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mask_edge(num_edges: int, mask_prob: float, device: torch.device) -> torch.Tensor:
    mask_rates = torch.FloatTensor(np.ones(num_edges) * mask_prob).to(device)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(edge_index: torch.Tensor, drop_rate: float, num_nodes: int):
    if drop_rate <= 0:
        return edge_index
    
    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    
    new_edge_index = edge_index[:, keep_mask]
    
    return new_edge_index
