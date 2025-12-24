import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    encoder_type: str = "gat"
    decoder_type: str = "gat"
    
    hidden_channels: int = 256
    out_channels: int = 128
    num_layers: int = 2
    decoder_layers: int = 1
    
    num_heads: int = 4
    num_out_heads: int = 1
    concat_hidden: bool = False
    
    dropout: float = 0.2
    attn_drop: float = 0.1
    negative_slope: float = 0.2
    
    mask_rate: float = 0.5
    replace_rate: float = 0.1
    
    loss_fn: str = "sce"
    alpha_l: float = 2.0


@dataclass  
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-5
    
    epochs: int = 500
    patience: int = 20
    
    use_scheduler: bool = True
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    grad_clip: float = 1.0
    val_interval: int = 5
    log_interval: int = 10


@dataclass
class AnomalyConfig:
    edge_score_strategy: str = "max"
    num_samples: int = 10
    
    threshold_percentile: float = 95.0
    
    top_k_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])


@dataclass
class MainConfig:
    preprocessed_dir: str = "./preprocess/preprocessed_data"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    
    device: int = 0
    seed: int = 42
    
    save_model: bool = True
    visualize: bool = True
    verbose: bool = True
    
    graph_data_file: str = "graph_data.pt"
    node_mapping_file: str = "node_mapping.pkl"
    statistics_file: str = "statistics.json"
    
    def get_preprocessed_path(self, filename: str) -> str:
        return os.path.join(self.preprocessed_dir, filename)
    
    def get_output_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)
    
    def ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def get_default_config() -> MainConfig:
    return MainConfig()
