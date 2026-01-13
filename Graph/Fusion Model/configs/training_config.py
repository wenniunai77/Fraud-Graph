"""
训练配置
用于模型训练、融合和评估
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .base_config import ColumnIndex


# ==================== 表格模型配置 ====================
@dataclass
class TabularModelConfig:
    """表格无监督模型配置"""
    # 模型选择: "isolation_forest", "lof", "autoencoder", "ensemble"
    model_type: str = "ensemble"
    
    # Isolation Forest 参数
    if_n_estimators: int = 100
    if_contamination: float = 0.01
    if_max_samples: str = "auto"
    if_random_state: int = 42
    
    # LOF 参数
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.01
    lof_metric: str = "euclidean"
    
    # AutoEncoder 参数
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16, 32, 64])
    ae_dropout: float = 0.1
    ae_epochs: int = 50
    ae_batch_size: int = 256
    ae_lr: float = 0.001
    
    # 集成权重（ensemble时使用）
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])  # IF, LOF, AE


# ==================== 图模型配置 ====================
@dataclass
class GraphModelConfig:
    """图模型（GraphMAE）配置
    
    注意: 
    - encoder 固定使用 GAT（不支持切换）
    - loss 固定使用 SCE（不支持切换）
    """
    # decoder_type 已实现：支持 "gat" 和 "mlp"
    decoder_type: str = "mlp"  # 解码器类型: "gat" 或 "mlp" (推荐 mlp，速度更快)
    
    hidden_channels: int = 256
    out_channels: int = 128
    num_layers: int = 2
    decoder_layers: int = 1
    
    num_heads: int = 4
    
    dropout: float = 0.2
    attn_drop: float = 0.1
    negative_slope: float = 0.2
    
    residual: bool = False
    norm: Optional[str] = None  # 支持 "batch", "layer" 或 None
    activation: str = "prelu"
    
    mask_rate: float = 0.5
    replace_rate: float = 0.1
    drop_edge_rate: float = 0.0  # 边 dropout 率，训练时随机丢弃边以增强鲁棒性
    
    alpha_l: float = 2.0  # SCE loss 的 alpha 参数


# ==================== 训练配置 ====================
@dataclass
class TrainConfig:
    """训练配置"""
    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-5
    
    epochs: int = 300
    patience: int = 20
    
    use_scheduler: bool = True
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    grad_clip: float = 1.0
    val_interval: int = 5
    log_interval: int = 10


# ==================== 融合策略配置 ====================
@dataclass
class FusionConfig:
    """融合策略配置"""
    # 融合方法: "gated", "weighted", "rank", "max", "consistent"
    strategy: str = "gated"
    
    # 门控融合参数
    degree_threshold: int = 5  # 活跃度阈值（统一默认值）
    alpha_high: float = 0.7
    alpha_low: float = 0.3
    use_hard_threshold: bool = False
    sigmoid_steepness: float = 1.0  # sigmoid 平滑的陡峭程度（k值）

    # 门控融合的分数空间:
    # - "minmax": 传统 min-max 归一化后再门控（默认，保持历史行为）
    # - "rank": 先转为 rank/quantile (0~1) 再门控（对重尾分布更鲁棒）
    gated_score_space: str = "minmax"
    
    # 加权融合参数
    fusion_alpha: float = 0.5
    
    # 一致性融合参数
    consistency_weight: float = 0.3
    consistent_threshold_percentile: float = 95.0
    
    # 边异常分数计算策略: "max", "mean", "sum"
    edge_score_strategy: str = "max"


# ==================== 评估配置 ====================
@dataclass
class EvaluationConfig:
    """无标签评估配置"""
    # Top-K 分析
    top_k: int = 1000
    top_k_values: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    
    # 稳定性评估
    stability_n_seeds: int = 5
    stability_k_values: List[int] = field(default_factory=lambda: [100, 500, 1000])
    stability_jaccard_k: int = 100
    
    # 阈值校准
    threshold_percentiles: List[float] = field(default_factory=lambda: [90.0, 95.0, 99.0, 99.5])


# ==================== 训练主配置 ====================
@dataclass
class TrainingMainConfig:
    """训练主配置"""
    # 输入路径（预处理输出目录）
    processed_data_dir: str = "./processed_data"
    
    # 输出路径
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    
    # 列索引
    col_idx: ColumnIndex = field(default_factory=ColumnIndex)
    
    # 子配置
    tabular_model: TabularModelConfig = field(default_factory=TabularModelConfig)
    graph_model: GraphModelConfig = field(default_factory=GraphModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 运行配置
    device: int = 0  # GPU设备号，-1表示CPU
    seed: int = 42
    
    # 输出控制
    save_model: bool = True
    save_scores: bool = True
    verbose: bool = True
    
    def get_graph_data_path(self) -> str:
        """获取图数据路径"""
        return os.path.join(self.processed_data_dir, "graph_data.pt")
    
    def get_tabular_features_path(self) -> str:
        """获取表格特征路径"""
        return os.path.join(self.processed_data_dir, "tabular_features.npy")
    
    def get_raw_data_path(self) -> str:
        """获取原始数据路径"""
        return os.path.join(self.processed_data_dir, "raw_data.pkl")
    
    def get_meta_info_path(self) -> str:
        """获取元信息路径"""
        return os.path.join(self.processed_data_dir, "preprocess_meta.json")
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件路径"""
        return os.path.join(self.output_dir, filename)
    
    def ensure_dirs(self):
        """确保所有必要目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
