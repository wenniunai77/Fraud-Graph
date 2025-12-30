"""
Fusion Model 配置文件
融合图模型（GraphMAE）与表格无监督模型（IsolationForest/LOF/AutoEncoder）
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ==================== 数据列索引（与 graph_main 保持一致）====================
@dataclass
class ColumnIndex:
    """数据列索引配置"""
    uetr: int = 0
    payment_channel: int = 1
    debit_bic_code: int = 2
    bene_bic_code: int = 3
    evt_tran_stat_cde: int = 4
    instructed_currency: int = 5
    instructed_amount: int = 6
    payment_currency: int = 7
    payment_amount: int = 8
    credit_currency: int = 9
    credit_amount: int = 10
    txn_dt: int = 11
    tds_dt: int = 12
    mop: int = 13
    debit_account_masked: int = 14
    bene_account_masked: int = 15


# ==================== 预处理配置 ====================
@dataclass
class PreprocessConfig:
    """数据预处理配置"""
    data_path: str = ""
    output_dir: str = "./processed_data"
    
    col_idx: ColumnIndex = field(default_factory=ColumnIndex)
    src_col: int = 14  # debit_account_masked
    dst_col: int = 15  # bene_account_masked
    
    # 数值特征列（金额）
    numerical_cols: List[int] = field(default_factory=lambda: [6, 8, 10])
    # 类别特征列
    categorical_cols: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 9, 13])
    # 时间特征列
    time_cols: List[int] = field(default_factory=lambda: [11, 12])
    
    # 嵌入维度
    embedding_dim: int = 8
    
    # 采样设置
    use_full_dataset: bool = True
    sample_size: int = 500000
    random_seed: int = 42
    
    def get_output_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)
    
    def ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)


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
    """图模型（GraphMAE）配置"""
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
    
    residual: bool = False
    norm: Optional[str] = None
    activation: str = "prelu"
    
    mask_rate: float = 0.5
    replace_rate: float = 0.1
    drop_edge_rate: float = 0.0
    
    loss_fn: str = "sce"
    alpha_l: float = 2.0


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
    strategy: str = "gated"  # 修改: fusion_method -> strategy，与 fusion.py 保持一致
    
    # 门控融合参数：节点活跃度阈值（degree < threshold 时更依赖表格模型）
    degree_threshold: int = 3  # 修改: gated_degree_threshold -> degree_threshold
    alpha_high: float = 0.7    # 修改: gated_graph_weight_high -> alpha_high
    alpha_low: float = 0.3     # 修改: gated_graph_weight_low -> alpha_low
    use_hard_threshold: bool = False  # 新增: True=硬阈值二分类, False=平滑过渡(默认)
    
    # 加权融合参数
    fusion_alpha: float = 0.5  # 修改: weighted_graph_weight -> fusion_alpha
    
    # 一致性融合参数
    consistency_weight: float = 0.3  # 添加: 一致性融合权重
    consistent_threshold_percentile: float = 95.0
    
    # 边异常分数计算策略: "max", "mean", "sum"
    edge_score_strategy: str = "max"


# ==================== 评估配置 ====================
@dataclass
class EvaluationConfig:
    """无标签评估配置"""
    # Top-K 分析
    top_k: int = 1000  # 添加: 默认 Top-K 值
    top_k_values: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    
    # 稳定性评估
    stability_n_seeds: int = 5
    stability_k_values: List[int] = field(default_factory=lambda: [100, 500, 1000])  # 添加: 稳定性评估的 K 值列表
    stability_jaccard_k: int = 100
    
    # 弱规则定义（用于命中率评估）
    # 格式: [(feature_name, operator, threshold, description), ...]
    weak_rules: List[tuple] = field(default_factory=lambda: [
        ("payment_amount", ">", "p99", "极端大额"),
        ("payment_amount", "<", "p1", "极端小额"),
        ("time_diff_seconds", ">", "p99", "极端时延"),
        ("time_diff_seconds", "<", "p1", "极端短时延"),
    ])
    
    # 阈值校准
    threshold_percentiles: List[float] = field(default_factory=lambda: [90.0, 95.0, 99.0, 99.5])


# ==================== 主配置 ====================
@dataclass
class FusionMainConfig:
    """主配置"""
    # 路径配置
    data_path: str = "../graph_main/raw_data/xxx.csv"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    
    # 子配置 - 修复变量名以匹配 run_fusion.py 中的使用
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tabular_model: TabularModelConfig = field(default_factory=TabularModelConfig)  # 修改: tabular -> tabular_model
    graph_model: GraphModelConfig = field(default_factory=GraphModelConfig)         # 修改: graph -> graph_model
    train: TrainConfig = field(default_factory=TrainConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 运行配置
    device: int = 0  # GPU设备号，-1表示CPU
    seed: int = 42
    
    # 输出控制
    save_model: bool = True
    save_scores: bool = True
    visualize: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        self.preprocess.data_path = self.data_path
        self.preprocess.output_dir = os.path.join(self.output_dir, "processed_data")
    
    def get_output_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)
    
    def ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.preprocess.ensure_output_dir()
