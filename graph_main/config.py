"""
GraphMAE 欺诈检测配置文件
用于支付交易图数据的无监督异常检测

字段对应关系（按列索引）：
第0列: uetr - 交易唯一标识
第1列: payment_channel - 交易渠道
第2列: debit_bic_code - 支付方BIC码
第3列: bene_bic_code - 收款方BIC码
第4列: evt_tran_stat_cde - 支付状态码
第5列: instructed_currency - 客户指定币种
第6列: instructed_amount - 客户指定金额
第7列: payment_currency - 银行使用币种
第8列: payment_amount - 银行使用金额
第9列: credit_currency - 收款方接收币种
第10列: credit_amount - 收款方接收金额
第11列: txn_dt - 事件发生时间
第12列: tds_dt - 入库时间戳
第13列: mop - 付款方式
第14列: debit_account_masked - 支付方账户(masked)
第15列: bene_account_masked - 收款方账户(masked)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_path: str = ""
    
    # 列索引定义（不使用列名，使用索引）
    col_uetr: int = 0                    # 交易唯一标识
    col_payment_channel: int = 1         # 交易渠道
    col_debit_bic_code: int = 2          # 支付方BIC码
    col_bene_bic_code: int = 3           # 收款方BIC码
    col_evt_tran_stat_cde: int = 4       # 支付状态码
    col_instructed_currency: int = 5     # 客户指定币种
    col_instructed_amount: int = 6       # 客户指定金额
    col_payment_currency: int = 7        # 银行使用币种
    col_payment_amount: int = 8          # 银行使用金额
    col_credit_currency: int = 9         # 收款方接收币种
    col_credit_amount: int = 10          # 收款方接收金额
    col_txn_dt: int = 11                 # 事件发生时间
    col_tds_dt: int = 12                 # 入库时间戳
    col_mop: int = 13                    # 付款方式
    col_debit_account_masked: int = 14   # 支付方账户
    col_bene_account_masked: int = 15    # 收款方账户
    
    # 图构建配置
    # 源节点列（支付方账户）
    src_col: int = 14
    # 目标节点列（收款方账户）  
    dst_col: int = 15
    
    # 数值特征列索引
    numerical_cols: List[int] = field(default_factory=lambda: [6, 8, 10])  # 金额相关列
    
    # 类别特征列索引
    categorical_cols: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 9, 13])
    
    # 时间特征列索引
    time_cols: List[int] = field(default_factory=lambda: [11, 12])
    
    # 采样配置
    use_full_dataset: bool = False
    sample_size: int = 500000
    
    # 输出目录
    output_dir: str = "./output"


@dataclass
class ModelConfig:
    """模型配置"""
    # 编码器配置
    encoder_type: str = "gat"  # 'gat', 'gcn', 'gin'
    decoder_type: str = "gat"  # 'gat', 'gcn', 'mlp'
    
    # 隐藏层配置
    hidden_channels: int = 256
    out_channels: int = 128
    num_layers: int = 2
    
    # GAT特定配置
    num_heads: int = 4
    num_out_heads: int = 1
    concat_hidden: bool = False
    
    # 正则化配置
    dropout: float = 0.2
    attn_drop: float = 0.1
    negative_slope: float = 0.2
    
    # 残差连接和归一化
    residual: bool = False
    norm: Optional[str] = None  # 'layernorm', 'batchnorm', None
    
    # 激活函数
    activation: str = "prelu"


@dataclass
class TrainConfig:
    """训练配置"""
    # 训练参数
    epochs: int = 500
    lr: float = 0.001
    weight_decay: float = 1e-5
    
    # 掩码策略
    mask_rate: float = 0.5
    replace_rate: float = 0.1
    drop_edge_rate: float = 0.0
    
    # 损失函数
    loss_fn: str = "sce"  # 'sce', 'mse'
    alpha_l: float = 2.0  # SCE损失的alpha参数
    
    # 优化器
    optimizer: str = "adam"
    use_scheduler: bool = True
    
    # 早停
    patience: int = 20
    
    # 设备
    device: int = 0  # -1 for CPU, >= 0 for GPU
    
    # 随机种子
    seeds: List[int] = field(default_factory=lambda: [42])
    
    # 保存和加载
    save_model: bool = True
    load_model: bool = False
    checkpoint_path: str = "./checkpoints"
    
    # 日志
    logging: bool = True
    log_interval: int = 10


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    # 节点异常分数采样次数
    num_samples: int = 10
    
    # 边异常检测策略
    edge_score_strategy: str = "max"  # 'max', 'mean', 'sum'
    
    # 异常阈值（百分位数）
    threshold_percentile: float = 95.0
    
    # Top-K分析
    top_k_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])


@dataclass
class Config:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    
    def __post_init__(self):
        """确保输出目录存在"""
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.train.checkpoint_path, exist_ok=True)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def load_config_from_dict(config_dict: dict) -> Config:
    """从字典加载配置"""
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    train_config = TrainConfig(**config_dict.get('train', {}))
    anomaly_config = AnomalyConfig(**config_dict.get('anomaly', {}))
    
    return Config(
        data=data_config,
        model=model_config,
        train=train_config,
        anomaly=anomaly_config
    )
