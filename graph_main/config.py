"""
GraphMAE 主配置文件 (Main Config)
用于模型训练和异常检测的配置

注意: 数据预处理相关配置已移至 preprocess/config.py
运行流程: 先运行 preprocess/run_preprocess.py -> 再运行 run_main.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    
    # ========== 编码器配置 ==========
    encoder_type: str = "gat"      # 编码器类型: 'gat', 'gcn'
    decoder_type: str = "gat"      # 解码器类型: 'gat', 'gcn', 'mlp'
    
    # ========== 网络结构配置 ==========
    hidden_channels: int = 256     # 隐藏层维度
    out_channels: int = 128        # 输出嵌入维度
    num_layers: int = 2            # GNN层数
    decoder_layers: int = 1        # 解码器MLP层数
    
    # ========== GAT特定配置 ==========
    num_heads: int = 4             # 注意力头数
    num_out_heads: int = 1         # 输出层注意力头数
    concat_hidden: bool = False    # 是否拼接隐藏层
    
    # ========== 正则化配置 ==========
    dropout: float = 0.2           # Dropout率
    attn_drop: float = 0.1         # 注意力Dropout
    negative_slope: float = 0.2    # LeakyReLU负斜率
    
    # ========== 掩码配置 ==========
    mask_rate: float = 0.5         # 掩码比例
    replace_rate: float = 0.1      # 随机替换比例
    
    # ========== 损失函数配置 ==========
    loss_fn: str = "sce"           # 损失函数: 'sce', 'mse'
    alpha_l: float = 2.0           # SCE损失的alpha参数


@dataclass  
class TrainConfig:
    """训练配置"""
    
    # ========== 优化器配置 ==========
    optimizer: str = "adam"        # 优化器类型: 'adam', 'sgd', 'adamw'
    lr: float = 0.001              # 学习率
    weight_decay: float = 1e-5     # L2正则化权重
    
    # ========== 训练配置 ==========
    epochs: int = 500              # 最大训练轮数
    patience: int = 20             # 早停耐心值
    
    # ========== 学习率调度器 ==========
    use_scheduler: bool = True     # 是否使用学习率调度器
    scheduler: str = "plateau"      # 调度器类型: 'plateau', 'cosine', 'none'
    scheduler_patience: int = 10   # 调度器耐心值
    scheduler_factor: float = 0.5  # 学习率衰减因子
    
    # ========== 其他 ==========
    grad_clip: float = 1.0         # 梯度裁剪
    val_interval: int = 5          # 验证间隔
    log_interval: int = 10         # 日志间隔


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    
    # ========== 异常分数计算 ==========
    edge_score_strategy: str = "max"    # 边异常分数策略: 'max', 'mean', 'sum'
    num_samples: int = 10               # 节点异常分数采样次数
    
    # ========== 阈值设置 ==========
    threshold_percentile: float = 95.0  # 异常阈值百分位数
    
    # ========== Top-K分析 ==========
    top_k_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])


@dataclass
class MainConfig:
    """主配置"""
    
    # ========== 路径配置 ==========
    preprocessed_dir: str = "./preprocess/preprocessed_data"  # 预处理数据目录
    output_dir: str = "./output"                              # 输出目录
    checkpoint_dir: str = "./checkpoints"                     # 模型检查点目录
    
    # ========== 子配置 ==========
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    
    # ========== 设备配置 ==========
    device: int = 0                 # GPU设备ID，-1表示CPU
    seed: int = 42                  # 随机种子
    
    # ========== 输出控制 ==========
    save_model: bool = True         # 是否保存模型
    visualize: bool = True          # 是否生成可视化
    verbose: bool = True            # 是否详细输出
    
    # ========== 预处理文件名 ==========
    graph_data_file: str = "graph_data.pt"
    node_mapping_file: str = "node_mapping.pkl"
    statistics_file: str = "statistics.json"
    
    def get_preprocessed_path(self, filename: str) -> str:
        """获取预处理文件路径"""
        return os.path.join(self.preprocessed_dir, filename)
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件路径"""
        return os.path.join(self.output_dir, filename)
    
    def ensure_dirs(self):
        """确保所有目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def get_default_config() -> MainConfig:
    """获取默认配置"""
    return MainConfig()
