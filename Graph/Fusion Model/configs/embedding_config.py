"""
类别 Embedding 预训练配置
用于在预处理阶段训练类别特征的 embedding
"""
from dataclasses import dataclass


@dataclass
class EmbeddingPretrainConfig:
    """Embedding 预训练配置"""
    
    # ============ 基本配置 ============
    enable_pretrain: bool = True  # 是否启用预训练（False 则使用随机初始化）
    embedding_dim: int = 8  # embedding 维度
    
    # ============ 训练策略 ============
    # 注意: 当前版本仅实现了 "masked_attribute" 方法
    # TODO: "link_prediction" 和 "co_occurrence" 尚未实现
    pretrain_method: str = "masked_attribute"  # 预训练方法: "masked_attribute" (已实现), "link_prediction" (未实现), "co_occurrence" (未实现)
    
    # ============ Masked Attribute Modeling (已实现) ============
    mask_prob: float = 0.15  # 每个属性被 mask 的概率
    
    # ============ 训练超参数 ============
    batch_size: int = 512  # 训练批次大小
    num_epochs: int = 20  # 训练轮数
    learning_rate: float = 1e-3  # 学习率
    weight_decay: float = 1e-5  # L2 正则化
    
    # ============ 优化器和调度器 ============
    optimizer: str = "adam"  # 优化器类型
    scheduler: str = "cosine"  # 学习率调度器: "cosine", "step", "none"
    warmup_epochs: int = 2  # [部分实现] warmup 轮数（当前未使用）
    
    # ============ 负采样（用于 link_prediction） ============
    # 注意: 以下字段仅在 pretrain_method="link_prediction" 时使用（当前未实现该方法）
    num_neg_samples: int = 5  # [未实现] 每条正样本对应的负样本数
    
    # ============ 共现窗口（用于 co_occurrence） ============
    # 注意: 以下字段仅在 pretrain_method="co_occurrence" 时使用（当前未实现该方法）
    window_size: int = 3  # [未实现] 字段共现窗口大小（类似 word2vec）
    
    # ============ 验证与早停 ============
    validation_split: float = 0.1  # 验证集比例
    early_stopping_patience: int = 5  # 早停耐心值
    
    # ============ 保存路径 ============
    save_path: str = "./processed_data/pretrained_embeddings.pt"  # 预训练 embedding 权重保存路径
    
    # ============ 日志 ============
    log_interval: int = 100  # 日志打印间隔（多少 batch）
    
    def __post_init__(self):
        """参数验证"""
        # 移除对未实现方法的断言，改为警告
        if self.pretrain_method not in ["masked_attribute", "link_prediction", "co_occurrence"]:
            raise ValueError(f"不支持的预训练方法: {self.pretrain_method}")
        
        if self.pretrain_method != "masked_attribute":
            import logging
            logging.warning(
                f"预训练方法 '{self.pretrain_method}' 当前版本尚未实现，"
                f"将使用默认的 'masked_attribute' 方法"
            )
            self.pretrain_method = "masked_attribute"
        
        assert 0 < self.mask_prob < 1, "mask_prob 必须在 (0, 1) 范围内"
        assert 0 < self.validation_split < 1, "validation_split 必须在 (0, 1) 范围内"
