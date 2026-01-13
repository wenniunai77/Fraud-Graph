"""
预处理配置
用于数据预处理、特征工程和图构建
"""
import os
from dataclasses import dataclass, field
from typing import List

from .base_config import ColumnIndex


@dataclass
class PreprocessConfig:
    """数据预处理配置"""
    # 输入路径
    data_path: str = ""
    
    # 输出路径
    output_dir: str = "./processed_data"
    
    # 列索引配置
    col_idx: ColumnIndex = field(default_factory=ColumnIndex)
    
    # 源节点和目标节点列
    src_col: int = 14  # debit_account_masked
    dst_col: int = 15  # bene_account_masked
    
    # 数值特征列（金额）
    numerical_cols: List[int] = field(default_factory=lambda: [6, 8, 10])
    
    # 类别特征列
    categorical_cols: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 9, 13])
    
    # 时间特征列
    time_cols: List[int] = field(default_factory=lambda: [11, 12])
    
    # 嵌入维度配置（用于类别特征）
    embedding_dim: int = 8  # 默认/最小 embedding 维度
    use_adaptive_embedding_dim: bool = True  # 是否根据类别数量自适应调整维度
    embedding_dim_multiplier: float = 0.25  # 维度计算公式: min(max_dim, int(num_categories ** multiplier))
    max_embedding_dim: int = 32  # 最大 embedding 维度（避免高维稀疏）
    min_embedding_dim: int = 4   # 最小 embedding 维度
    
    # ============ Embedding 预训练配置 ============
    use_pretrained_embeddings: bool = True  # 是否使用预训练 embedding
    pretrained_embedding_path: str = "./processed_data/pretrained_embeddings.pt"  # 预训练权重路径
    train_embeddings_if_not_exist: bool = True  # 如果预训练权重不存在，是否自动训练
    
    # 随机种子
    random_seed: int = 42
    
    # 图构建选项
    add_self_loops: bool = True
    
    # 保存选项
    save_components: bool = True  # 是否分开保存各组件
    save_tabular_features: bool = True  # 是否保存表格特征
    save_raw_data: bool = True  # 是否保存原始数据副本
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件路径"""
        return os.path.join(self.output_dir, filename)
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_graph_data_path(self) -> str:
        """获取图数据路径"""
        return self.get_output_path("graph_data.pt")
    
    def get_tabular_features_path(self) -> str:
        """获取表格特征路径"""
        return self.get_output_path("tabular_features.npy")
    
    def get_meta_info_path(self) -> str:
        """获取元信息路径"""
        return self.get_output_path("preprocess_meta.json")
    
    def get_raw_data_path(self) -> str:
        """获取原始数据副本路径"""
        return self.get_output_path("raw_data.pkl")
