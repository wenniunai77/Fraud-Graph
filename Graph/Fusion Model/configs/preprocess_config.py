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
    
    # 嵌入维度（用于类别特征）
    embedding_dim: int = 8
    
    # 采样设置
    use_full_dataset: bool = True
    sample_size: int = 500000
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
