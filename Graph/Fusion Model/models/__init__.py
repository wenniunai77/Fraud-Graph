"""
模型模块
包含表格无监督模型和图自监督模型
"""
from .tabular import TabularAnomalyDetector, AutoEncoder
from .graph_model import GraphAnomalyDetector, GraphMAE

__all__ = [
    # 表格模型
    "TabularAnomalyDetector",
    "AutoEncoder",
    # 图模型
    "GraphAnomalyDetector",
    "GraphMAE"
]
