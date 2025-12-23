"""
GraphMAE Fraud Detection - Main Module

项目结构:
- preprocess/: 数据预处理模块，将CSV数据转换为图结构
- main模块: 模型训练、异常检测、可视化

运行流程:
1. 先运行 preprocess/run_preprocess.py 进行数据预处理
2. 再运行 run_main.py 进行模型训练和异常检测
"""

__version__ = "2.0.0"

from .config import MainConfig, ModelConfig, TrainConfig, AnomalyConfig

__all__ = [
    'MainConfig',
    'ModelConfig',
    'TrainConfig', 
    'AnomalyConfig'
]
