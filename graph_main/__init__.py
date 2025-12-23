"""
GraphMAE Fraud Detection Package

基于GraphMAE的支付交易欺诈检测系统
"""

__version__ = "1.0.0"
__author__ = "GraphMAE Fraud Detection Team"

from .config import Config, DataConfig, ModelConfig, TrainConfig, AnomalyConfig
from .data_loader import DataLoader, load_fraud_graph_data
from .statistics import GraphStatistics, generate_statistics_report
from .trainer import Trainer, train_graphmae
from .anomaly_detector import AnomalyDetector, UnsupervisedEvaluator, detect_anomalies
from .visualization import Visualizer, create_visualizer

__all__ = [
    'Config',
    'DataConfig', 
    'ModelConfig',
    'TrainConfig',
    'AnomalyConfig',
    'DataLoader',
    'load_fraud_graph_data',
    'GraphStatistics',
    'generate_statistics_report',
    'Trainer',
    'train_graphmae',
    'AnomalyDetector',
    'UnsupervisedEvaluator',
    'detect_anomalies',
    'Visualizer',
    'create_visualizer'
]
