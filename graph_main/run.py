"""
简化的运行脚本
用于快速测试和运行GraphMAE欺诈检测

使用方法:
    python run.py --data_path /path/to/data.csv
"""

import os
import sys

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import torch

from config import Config, DataConfig, ModelConfig, TrainConfig, AnomalyConfig
from data_loader import DataLoader
from statistics import GraphStatistics
from models import GraphMAE, set_random_seed, count_parameters
from trainer import Trainer
from anomaly_detector import AnomalyDetector, UnsupervisedEvaluator
from visualization import Visualizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_fraud_detection(
    data_path: str,
    output_dir: str = "./output",
    sample_size: int = 500000,
    epochs: int = 300,
    device_id: int = 0
):
    """
    运行完整的欺诈检测流程
    
    Args:
        data_path: CSV数据文件路径
        output_dir: 输出目录
        sample_size: 采样大小（0表示全量数据）
        epochs: 训练轮数
        device_id: GPU设备ID（-1表示CPU）
    """
    # 设置随机种子
    set_random_seed(42)
    
    # 设置设备
    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # 创建配置
    config = Config()
    config.data.data_path = data_path
    config.data.sample_size = sample_size
    config.data.use_full_dataset = (sample_size == 0)
    config.data.output_dir = output_dir
    config.train.epochs = epochs
    config.train.device = device_id
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GraphMAE Fraud Detection")
    logger.info("=" * 60)
    
    # ===== 1. 加载数据 =====
    logger.info("\n[Step 1/6] Loading Data...")
    data_loader = DataLoader(config.data)
    data_loader.load_csv()
    
    # ===== 2. 构建图 =====
    logger.info("\n[Step 2/6] Building Graph...")
    data = data_loader.build_pyg_data()
    num_features = data.x.shape[1]
    
    # ===== 3. 统计分析 =====
    logger.info("\n[Step 3/6] Statistical Analysis...")
    stats = GraphStatistics(data, data_loader)
    stats.print_full_report()
    
    # ===== 4. 构建并训练模型 =====
    logger.info("\n[Step 4/6] Building and Training Model...")
    
    model = GraphMAE(
        in_channels=num_features,
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        encoder_type=config.model.encoder_type,
        decoder_type=config.model.decoder_type,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        mask_rate=config.train.mask_rate,
        replace_rate=config.train.replace_rate,
        loss_fn=config.train.loss_fn,
        use_dgl=False
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    trainer = Trainer(model, config.train, device)
    history = trainer.train(data, verbose=True)
    
    # ===== 5. 异常检测 =====
    logger.info("\n[Step 5/6] Anomaly Detection...")
    
    data = data.to(device)
    detector = AnomalyDetector(model, config.anomaly)
    
    node_scores = detector.compute_reconstruction_error(data)
    edge_scores = detector.compute_edge_anomaly_scores(data)
    
    evaluator = UnsupervisedEvaluator(detector)
    evaluator.print_report('edge')
    
    # 获取Top异常
    top_indices, top_scores = detector.get_top_anomalies(k=100, level='edge')
    
    logger.info("\nTop 10 Anomalous Transactions:")
    for i, (idx, score) in enumerate(zip(top_indices[:10], top_scores[:10])):
        logger.info(f"  {i+1}. Transaction {idx}: Score = {score:.6f}")
    
    # ===== 6. 可视化 =====
    logger.info("\n[Step 6/6] Generating Visualizations...")
    
    try:
        from torch_geometric.utils import degree
        node_degrees = degree(
            data.original_edge_index[0],
            num_nodes=data.num_nodes
        ).cpu().numpy()
        
        visualizer = Visualizer(output_dir)
        visualizer.plot_comprehensive_report(
            train_losses=history['train_losses'],
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_degrees=node_degrees,
            save_path=os.path.join(output_dir, "report.png")
        )
        logger.info(f"Visualization saved to {output_dir}/report.png")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    # ===== 完成 =====
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Nodes: {data.num_nodes:,}")
    logger.info(f"Edges: {data.original_edge_index.shape[1]:,}")
    logger.info(f"Best Loss: {history['best_loss']:.4f}")
    logger.info(f"Anomalies Detected (95th pct): {(edge_scores > np.percentile(edge_scores, 95)).sum():,}")
    
    return {
        'node_scores': node_scores,
        'edge_scores': edge_scores,
        'top_indices': top_indices,
        'top_scores': top_scores,
        'history': history
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GraphMAE Fraud Detection")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to CSV data file")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--sample_size", type=int, default=500000,
                       help="Sample size (0 for full data)")
    parser.add_argument("--epochs", type=int, default=300,
                       help="Training epochs")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (-1 for CPU)")
    
    args = parser.parse_args()
    
    run_fraud_detection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        epochs=args.epochs,
        device_id=args.device
    )
