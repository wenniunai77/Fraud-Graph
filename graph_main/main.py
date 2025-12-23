"""
GraphMAE 欺诈检测主程序

用于支付交易图数据的无监督异常检测

使用方法:
    python main.py --data_path /path/to/data.csv --output_dir ./output

功能:
    1. 从CSV加载支付交易数据
    2. 构建交易图结构
    3. 生成描述性统计
    4. 使用GraphMAE进行无监督特征学习
    5. 基于重建误差进行异常检测
    6. 输出可视化报告和异常结果
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, DataConfig, ModelConfig, TrainConfig, AnomalyConfig
from data_loader import DataLoader, load_fraud_graph_data
from statistics import GraphStatistics, generate_statistics_report
from models import GraphMAE, set_random_seed, create_optimizer, count_parameters
from trainer import Trainer, train_graphmae
from anomaly_detector import AnomalyDetector, UnsupervisedEvaluator, detect_anomalies
from visualization import Visualizer, create_visualizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GraphMAE Fraud Detection for Payment Transactions"
    )
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the CSV data file")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--sample_size", type=int, default=500000,
                       help="Number of transactions to sample (0 for full dataset)")
    
    # 图构建参数
    parser.add_argument("--src_col", type=int, default=14,
                       help="Column index for source node (debit account)")
    parser.add_argument("--dst_col", type=int, default=15,
                       help="Column index for destination node (beneficiary account)")
    
    # 模型参数
    parser.add_argument("--encoder_type", type=str, default="gat",
                       choices=["gat", "gcn"],
                       help="Encoder type")
    parser.add_argument("--decoder_type", type=str, default="gat",
                       choices=["gat", "gcn", "mlp"],
                       help="Decoder type")
    parser.add_argument("--hidden_channels", type=int, default=256,
                       help="Hidden layer dimension")
    parser.add_argument("--out_channels", type=int, default=128,
                       help="Output embedding dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of GNN layers")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="Number of attention heads (for GAT)")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=500,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--mask_rate", type=float, default=0.5,
                       help="Node masking rate")
    parser.add_argument("--replace_rate", type=float, default=0.1,
                       help="Random replacement rate")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    
    # 异常检测参数
    parser.add_argument("--anomaly_threshold", type=float, default=95.0,
                       help="Anomaly threshold percentile")
    parser.add_argument("--edge_score_strategy", type=str, default="max",
                       choices=["max", "mean", "sum"],
                       help="Edge anomaly score strategy")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (-1 for CPU)")
    parser.add_argument("--save_model", action="store_true",
                       help="Save trained model")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualizations")
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """从命令行参数创建配置对象"""
    data_config = DataConfig(
        data_path=args.data_path,
        src_col=args.src_col,
        dst_col=args.dst_col,
        use_full_dataset=(args.sample_size == 0),
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )
    
    model_config = ModelConfig(
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_rate=args.mask_rate,
        replace_rate=args.replace_rate,
        patience=args.patience,
        device=args.device,
        seeds=[args.seed],
        save_model=args.save_model
    )
    
    anomaly_config = AnomalyConfig(
        threshold_percentile=args.anomaly_threshold,
        edge_score_strategy=args.edge_score_strategy
    )
    
    return Config(
        data=data_config,
        model=model_config,
        train=train_config,
        anomaly=anomaly_config
    )


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置设备
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # 创建输出目录
    os.makedirs(config.data.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.data.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("GraphMAE Fraud Detection Pipeline")
    logger.info("=" * 80)
    
    # ========== Step 1: 数据加载 ==========
    logger.info("\n" + "=" * 40)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 40)
    
    data_loader = DataLoader(config.data)
    data_loader.load_csv()
    
    # ========== Step 2: 构建图 ==========
    logger.info("\n" + "=" * 40)
    logger.info("Step 2: Building Graph")
    logger.info("=" * 40)
    
    data = data_loader.build_pyg_data()
    num_features = data.x.shape[1]
    
    # ========== Step 3: 描述性统计 ==========
    logger.info("\n" + "=" * 40)
    logger.info("Step 3: Statistical Analysis")
    logger.info("=" * 40)
    
    stats = GraphStatistics(data, data_loader)
    report = stats.print_full_report()
    
    # 保存统计报告
    stats_path = os.path.join(run_dir, "statistics_report.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        json.dump(convert_to_serializable(report), f, indent=2, ensure_ascii=False)
    logger.info(f"Statistics report saved to {stats_path}")
    
    # ========== Step 4: 构建模型 ==========
    logger.info("\n" + "=" * 40)
    logger.info("Step 4: Building Model")
    logger.info("=" * 40)
    
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
        alpha_l=config.train.alpha_l,
        use_dgl=False
    ).to(device)
    
    logger.info(f"Model created:")
    logger.info(f"  - Encoder: {config.model.encoder_type.upper()}")
    logger.info(f"  - Decoder: {config.model.decoder_type.upper()}")
    logger.info(f"  - Parameters: {count_parameters(model):,}")
    
    # ========== Step 5: 训练 ==========
    logger.info("\n" + "=" * 40)
    logger.info("Step 5: Training")
    logger.info("=" * 40)
    
    trainer = Trainer(model, config.train, device)
    history = trainer.train(data, verbose=True)
    
    # 保存模型
    if config.train.save_model:
        model_path = os.path.join(run_dir, "graphmae_model.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    # ========== Step 6: 异常检测 ==========
    logger.info("\n" + "=" * 40)
    logger.info("Step 6: Anomaly Detection")
    logger.info("=" * 40)
    
    data = data.to(device)
    detector = AnomalyDetector(model, config.anomaly)
    
    # 计算节点异常分数
    node_scores = detector.compute_reconstruction_error(data)
    
    # 计算边异常分数
    edge_scores = detector.compute_edge_anomaly_scores(data)
    
    # 获取节点嵌入
    node_embeddings = detector.get_node_embeddings(data)
    
    # 评估报告
    evaluator = UnsupervisedEvaluator(detector)
    evaluator.print_report('edge')
    
    # 获取Top异常
    top_k = 100
    top_indices, top_scores = detector.get_top_anomalies(k=top_k, level='edge')
    
    logger.info(f"\nTop {top_k} anomalous transactions:")
    for i, (idx, score) in enumerate(zip(top_indices[:10], top_scores[:10])):
        logger.info(f"  {i+1}. Transaction {idx}: Score = {score:.6f}")
    
    # 保存异常结果
    results = {
        'node_scores': node_scores.tolist(),
        'edge_scores': edge_scores.tolist(),
        'top_anomaly_indices': top_indices.tolist(),
        'top_anomaly_scores': top_scores.tolist(),
        'training_history': {
            'losses': history['train_losses'],
            'best_loss': history['best_loss'],
            'epochs': history['epochs_trained']
        },
        'config': {
            'encoder_type': config.model.encoder_type,
            'decoder_type': config.model.decoder_type,
            'hidden_channels': config.model.hidden_channels,
            'out_channels': config.model.out_channels,
            'num_layers': config.model.num_layers,
            'mask_rate': config.train.mask_rate,
            'epochs': config.train.epochs,
            'lr': config.train.lr
        }
    }
    
    results_path = os.path.join(run_dir, "anomaly_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Anomaly results saved to {results_path}")
    
    # ========== Step 7: 可视化 ==========
    if args.visualize:
        logger.info("\n" + "=" * 40)
        logger.info("Step 7: Visualization")
        logger.info("=" * 40)
        
        visualizer = create_visualizer(run_dir)
        
        # 计算节点度数
        from torch_geometric.utils import degree
        node_degrees = degree(
            data.original_edge_index[0], 
            num_nodes=data.num_nodes
        ).cpu().numpy()
        
        # 综合报告
        visualizer.plot_comprehensive_report(
            train_losses=history['train_losses'],
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_degrees=node_degrees,
            title="GraphMAE Fraud Detection Report",
            save_path=os.path.join(run_dir, "comprehensive_report.png")
        )
        
        # t-SNE可视化
        visualizer.plot_embeddings_tsne(
            embeddings=node_embeddings,
            scores=node_scores,
            sample_size=3000,
            title="Node Embeddings (colored by anomaly score)",
            save_path=os.path.join(run_dir, "embeddings_tsne.png")
        )
        
        logger.info(f"Visualizations saved to {run_dir}")
    
    # ========== 总结 ==========
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Complete!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {run_dir}")
    logger.info(f"\nSummary:")
    logger.info(f"  - Nodes: {data.num_nodes:,}")
    logger.info(f"  - Edges: {data.original_edge_index.shape[1]:,}")
    logger.info(f"  - Training Loss: {history['best_loss']:.4f}")
    logger.info(f"  - Epochs Trained: {history['epochs_trained']}")
    logger.info(f"  - Detected Anomalies (95th pct): {(edge_scores > np.percentile(edge_scores, 95)).sum():,}")
    
    return results


if __name__ == "__main__":
    main()
