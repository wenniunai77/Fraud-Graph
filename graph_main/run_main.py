"""
GraphMAE 主运行脚本 (Main Runner)
运行模型训练、异常检测和可视化

使用方法:
    # 先运行预处理
    python preprocess/run_preprocess.py --data_path /path/to/data.csv --output_dir ./preprocess/preprocessed_data
    
    # 再运行主程序
    python run_main.py --preprocessed_dir ./preprocess/preprocessed_data --output_dir ./output
"""

import os
import sys
import json
import argparse
import logging
import pickle
import numpy as np
import torch
from datetime import datetime

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MainConfig
from models import GraphMAE
from trainer import Trainer
from anomaly_detector import AnomalyDetector
from visualization import Visualizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_preprocessed_data(config: MainConfig):
    """
    加载预处理后的数据
    
    Args:
        config: 主配置对象
        
    Returns:
        (data, node_mapping, statistics)
    """
    logging.info("加载预处理数据...")
    
    # 检查预处理目录是否存在
    if not os.path.exists(config.preprocessed_dir):
        raise FileNotFoundError(
            f"预处理目录不存在: {config.preprocessed_dir}\n"
            f"请先运行: python preprocess/run_preprocess.py --data_path <your_data.csv>"
        )
    
    # 加载图数据
    graph_path = config.get_preprocessed_path(config.graph_data_file)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"图数据文件不存在: {graph_path}")
    
    data = torch.load(graph_path)
    logging.info(f"  - 图数据加载成功: {data.num_nodes} 节点, {data.edge_index.shape[1]} 边")
    
    # 加载节点映射
    mapping_path = config.get_preprocessed_path(config.node_mapping_file)
    if os.path.exists(mapping_path):
        with open(mapping_path, 'rb') as f:
            node_mapping = pickle.load(f)
        logging.info(f"  - 节点映射加载成功")
    else:
        node_mapping = None
        logging.warning(f"  - 节点映射文件不存在")
    
    # 加载统计信息
    stats_path = config.get_preprocessed_path(config.statistics_file)
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            statistics = json.load(f)
        logging.info(f"  - 统计信息加载成功")
    else:
        statistics = None
        logging.warning(f"  - 统计信息文件不存在")
    
    return data, node_mapping, statistics


def run_main(config: MainConfig):
    """
    运行主流程
    
    流程:
    1. 加载预处理数据
    2. 构建模型
    3. 训练模型
    4. 异常检测
    5. 可视化结果
    
    Args:
        config: 主配置对象
    """
    logging.info("=" * 80)
    logging.info("GraphMAE 主程序开始 (Main Program Start)")
    logging.info("=" * 80)
    
    start_time = datetime.now()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 确保输出目录存在
    config.ensure_dirs()
    
    # 设置设备
    if config.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.device}')
        logging.info(f"使用GPU: {torch.cuda.get_device_name(config.device)}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU")
    
    # ========================================
    # Step 1: 加载预处理数据
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 1: 加载预处理数据")
    logging.info("=" * 40)
    
    data, node_mapping, statistics = load_preprocessed_data(config)
    data = data.to(device)
    
    in_channels = data.x.shape[1]
    logging.info(f"输入特征维度: {in_channels}")
    
    # ========================================
    # Step 2: 构建模型
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 2: 构建模型")
    logging.info("=" * 40)
    
    model = GraphMAE(
        in_channels=in_channels,
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        encoder_type=config.model.encoder_type,
        decoder_type=config.model.decoder_type,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        decoder_layers=config.model.decoder_layers,
        dropout=config.model.dropout,
        mask_rate=config.model.mask_rate,
        replace_rate=config.model.replace_rate,
        loss_fn=config.model.loss_fn,
        alpha_l=config.model.alpha_l
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型构建完成:")
    logging.info(f"  - 总参数量: {total_params:,}")
    logging.info(f"  - 可训练参数: {trainable_params:,}")
    
    # ========================================
    # Step 3: 训练模型
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 3: 训练模型")
    logging.info("=" * 40)
    
    trainer = Trainer(model, config.train, device)
    history = trainer.train(data, verbose=config.verbose)
    
    logging.info(f"训练完成:")
    logging.info(f"  - 训练轮数: {history['epochs_trained']}")
    logging.info(f"  - 最终损失: {history['train_losses'][-1]:.6f}")
    logging.info(f"  - 最佳损失: {history['best_loss']:.6f}")
    
    # 保存模型
    if config.save_model:
        model_path = config.get_output_path("graphmae_model.pt")
        torch.save(model.state_dict(), model_path)
        logging.info(f"模型已保存: {model_path}")
    
    # ========================================
    # Step 4: 异常检测
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 4: 异常检测")
    logging.info("=" * 40)
    
    detector = AnomalyDetector(model, config.anomaly, device)
    
    # 计算节点异常分数
    node_scores = detector.compute_reconstruction_error(data)
    logging.info(f"节点异常分数: shape={node_scores.shape}")
    
    # 计算边异常分数
    edge_scores = detector.compute_edge_anomaly_scores(data)
    logging.info(f"边异常分数: shape={edge_scores.shape}")
    
    # 获取节点嵌入
    node_embeddings = detector.get_node_embeddings(data)
    logging.info(f"节点嵌入: shape={node_embeddings.shape}")
    
    # 异常统计
    threshold = np.percentile(edge_scores, config.anomaly.threshold_percentile)
    num_anomalies = (edge_scores > threshold).sum()
    logging.info(f"异常阈值 ({config.anomaly.threshold_percentile}th): {threshold:.6f}")
    logging.info(f"检测到异常边: {num_anomalies} ({num_anomalies/len(edge_scores)*100:.2f}%)")
    
    # Top异常
    top_k = 100
    top_indices, top_scores = detector.get_top_anomalies(k=top_k, level='edge')
    logging.info(f"\nTop {top_k} 最异常的交易:")
    for i in range(min(10, len(top_indices))):
        logging.info(f"  {i+1}. 交易 {top_indices[i]}: 分数 {top_scores[i]:.6f}")
    
    # 保存异常检测结果
    results = {
        "node_scores": node_scores.tolist(),
        "edge_scores": edge_scores.tolist(),
        "top_anomaly_indices": top_indices.tolist(),
        "top_anomaly_scores": top_scores.tolist(),
        "threshold": float(threshold),
        "num_anomalies": int(num_anomalies),
        "statistics": {
            "node_score_mean": float(np.mean(node_scores)),
            "node_score_std": float(np.std(node_scores)),
            "edge_score_mean": float(np.mean(edge_scores)),
            "edge_score_std": float(np.std(edge_scores))
        }
    }
    
    results_path = config.get_output_path("anomaly_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"异常检测结果已保存: {results_path}")
    
    # ========================================
    # Step 5: 可视化
    # ========================================
    if config.visualize:
        logging.info("\n" + "=" * 40)
        logging.info("Step 5: 可视化")
        logging.info("=" * 40)
        
        try:
            from torch_geometric.utils import degree
            node_degrees = degree(
                data.original_edge_index[0], 
                num_nodes=data.num_nodes
            ).cpu().numpy()
        except:
            node_degrees = np.ones(data.num_nodes)
        
        visualizer = Visualizer(config.output_dir)
        
        # 综合报告
        visualizer.plot_comprehensive_report(
            train_losses=history['train_losses'],
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_degrees=node_degrees,
            save_path=config.get_output_path("comprehensive_report.png")
        )
        
        # t-SNE可视化
        visualizer.plot_embeddings_tsne(
            embeddings=node_embeddings,
            scores=node_scores,
            sample_size=min(3000, len(node_scores)),
            save_path=config.get_output_path("embeddings_tsne.png")
        )
        
        logging.info("可视化完成")
    
    # ========================================
    # 完成
    # ========================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logging.info("\n" + "=" * 80)
    logging.info("主程序完成 (Main Program Complete)")
    logging.info("=" * 80)
    logging.info(f"总耗时: {duration:.2f} 秒")
    logging.info(f"输出目录: {config.output_dir}")
    
    return {
        "history": history,
        "node_scores": node_scores,
        "edge_scores": edge_scores,
        "node_embeddings": node_embeddings,
        "top_anomalies": (top_indices, top_scores)
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GraphMAE 主程序")
    
    # 路径参数
    parser.add_argument(
        "--preprocessed_dir", 
        type=str, 
        default="./preprocess/preprocessed_data",
        help="预处理数据目录"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output",
        help="输出目录"
    )
    
    # 模型参数
    parser.add_argument("--encoder_type", type=str, default="gat", help="编码器类型")
    parser.add_argument("--hidden_channels", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--out_channels", type=int, default=128, help="输出维度")
    parser.add_argument("--num_layers", type=int, default=2, help="GNN层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="掩码比例")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    
    # 其他参数
    parser.add_argument("--device", type=int, default=0, help="GPU设备ID，-1为CPU")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_visualize", action="store_true", help="禁用可视化")
    parser.add_argument("--no_save", action="store_true", help="禁用模型保存")
    
    args = parser.parse_args()
    
    # 创建配置
    config = MainConfig()
    config.preprocessed_dir = args.preprocessed_dir
    config.output_dir = args.output_dir
    config.device = args.device
    config.seed = args.seed
    config.visualize = not args.no_visualize
    config.save_model = not args.no_save
    
    # 模型配置
    config.model.encoder_type = args.encoder_type
    config.model.hidden_channels = args.hidden_channels
    config.model.out_channels = args.out_channels
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.mask_rate = args.mask_rate
    config.model.dropout = args.dropout
    
    # 训练配置
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    config.train.patience = args.patience
    
    # 运行
    run_main(config)


if __name__ == "__main__":
    main()
