"""
分步骤运行GraphMAE欺诈检测
每个步骤都可以独立查看和调试

运行方式:
    python step_by_step_demo.py --data_path /path/to/data.csv
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def step1_load_data(config):
    """
    步骤1: 数据加载
    
    功能:
        - 从CSV读取数据
        - 检查数据维度和类型
    
    可查看内容:
        - data_loader.df: 原始DataFrame
        - 数据形状、缺失值等
    """
    logger.info("\n" + "="*80)
    logger.info("步骤1: 数据加载 (Data Loading)")
    logger.info("="*80)
    
    data_loader = DataLoader(config.data)
    data_loader.load_csv()
    
    # 可查看的中间状态
    df = data_loader.df
    logger.info(f"\n✓ 数据加载完成:")
    logger.info(f"  - 数据形状: {df.shape}")
    logger.info(f"  - 总交易数: {len(df):,}")
    logger.info(f"  - 列数: {df.shape[1]}")
    
    # 查看前几行数据
    logger.info(f"\n前3行数据:")
    print(df.head(3))
    
    # 检查缺失值
    logger.info(f"\n各列缺失值情况:")
    missing = df.isnull().sum()
    for i, count in enumerate(missing):
        if count > 0:
            logger.info(f"  第{i}列: {count} ({count/len(df)*100:.2f}%)")
    
    input("\n按回车继续到步骤2...")
    return data_loader


def step2_build_graph(data_loader):
    """
    步骤2: 图构建
    
    功能:
        - 创建节点映射
        - 构建边索引
        - 聚合节点特征
    
    可查看内容:
        - node_map: 节点ID映射
        - edge_index: 边索引
        - 节点特征矩阵
    """
    logger.info("\n" + "="*80)
    logger.info("步骤2: 图构建 (Graph Construction)")
    logger.info("="*80)
    
    # 2.1 创建节点映射
    logger.info("\n[2.1] 创建节点映射...")
    node_map, num_nodes = data_loader.create_node_mapping()
    logger.info(f"✓ 节点映射创建完成: {num_nodes:,} 个节点")
    
    # 查看部分映射
    logger.info(f"\n前5个节点映射:")
    for i, (account, idx) in enumerate(list(node_map.items())[:5]):
        logger.info(f"  账户 '{account}' -> 节点ID {idx}")
    
    # 2.2 构建边索引
    logger.info(f"\n[2.2] 构建边索引...")
    edge_index = data_loader.build_edge_index()
    logger.info(f"✓ 边索引创建完成:")
    logger.info(f"  - 形状: {edge_index.shape}")
    logger.info(f"  - 总边数: {edge_index.shape[1]:,}")
    
    # 查看前几条边
    logger.info(f"\n前5条边:")
    for i in range(min(5, edge_index.shape[1])):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        logger.info(f"  边 {i}: {src} -> {dst}")
    
    # 2.3 构建特征
    logger.info(f"\n[2.3] 构建边特征...")
    edge_features = data_loader.build_edge_features()
    logger.info(f"✓ 边特征创建完成:")
    logger.info(f"  - 形状: {edge_features.shape}")
    logger.info(f"  - 特征维度: {edge_features.shape[1]}")
    
    # 2.4 聚合节点特征
    logger.info(f"\n[2.4] 聚合节点特征...")
    node_features = data_loader.aggregate_node_features(edge_features, edge_index)
    logger.info(f"✓ 节点特征聚合完成:")
    logger.info(f"  - 形状: {node_features.shape}")
    logger.info(f"  - 特征维度: {node_features.shape[1]}")
    
    # 2.5 构建完整图
    logger.info(f"\n[2.5] 构建PyG数据对象...")
    data = data_loader.build_pyg_data()
    logger.info(f"✓ 图数据对象创建完成:")
    logger.info(f"  - 节点数: {data.num_nodes:,}")
    logger.info(f"  - 边数（含自环）: {data.edge_index.shape[1]:,}")
    logger.info(f"  - 原始边数: {data.original_edge_index.shape[1]:,}")
    logger.info(f"  - 节点特征维度: {data.x.shape[1]}")
    
    input("\n按回车继续到步骤3...")
    return data


def step3_statistics(data, data_loader):
    """
    步骤3: 统计分析
    
    功能:
        - 图结构统计
        - 度数分布
        - 特征统计
    
    可查看内容:
        - 详细统计报告
        - 保存的JSON文件
    """
    logger.info("\n" + "="*80)
    logger.info("步骤3: 统计分析 (Statistical Analysis)")
    logger.info("="*80)
    
    stats = GraphStatistics(data, data_loader)
    report = stats.print_full_report()
    
    # 保存统计报告
    output_dir = "./output/intermediate"
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, "statistics_step3.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
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
    
    logger.info(f"\n✓ 统计报告已保存到: {stats_file}")
    logger.info(f"  你可以用文本编辑器或JSON查看器打开此文件")
    
    input("\n按回车继续到步骤4...")
    return report


def step4_build_model(data, config):
    """
    步骤4: 构建模型
    
    功能:
        - 初始化GraphMAE模型
        - 设置优化器
    
    可查看内容:
        - 模型结构
        - 参数数量
        - 模型配置
    """
    logger.info("\n" + "="*80)
    logger.info("步骤4: 构建模型 (Model Building)")
    logger.info("="*80)
    
    # 设置设备
    if config.train.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.train.device}')
        logger.info(f"✓ 使用设备: GPU {torch.cuda.get_device_name(config.train.device)}")
    else:
        device = torch.device('cpu')
        logger.info(f"✓ 使用设备: CPU")
    
    # 构建模型
    logger.info(f"\n[4.1] 初始化GraphMAE模型...")
    num_features = data.x.shape[1]
    
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
    
    # 模型信息
    logger.info(f"✓ 模型创建完成:")
    logger.info(f"  - 编码器: {config.model.encoder_type.upper()}")
    logger.info(f"  - 解码器: {config.model.decoder_type.upper()}")
    logger.info(f"  - 输入维度: {num_features}")
    logger.info(f"  - 隐藏层维度: {config.model.hidden_channels}")
    logger.info(f"  - 输出维度: {config.model.out_channels}")
    logger.info(f"  - GNN层数: {config.model.num_layers}")
    logger.info(f"  - 注意力头数: {config.model.num_heads}")
    logger.info(f"  - 总参数量: {count_parameters(model):,}")
    
    # 显示模型结构
    logger.info(f"\n模型结构:")
    print(model)
    
    input("\n按回车继续到步骤5...")
    return model, device


def step5_train_model(model, data, config, device):
    """
    步骤5: 训练模型
    
    功能:
        - 无监督训练
        - 早停机制
    
    可查看内容:
        - 每个epoch的损失
        - 学习率变化
        - 训练曲线
    """
    logger.info("\n" + "="*80)
    logger.info("步骤5: 训练模型 (Model Training)")
    logger.info("="*80)
    
    logger.info(f"\n训练配置:")
    logger.info(f"  - 总轮数: {config.train.epochs}")
    logger.info(f"  - 学习率: {config.train.lr}")
    logger.info(f"  - 掩码比例: {config.train.mask_rate}")
    logger.info(f"  - 早停耐心值: {config.train.patience}")
    logger.info(f"  - 损失函数: {config.train.loss_fn.upper()}")
    
    # 创建训练器
    trainer = Trainer(model, config.train, device)
    
    # 训练
    logger.info(f"\n开始训练...")
    history = trainer.train(data, verbose=True)
    
    logger.info(f"\n✓ 训练完成:")
    logger.info(f"  - 最佳损失: {history['best_loss']:.6f}")
    logger.info(f"  - 训练轮数: {history['epochs_trained']}")
    logger.info(f"  - 最终损失: {history['train_losses'][-1]:.6f}")
    
    # 保存训练曲线数据
    output_dir = "./output/intermediate"
    history_file = os.path.join(output_dir, "training_history_step5.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n✓ 训练历史已保存到: {history_file}")
    
    input("\n按回车继续到步骤6...")
    return model, history


def step6_anomaly_detection(model, data, config, device):
    """
    步骤6: 异常检测
    
    功能:
        - 计算重建误差
        - 节点异常分数
        - 边异常分数
    
    可查看内容:
        - 异常分数分布
        - Top-K异常
        - 统计报告
    """
    logger.info("\n" + "="*80)
    logger.info("步骤6: 异常检测 (Anomaly Detection)")
    logger.info("="*80)
    
    data = data.to(device)
    detector = AnomalyDetector(model, config.anomaly)
    
    # 6.1 计算节点重建误差
    logger.info(f"\n[6.1] 计算节点重建误差...")
    node_scores = detector.compute_reconstruction_error(data)
    
    logger.info(f"✓ 节点异常分数:")
    logger.info(f"  - 数量: {len(node_scores):,}")
    logger.info(f"  - 均值: {np.mean(node_scores):.6f}")
    logger.info(f"  - 标准差: {np.std(node_scores):.6f}")
    logger.info(f"  - 最小值: {np.min(node_scores):.6f}")
    logger.info(f"  - 最大值: {np.max(node_scores):.6f}")
    logger.info(f"  - 95th分位数: {np.percentile(node_scores, 95):.6f}")
    
    # 6.2 计算边异常分数
    logger.info(f"\n[6.2] 计算边异常分数...")
    edge_scores = detector.compute_edge_anomaly_scores(data)
    
    logger.info(f"✓ 边异常分数:")
    logger.info(f"  - 数量: {len(edge_scores):,}")
    logger.info(f"  - 均值: {np.mean(edge_scores):.6f}")
    logger.info(f"  - 标准差: {np.std(edge_scores):.6f}")
    logger.info(f"  - 最小值: {np.min(edge_scores):.6f}")
    logger.info(f"  - 最大值: {np.max(edge_scores):.6f}")
    logger.info(f"  - 95th分位数: {np.percentile(edge_scores, 95):.6f}")
    
    # 6.3 获取节点嵌入
    logger.info(f"\n[6.3] 获取节点嵌入...")
    node_embeddings = detector.get_node_embeddings(data)
    logger.info(f"✓ 节点嵌入形状: {node_embeddings.shape}")
    
    # 6.4 获取Top异常
    logger.info(f"\n[6.4] 获取Top-100异常交易...")
    top_indices, top_scores = detector.get_top_anomalies(k=100, level='edge')
    
    logger.info(f"\nTop 10 最异常的交易:")
    for i, (idx, score) in enumerate(zip(top_indices[:10], top_scores[:10])):
        logger.info(f"  {i+1}. 交易索引 {idx}: 分数 = {score:.6f}")
    
    # 6.5 打印评估报告
    logger.info(f"\n[6.5] 生成评估报告...")
    evaluator = UnsupervisedEvaluator(detector)
    evaluator.print_report('edge')
    
    # 保存结果
    output_dir = "./output/intermediate"
    results = {
        'node_scores': node_scores.tolist(),
        'edge_scores': edge_scores.tolist(),
        'top_100_indices': top_indices.tolist(),
        'top_100_scores': top_scores.tolist(),
        'statistics': {
            'node': {
                'mean': float(np.mean(node_scores)),
                'std': float(np.std(node_scores)),
                'min': float(np.min(node_scores)),
                'max': float(np.max(node_scores)),
                'percentile_95': float(np.percentile(node_scores, 95))
            },
            'edge': {
                'mean': float(np.mean(edge_scores)),
                'std': float(np.std(edge_scores)),
                'min': float(np.min(edge_scores)),
                'max': float(np.max(edge_scores)),
                'percentile_95': float(np.percentile(edge_scores, 95))
            }
        }
    }
    
    results_file = os.path.join(output_dir, "anomaly_scores_step6.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ 异常检测结果已保存到: {results_file}")
    
    input("\n按回车继续到步骤7...")
    return detector, node_scores, edge_scores


def step7_visualization(detector, history, data, node_scores, edge_scores):
    """
    步骤7: 可视化
    
    功能:
        - 训练曲线
        - 分数分布
        - 嵌入可视化
    
    可查看内容:
        - 生成的图片文件
    """
    logger.info("\n" + "="*80)
    logger.info("步骤7: 可视化 (Visualization)")
    logger.info("="*80)
    
    output_dir = "./output/intermediate"
    visualizer = Visualizer(output_dir)
    
    try:
        # 7.1 训练损失曲线
        logger.info(f"\n[7.1] 绘制训练损失曲线...")
        visualizer.plot_training_loss(
            history['train_losses'],
            title="Training Loss Curve",
            save_path=os.path.join(output_dir, "training_loss.png")
        )
        logger.info(f"✓ 训练损失曲线已保存")
        
        # 7.2 异常分数分布
        logger.info(f"\n[7.2] 绘制异常分数分布...")
        visualizer.plot_score_distribution(
            edge_scores,
            title="Edge Anomaly Score Distribution",
            save_path=os.path.join(output_dir, "score_distribution.png")
        )
        logger.info(f"✓ 分数分布图已保存")
        
        # 7.3 Top异常
        logger.info(f"\n[7.3] 绘制Top异常...")
        top_indices, top_scores = detector.get_top_anomalies(k=50, level='edge')
        visualizer.plot_top_anomalies(
            top_indices,
            top_scores,
            title="Top 50 Anomalous Transactions",
            save_path=os.path.join(output_dir, "top_anomalies.png")
        )
        logger.info(f"✓ Top异常图已保存")
        
        # 7.4 节点度数vs分数
        logger.info(f"\n[7.4] 绘制度数vs分数关系...")
        from torch_geometric.utils import degree
        node_degrees = degree(
            data.original_edge_index[0],
            num_nodes=data.num_nodes
        ).cpu().numpy()
        
        visualizer.plot_node_degree_vs_score(
            node_degrees,
            node_scores,
            title="Node Degree vs Anomaly Score",
            save_path=os.path.join(output_dir, "degree_vs_score.png")
        )
        logger.info(f"✓ 度数vs分数图已保存")
        
        # 7.5 综合报告
        logger.info(f"\n[7.5] 生成综合报告...")
        visualizer.plot_comprehensive_report(
            train_losses=history['train_losses'],
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_degrees=node_degrees,
            title="GraphMAE Comprehensive Report",
            save_path=os.path.join(output_dir, "comprehensive_report.png")
        )
        logger.info(f"✓ 综合报告已保存")
        
        logger.info(f"\n✓ 所有可视化文件已保存到: {output_dir}")
        logger.info(f"  你可以打开这些PNG文件查看结果")
        
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        logger.info(f"跳过可视化步骤")
    
    input("\n按回车查看最终总结...")


def final_summary(data, history, node_scores, edge_scores):
    """
    最终总结
    """
    logger.info("\n" + "="*80)
    logger.info("最终总结 (Final Summary)")
    logger.info("="*80)
    
    logger.info(f"\n📊 数据统计:")
    logger.info(f"  - 节点数（账户）: {data.num_nodes:,}")
    logger.info(f"  - 边数（交易）: {data.original_edge_index.shape[1]:,}")
    logger.info(f"  - 节点特征维度: {data.x.shape[1]}")
    
    logger.info(f"\n🎯 训练结果:")
    logger.info(f"  - 训练轮数: {history['epochs_trained']}")
    logger.info(f"  - 最佳损失: {history['best_loss']:.6f}")
    logger.info(f"  - 最终损失: {history['train_losses'][-1]:.6f}")
    
    logger.info(f"\n🔍 异常检测结果:")
    threshold_95 = np.percentile(edge_scores, 95)
    threshold_99 = np.percentile(edge_scores, 99)
    anomalies_95 = (edge_scores > threshold_95).sum()
    anomalies_99 = (edge_scores > threshold_99).sum()
    
    logger.info(f"  - 总交易数: {len(edge_scores):,}")
    logger.info(f"  - 异常交易（95th分位数）: {anomalies_95:,} ({anomalies_95/len(edge_scores)*100:.2f}%)")
    logger.info(f"  - 异常交易（99th分位数）: {anomalies_99:,} ({anomalies_99/len(edge_scores)*100:.2f}%)")
    logger.info(f"  - 平均异常分数: {np.mean(edge_scores):.6f}")
    logger.info(f"  - 最高异常分数: {np.max(edge_scores):.6f}")
    
    logger.info(f"\n📁 生成的文件:")
    logger.info(f"  - ./output/intermediate/statistics_step3.json")
    logger.info(f"  - ./output/intermediate/training_history_step5.json")
    logger.info(f"  - ./output/intermediate/anomaly_scores_step6.json")
    logger.info(f"  - ./output/intermediate/*.png (可视化图片)")
    
    logger.info(f"\n" + "="*80)
    logger.info("所有步骤完成！")
    logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Step-by-step GraphMAE Demo")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to CSV data file")
    parser.add_argument("--sample_size", type=int, default=100000,
                       help="Sample size (smaller for demo)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs (smaller for demo)")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (-1 for CPU)")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(42)
    
    # 创建配置
    config = Config()
    config.data.data_path = args.data_path
    config.data.sample_size = args.sample_size
    config.data.use_full_dataset = False
    config.train.epochs = args.epochs
    config.train.device = args.device
    
    logger.info("="*80)
    logger.info("GraphMAE 欺诈检测 - 分步骤演示")
    logger.info("="*80)
    logger.info(f"\n配置:")
    logger.info(f"  - 数据文件: {args.data_path}")
    logger.info(f"  - 采样大小: {args.sample_size:,}")
    logger.info(f"  - 训练轮数: {args.epochs}")
    logger.info(f"  - 设备: {'GPU' if args.device >= 0 else 'CPU'}")
    
    input("\n按回车开始步骤1...")
    
    # 逐步执行
    data_loader = step1_load_data(config)
    data = step2_build_graph(data_loader)
    report = step3_statistics(data, data_loader)
    model, device = step4_build_model(data, config)
    model, history = step5_train_model(model, data, config, device)
    detector, node_scores, edge_scores = step6_anomaly_detection(model, data, config, device)
    step7_visualization(detector, history, data, node_scores, edge_scores)
    final_summary(data, history, node_scores, edge_scores)


if __name__ == "__main__":
    main()
