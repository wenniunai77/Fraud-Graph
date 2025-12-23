"""
预处理主运行脚本
运行此脚本完成数据预处理，生成图数据供main部分使用

使用方法:
    python run_preprocess.py --data_path /path/to/data.csv --output_dir ./preprocessed_data
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PreprocessConfig
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from graph_builder import GraphBuilder
from statistics import GraphStatistics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


def run_preprocess(config: PreprocessConfig) -> dict:
    """
    运行完整预处理流程
    
    流程:
    1. 加载CSV数据
    2. 构建边特征（包含time_diff）
    3. 创建节点映射和边索引
    4. 聚合节点特征
    5. 构建PyG图数据
    6. 计算统计信息
    7. 保存所有数据和元信息
    
    Args:
        config: 预处理配置对象
        
    Returns:
        预处理元信息字典
    """
    logging.info("=" * 80)
    logging.info("开始数据预处理 (Preprocessing Start)")
    logging.info("=" * 80)
    
    start_time = datetime.now()
    
    # 确保输出目录存在
    config.ensure_output_dir()
    
    # 收集所有元信息（用于preprocess_check.ipynb检查）
    preprocess_meta = {
        "run_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "data_path": config.data_path,
            "output_dir": config.output_dir,
            "sample_size": config.sample_size,
            "use_full_dataset": config.use_full_dataset
        },
        "steps": {}
    }
    
    # ========================================
    # Step 1: 数据加载
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 1: 数据加载 (Data Loading)")
    logging.info("=" * 40)
    
    loader = DataLoader(config)
    df = loader.load_csv()
    loader.print_data_overview()
    
    preprocess_meta["steps"]["data_loading"] = loader.get_meta_info()
    
    # ========================================
    # Step 2: 特征工程 - 边特征
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 2: 特征工程 - 边特征 (Edge Features)")
    logging.info("=" * 40)
    
    feature_engineer = FeatureEngineer(config)
    edge_features, edge_feature_names = feature_engineer.build_edge_features(df)
    
    preprocess_meta["steps"]["edge_features"] = feature_engineer.get_meta_info()
    
    # ========================================
    # Step 3: 图构建 - 节点映射和边索引
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 3: 图构建 (Graph Construction)")
    logging.info("=" * 40)
    
    graph_builder = GraphBuilder(config)
    node_map, num_nodes = graph_builder.create_node_mapping(df)
    edge_index = graph_builder.build_edge_index(df)
    
    preprocess_meta["steps"]["graph_building"] = graph_builder.get_meta_info()
    
    # ========================================
    # Step 4: 特征工程 - 节点特征聚合
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 4: 节点特征聚合 (Node Feature Aggregation)")
    logging.info("=" * 40)
    
    node_features, node_feature_names = feature_engineer.build_node_features(
        edge_features, edge_index, num_nodes, df
    )
    
    preprocess_meta["steps"]["node_features"] = {
        "shape": list(node_features.shape),
        "feature_names": node_feature_names
    }
    
    # ========================================
    # Step 5: 构建PyG图数据对象
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 5: 构建PyG图数据 (Build PyG Data)")
    logging.info("=" * 40)
    
    data = graph_builder.build_pyg_data(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        add_self_loop=True
    )
    
    # ========================================
    # Step 6: 统计分析
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 6: 统计分析 (Statistics Analysis)")
    logging.info("=" * 40)
    
    stats_analyzer = GraphStatistics(config)
    statistics = stats_analyzer.compute_statistics(
        data, df, node_feature_names, edge_feature_names
    )
    stats_analyzer.print_report()
    
    preprocess_meta["steps"]["statistics"] = statistics
    
    # ========================================
    # Step 7: 保存数据
    # ========================================
    logging.info("\n" + "=" * 40)
    logging.info("Step 7: 保存数据 (Save Data)")
    logging.info("=" * 40)
    
    # 保存图数据
    graph_builder.save_graph_data(
        data, node_feature_names, edge_feature_names
    )
    
    # 保存统计信息
    stats_analyzer.save_statistics()
    
    # 保存预处理元信息（用于检查）
    end_time = datetime.now()
    preprocess_meta["duration_seconds"] = (end_time - start_time).total_seconds()
    preprocess_meta["success"] = True
    
    meta_path = os.path.join(config.output_dir, config.preprocess_meta_file)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(preprocess_meta, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"\n预处理元信息已保存到: {meta_path}")
    
    # ========================================
    # 完成
    # ========================================
    logging.info("\n" + "=" * 80)
    logging.info("预处理完成 (Preprocessing Complete)")
    logging.info("=" * 80)
    logging.info(f"耗时: {preprocess_meta['duration_seconds']:.2f} 秒")
    logging.info(f"输出目录: {config.output_dir}")
    logging.info(f"\n输出文件:")
    logging.info(f"  - {config.graph_data_file}: PyG图数据")
    logging.info(f"  - {config.node_features_file}: 节点特征")
    logging.info(f"  - {config.edge_features_file}: 边特征")
    logging.info(f"  - {config.edge_index_file}: 边索引")
    logging.info(f"  - {config.node_mapping_file}: 节点映射")
    logging.info(f"  - {config.statistics_file}: 统计信息")
    logging.info(f"  - {config.preprocess_meta_file}: 预处理元信息")
    
    logging.info("\n✅ 请在 preprocess_check.ipynb 中检查预处理结果!")
    
    return preprocess_meta


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GraphMAE 数据预处理")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="原始CSV数据文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./preprocessed_data",
        help="预处理输出目录 (default: ./preprocessed_data)"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=500000,
        help="采样大小，0表示使用全量数据 (default: 500000)"
    )
    parser.add_argument(
        "--src_col", 
        type=int, 
        default=14,
        help="源节点列索引（支付方账户）(default: 14)"
    )
    parser.add_argument(
        "--dst_col", 
        type=int, 
        default=15,
        help="目标节点列索引（收款方账户）(default: 15)"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = PreprocessConfig()
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.src_col = args.src_col
    config.dst_col = args.dst_col
    
    if args.sample_size == 0:
        config.use_full_dataset = True
    else:
        config.use_full_dataset = False
        config.sample_size = args.sample_size
    
    # 运行预处理
    run_preprocess(config)


if __name__ == "__main__":
    main()
