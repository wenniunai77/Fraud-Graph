import os
import sys
import json
import argparse
import logging
from datetime import datetime

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
    logging.info("=" * 80)
    logging.info("Preprocessing Start")
    logging.info("=" * 80)
    
    start_time = datetime.now()
    config.ensure_output_dir()
    
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
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 1: Data Loading")
    logging.info("=" * 40)
    
    loader = DataLoader(config)
    df = loader.load_csv()
    loader.print_data_overview()
    
    preprocess_meta["steps"]["data_loading"] = loader.get_meta_info()
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 2: Feature Engineering - Edge Features")
    logging.info("=" * 40)
    
    feature_engineer = FeatureEngineer(config)
    edge_features, edge_feature_names = feature_engineer.build_edge_features(df)
    
    preprocess_meta["steps"]["edge_features"] = feature_engineer.get_meta_info()
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 3: Graph Construction")
    logging.info("=" * 40)
    
    graph_builder = GraphBuilder(config)
    node_map, num_nodes = graph_builder.create_node_mapping(df)
    edge_index = graph_builder.build_edge_index(df)
    
    preprocess_meta["steps"]["graph_building"] = graph_builder.get_meta_info()
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 4: Node Feature Aggregation")
    logging.info("=" * 40)
    
    node_features, node_feature_names = feature_engineer.build_node_features(
        edge_features, edge_index, num_nodes, df
    )
    
    preprocess_meta["steps"]["node_features"] = {
        "shape": list(node_features.shape),
        "feature_names": node_feature_names
    }
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 5: Build PyG Data")
    logging.info("=" * 40)
    
    data = graph_builder.build_pyg_data(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        add_self_loop=True
    )
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 6: Statistics Analysis")
    logging.info("=" * 40)
    
    stats_analyzer = GraphStatistics(config)
    statistics = stats_analyzer.compute_statistics(
        data, df, node_feature_names, edge_feature_names
    )
    stats_analyzer.print_report()
    
    preprocess_meta["steps"]["statistics"] = statistics
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 7: Save Data")
    logging.info("=" * 40)
    
    graph_builder.save_graph_data(
        data, node_feature_names, edge_feature_names
    )
    
    stats_analyzer.save_statistics()
    
    end_time = datetime.now()
    preprocess_meta["duration_seconds"] = (end_time - start_time).total_seconds()
    preprocess_meta["success"] = True
    
    meta_path = os.path.join(config.output_dir, config.preprocess_meta_file)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(preprocess_meta, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"\nPreprocess meta saved to: {meta_path}")
    
    logging.info("\n" + "=" * 80)
    logging.info("Preprocessing Complete")
    logging.info("=" * 80)
    logging.info(f"Duration: {preprocess_meta['duration_seconds']:.2f} seconds")
    logging.info(f"Output directory: {config.output_dir}")
    logging.info(f"\nOutput files:")
    logging.info(f"  - {config.graph_data_file}: PyG graph data")
    logging.info(f"  - {config.node_features_file}: Node features")
    logging.info(f"  - {config.edge_features_file}: Edge features")
    logging.info(f"  - {config.edge_index_file}: Edge index")
    logging.info(f"  - {config.node_mapping_file}: Node mapping")
    logging.info(f"  - {config.statistics_file}: Statistics")
    logging.info(f"  - {config.preprocess_meta_file}: Preprocess meta")
    
    logging.info("\nPlease check preprocessing results in preprocess_check.ipynb!")
    
    return preprocess_meta


def main():
    parser = argparse.ArgumentParser(description="GraphMAE Data Preprocessing")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to raw CSV data file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./preprocessed_data",
        help="Preprocessing output directory (default: ./preprocessed_data)"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=0,
        help="Sample size, 0 means full dataset (default: 0)"
    )
    parser.add_argument(
        "--src_col", 
        type=int, 
        default=14,
        help="Source node column index (payer account) (default: 14)"
    )
    parser.add_argument(
        "--dst_col", 
        type=int, 
        default=15,
        help="Target node column index (payee account) (default: 15)"
    )
    
    args = parser.parse_args()
    
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
    
    run_preprocess(config)


if __name__ == "__main__":
    main()
