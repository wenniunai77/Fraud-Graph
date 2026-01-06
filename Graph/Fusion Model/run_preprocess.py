"""
预处理主脚本
负责数据加载、特征工程和图构建
输出预处理完成的图数据和表格特征
"""
import argparse
import logging
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import PreprocessConfig
from configs.embedding_config import EmbeddingPretrainConfig
from preprocess import DataLoader, FeatureEngineer, GraphBuilder
from preprocess.train_embeddings import EmbeddingPretrainer, load_pretrained_embeddings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class PreprocessPipeline:
    """预处理流水线"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        
        # 组件
        self.data_loader: Optional[DataLoader] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.graph_builder: Optional[GraphBuilder] = None
        
        # 数据
        self.df: Optional[pd.DataFrame] = None
        self.tabular_features: Optional[np.ndarray] = None
        self.tabular_feature_names: Optional[list] = None
        self.graph_data = None
        
        # 元信息
        self.meta_info: Dict[str, Any] = {
            "preprocess_time": None,
            "data_info": {},
            "graph_info": {},
            "tabular_info": {},
            "config": {}
        }
    
    def load_data(self, data_path: str):
        """加载数据"""
        logging.info("=" * 60)
        logging.info("步骤 1: 加载数据")
        logging.info("=" * 60)
        
        self.data_loader = DataLoader(self.config)
        self.df = self.data_loader.load_csv(data_path)
        
        self.meta_info["data_info"] = self.data_loader.meta_info
        logging.info(f"数据加载完成: {len(self.df):,} 行, {len(self.df.columns)} 列")
    
    def build_features(self):
        """构建特征"""
        logging.info("=" * 60)
        logging.info("步骤 2: 特征工程")
        logging.info("=" * 60)
        
        self.feature_engineer = FeatureEngineer(self.config)
        
        # 构建表格特征
        logging.info("构建表格特征...")
        self.tabular_features, self.tabular_feature_names = \
            self.feature_engineer.build_tabular_features(self.df)
        logging.info(f"表格特征: {self.tabular_features.shape}")
        
        self.meta_info["tabular_info"] = {
            "shape": list(self.tabular_features.shape),
            "feature_names": self.tabular_feature_names,
            "num_features": len(self.tabular_feature_names)
        }
    
    def pretrain_embeddings_if_needed(self):
        """如果需要，进行 embedding 预训练"""
        # 如果不使用预训练，直接跳过
        if not self.config.use_pretrained_embeddings:
            logging.info("未启用预训练 embedding，将使用随机初始化")
            return None
        
        pretrained_path = self.config.pretrained_embedding_path
        
        # 如果预训练权重已存在，直接加载
        if os.path.exists(pretrained_path):
            logging.info(f"加载现有预训练 embedding: {pretrained_path}")
            return load_pretrained_embeddings(pretrained_path)
        
        # 如果不存在但不允许自动训练，返回 None（将使用随机初始化）
        if not self.config.train_embeddings_if_not_exist:
            logging.warning(f"预训练 embedding 不存在: {pretrained_path}")
            logging.warning("未启用自动训练，将使用随机初始化")
            return None
        
        # 自动训练 embedding
        logging.info("=" * 60)
        logging.info("预训练 Embedding（自动触发）")
        logging.info("=" * 60)
        
        # 创建预训练配置
        pretrain_config = EmbeddingPretrainConfig()
        pretrain_config.embedding_dim = self.config.embedding_dim
        pretrain_config.save_path = pretrained_path
        
        # 创建列名映射
        col_idx = self.config.col_idx
        col_name_map = {
            col_idx.payment_channel: "payment_channel",
            col_idx.debit_bic_code: "debit_bic_code",
            col_idx.bene_bic_code: "bene_bic_code",
            col_idx.instructed_currency: "instructed_currency",
            col_idx.payment_currency: "payment_currency",
            col_idx.credit_currency: "credit_currency",
            col_idx.mop: "mop"
        }
        
        # 训练
        trainer = EmbeddingPretrainer(pretrain_config)
        trainer.train(
            df=self.df,
            categorical_cols=self.config.categorical_cols,
            col_name_map=col_name_map
        )
        
        # 加载训练好的权重
        return load_pretrained_embeddings(pretrained_path)
    
    def build_graph(self, pretrained_embeddings: Optional[Dict] = None):
        """构建图"""
        logging.info("=" * 60)
        logging.info("步骤 3: 图构建")
        logging.info("=" * 60)
        
        self.graph_builder = GraphBuilder(self.config)
        
        # 1. 创建节点映射
        node_map, num_nodes = self.graph_builder.create_node_mapping(self.df)
        logging.info(f"节点映射创建完成: {num_nodes:,} 个节点")
        
        # 2. 构建边索引
        edge_index = self.graph_builder.build_edge_index(self.df)
        logging.info(f"边索引构建完成: {edge_index.shape[1]:,} 条边")
        
        # 3. 构建边特征（传入预训练 embedding）
        edge_features, edge_feature_names = self.feature_engineer.build_edge_features(
            self.df, pretrained_embeddings
        )
        logging.info(f"边特征构建完成: {edge_features.shape}")
        
        # 4. 构建节点特征
        node_features, node_feature_names = self.feature_engineer.build_node_features(
            edge_features, edge_index, num_nodes, edge_feature_names
        )
        logging.info(f"节点特征构建完成: {node_features.shape}")
        
        # 5. 构建 PyG Data 对象
        self.graph_data = self.graph_builder.build_pyg_data(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            add_self_loop=self.config.add_self_loops
        )
        
        self.meta_info["graph_info"] = self.graph_builder.meta_info
        self.meta_info["graph_info"]["node_feature_names"] = node_feature_names
        self.meta_info["graph_info"]["edge_feature_names"] = edge_feature_names
        
        logging.info(f"图数据构建完成:")
        logging.info(f"  - 节点数: {self.graph_data.num_nodes:,}")
        logging.info(f"  - 边数: {self.graph_data.edge_index.shape[1]:,}")
        logging.info(f"  - 节点特征维度: {self.graph_data.x.shape[1]}")
    
    def save_results(self):
        """保存预处理结果"""
        logging.info("=" * 60)
        logging.info("步骤 4: 保存预处理结果")
        logging.info("=" * 60)
        
        import torch
        
        # 确保输出目录存在
        self.config.ensure_output_dir()
        output_dir = self.config.output_dir
        
        # 1. 保存图数据
        graph_path = self.config.get_graph_data_path()
        torch.save(self.graph_data, graph_path)
        logging.info(f"图数据已保存: {graph_path}")
        
        # 2. 保存表格特征
        if self.config.save_tabular_features:
            tabular_path = self.config.get_tabular_features_path()
            np.save(tabular_path, self.tabular_features)
            logging.info(f"表格特征已保存: {tabular_path}")
        
        # 3. 保存组件（可选）
        if self.config.save_components:
            # 节点特征
            torch.save(self.graph_data.x, 
                      os.path.join(output_dir, "node_features.pt"))
            # 边索引
            torch.save(self.graph_data.edge_index, 
                      os.path.join(output_dir, "edge_index.pt"))
            # 原始边索引（不含自环）
            torch.save(self.graph_data.original_edge_index, 
                      os.path.join(output_dir, "original_edge_index.pt"))
            # 边特征
            if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
                torch.save(self.graph_data.edge_attr, 
                          os.path.join(output_dir, "edge_features.pt"))
            logging.info("图组件已分别保存")
        
        # 4. 保存节点映射
        mapping_path = os.path.join(output_dir, "node_mapping.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                "node_map": self.graph_builder.node_map,
                "reverse_node_map": self.graph_builder.reverse_node_map,
                "num_nodes": self.graph_builder.num_nodes
            }, f)
        logging.info(f"节点映射已保存: {mapping_path}")
        
        # 5. 保存原始数据副本（用于评估）
        if self.config.save_raw_data:
            raw_data_path = self.config.get_raw_data_path()
            self.df.to_pickle(raw_data_path)
            logging.info(f"原始数据已保存: {raw_data_path}")
        
        # 6. 保存特征工程器（用于后续推理）
        fe_path = os.path.join(output_dir, "feature_engineer.pkl")
        with open(fe_path, 'wb') as f:
            pickle.dump({
                "numerical_scaler": self.feature_engineer.numerical_scaler,
                "time_scaler": self.feature_engineer.time_scaler,
                "embeddings": {k: v.state_dict() for k, v in self.feature_engineer.embeddings.items()},
                "category_mappings": self.feature_engineer.category_mappings,
                "tabular_feature_names": self.tabular_feature_names
            }, f)
        logging.info(f"特征工程器已保存: {fe_path}")
        
        # 7. 保存元信息
        self.meta_info["preprocess_time"] = datetime.now().isoformat()
        self.meta_info["config"] = {
            "data_path": self.config.data_path,
            "output_dir": self.config.output_dir,
            "src_col": self.config.src_col,
            "dst_col": self.config.dst_col,
            "numerical_cols": self.config.numerical_cols,
            "categorical_cols": self.config.categorical_cols,
            "time_cols": self.config.time_cols,
            "embedding_dim": self.config.embedding_dim,
            "use_full_dataset": self.config.use_full_dataset,
            "sample_size": self.config.sample_size,
            "random_seed": self.config.random_seed,
            "add_self_loops": self.config.add_self_loops
        }
        
        meta_path = self.config.get_meta_info_path()
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta_info, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"元信息已保存: {meta_path}")
        
        logging.info(f"\n所有预处理结果已保存到: {output_dir}")
        self._print_summary()
    
    def _print_summary(self):
        """打印预处理摘要"""
        logging.info("\n" + "=" * 60)
        logging.info("预处理摘要")
        logging.info("=" * 60)
        logging.info(f"数据行数: {len(self.df):,}")
        logging.info(f"数据列数: {len(self.df.columns)}")
        logging.info(f"图节点数: {self.graph_data.num_nodes:,}")
        logging.info(f"图边数: {self.graph_data.edge_index.shape[1]:,}")
        logging.info(f"节点特征维度: {self.graph_data.x.shape[1]}")
        logging.info(f"表格特征维度: {self.tabular_features.shape[1]}")
        logging.info(f"输出目录: {self.config.output_dir}")
        logging.info("=" * 60)
    
    def run(self, data_path: str):
        """运行完整预处理流水线"""
        start_time = datetime.now()
        logging.info(f"\n开始预处理流水线 - {start_time}")
        
        try:
            # 1. 加载数据
            self.load_data(data_path)
            
            # 2. 构建表格特征
            self.build_features()
            
            # 3. 预训练 embedding（如果需要）
            pretrained_embeddings = self.pretrain_embeddings_if_needed()
            
            # 4. 构建图（传入预训练 embedding）
            self.build_graph(pretrained_embeddings)
            
            # 5. 保存结果
            self.save_results()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"\n预处理完成. 总耗时: {duration:.1f} 秒")
            
            return True
            
        except Exception as e:
            logging.error(f"预处理失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据预处理 - 图构建与特征工程")
    
    parser.add_argument(
        "--data",
        type=str,
        default="../graph_main/raw_data/xxx.csv",
        help="输入数据路径 (CSV) (默认: ../graph_main/raw_data/xxx.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./processed_data",
        help="输出目录"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="采样大小（默认使用全量数据）"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--no-self-loops",
        action="store_true",
        help="不添加自环"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = PreprocessConfig()
    config.data_path = args.data
    config.output_dir = args.output
    config.random_seed = args.seed
    config.add_self_loops = not args.no_self_loops
    
    if args.sample_size:
        config.use_full_dataset = False
        config.sample_size = args.sample_size
    
    # 运行预处理
    pipeline = PreprocessPipeline(config)
    pipeline.run(args.data)


if __name__ == "__main__":
    main()
