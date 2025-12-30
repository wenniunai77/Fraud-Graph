"""
Fusion Model 主运行脚本
完整实现图+表格融合异常检测流水线
"""
import argparse
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FusionMainConfig,
    PreprocessConfig,
    TabularModelConfig,
    GraphModelConfig,
    TrainConfig,
    FusionConfig,
    EvaluationConfig
)
from preprocess import DataLoader, FeatureEngineer, GraphBuilder
from models import TabularAnomalyDetector, GraphAnomalyDetector
from fusion import create_fusion_strategy, analyze_fusion, print_fusion_report
from evaluation import UnsupervisedEvaluator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class FusionPipeline:
    """融合检测流水线"""
    
    def __init__(self, config: FusionMainConfig):
        self.config = config
        self.preprocess_config = config.preprocess
        self.tabular_config = config.tabular_model
        self.graph_config = config.graph_model
        self.train_config = config.train
        self.fusion_config = config.fusion
        self.eval_config = config.evaluation
        
        # 组件
        self.data_loader: Optional[DataLoader] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.tabular_detector: Optional[TabularAnomalyDetector] = None
        self.graph_detector: Optional[GraphAnomalyDetector] = None
        self.fusion_strategy = None
        self.evaluator: Optional[UnsupervisedEvaluator] = None
        
        # 数据
        self.df: Optional[pd.DataFrame] = None
        self.tabular_features: Optional[np.ndarray] = None
        self.graph_data = None
        
        # 结果
        self.tabular_scores: Optional[np.ndarray] = None
        self.graph_scores: Optional[np.ndarray] = None
        self.fused_scores: Optional[np.ndarray] = None
        self.fusion_result = None
    
    def load_data(self, data_path: str):
        """加载数据"""
        logging.info("=" * 60)
        logging.info("步骤 1: 加载数据")
        logging.info("=" * 60)
        
        self.data_loader = DataLoader(self.preprocess_config)
        self.df = self.data_loader.load_csv(data_path)
        
        logging.info(f"数据加载完成: {len(self.df):,} 行, {len(self.df.columns)} 列")
    
    def preprocess(self):
        """预处理数据"""
        logging.info("=" * 60)
        logging.info("步骤 2: 预处理")
        logging.info("=" * 60)
        
        # 特征工程
        self.feature_engineer = FeatureEngineer(self.preprocess_config)
        
        # 构建表格特征
        self.tabular_features = self.feature_engineer.build_tabular_features(self.df)
        logging.info(f"表格特征: {self.tabular_features.shape}")
        
        # 构建图
        self.graph_builder = GraphBuilder(self.preprocess_config)
        self.graph_data = self.graph_builder.build_graph(self.df)
        logging.info(f"图数据: {self.graph_data.num_nodes} 节点, {self.graph_data.num_edges} 边")
    
    def train_tabular_model(self):
        """训练表格模型"""
        logging.info("=" * 60)
        logging.info("步骤 3: 训练表格模型")
        logging.info("=" * 60)
        
        self.tabular_detector = TabularAnomalyDetector(self.tabular_config)
        self.tabular_detector.fit(self.tabular_features)
        
        # 获取分数
        self.tabular_scores = self.tabular_detector.predict_fusion_score(
            self.tabular_features
        )
        
        logging.info(f"表格分数范围: [{self.tabular_scores.min():.4f}, {self.tabular_scores.max():.4f}]")
    
    def train_graph_model(self):
        """训练图模型"""
        logging.info("=" * 60)
        logging.info("步骤 4: 训练图模型")
        logging.info("=" * 60)
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用设备: {device}")
        
        self.graph_detector = GraphAnomalyDetector(
            self.graph_config,
            self.train_config,
            device=device
        )
        
        self.graph_detector.train(self.graph_data)
        
        # 获取边级别分数
        self.graph_scores = self.graph_detector.predict_scores(
            self.graph_data,
            level="edge",
            strategy="max"
        )
        
        logging.info(f"图分数范围: [{self.graph_scores.min():.4f}, {self.graph_scores.max():.4f}]")
    
    def fuse_scores(self):
        """融合分数"""
        logging.info("=" * 60)
        logging.info("步骤 5: 融合")
        logging.info("=" * 60)
        
        # 确保分数长度一致
        min_len = min(len(self.tabular_scores), len(self.graph_scores))
        tabular_scores = self.tabular_scores[:min_len]
        graph_scores = self.graph_scores[:min_len]
        
        # 获取节点度数（用于门控融合）
        node_degrees = None
        if hasattr(self.graph_data, 'edge_index'):
            edge_index = self.graph_data.edge_index.cpu().numpy()
            # 计算每条边对应的源节点度数
            from collections import Counter
            degree_dict = Counter(edge_index[0])
            src_degrees = np.array([degree_dict.get(i, 0) for i in edge_index[0]])
            node_degrees = src_degrees[:min_len]
        
        # 创建融合策略
        self.fusion_strategy = create_fusion_strategy(self.fusion_config)
        
        # 融合
        self.fusion_result = self.fusion_strategy.fuse(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            node_degrees=node_degrees
        )
        
        self.fused_scores = self.fusion_result.fused_scores
        
        # 分析融合结果
        report = analyze_fusion(self.fusion_result, top_k=self.eval_config.top_k)
        print_fusion_report(report)
    
    def evaluate(self):
        """评估"""
        logging.info("=" * 60)
        logging.info("步骤 6: 无标签评估")
        logging.info("=" * 60)
        
        self.evaluator = UnsupervisedEvaluator(self.eval_config)
        
        # 添加弱规则
        amount_col = self.preprocess_config.columns.payment_amount
        if amount_col in self.df.columns:
            # 大额交易规则
            for p in [95, 99]:
                threshold = np.percentile(self.df[amount_col], p)
                def make_rule(thresh, col):
                    def rule(data: pd.DataFrame) -> np.ndarray:
                        return (data[col] >= thresh).values
                    return rule
                self.evaluator.add_weak_rule(f"large_amount_p{p}", make_rule(threshold, amount_col))
        
        # 评估
        df_subset = self.df.iloc[:len(self.fused_scores)]
        report = self.evaluator.evaluate(
            df=df_subset,
            scores=self.fused_scores,
            top_k=self.eval_config.top_k
        )
        
        self.evaluator.print_report(report)
        
        return report
    
    def save_results(self, output_dir: str):
        """保存结果"""
        logging.info("=" * 60)
        logging.info("步骤 7: 保存结果")
        logging.info("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存分数
        results_df = pd.DataFrame({
            "graph_score": self.graph_scores[:len(self.fused_scores)],
            "tabular_score": self.tabular_scores[:len(self.fused_scores)],
            "fused_score": self.fused_scores
        })
        
        if self.fusion_result.fusion_weights is not None:
            results_df["fusion_weight"] = self.fusion_result.fusion_weights
        
        results_path = os.path.join(output_dir, "fusion_scores.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"分数已保存: {results_path}")
        
        # 保存 Top-K
        topk_idx = np.argsort(-self.fused_scores)[:self.eval_config.top_k]
        topk_df = self.df.iloc[topk_idx].copy()
        topk_df["fused_score"] = self.fused_scores[topk_idx]
        topk_df["rank"] = range(1, len(topk_idx) + 1)
        
        topk_path = os.path.join(output_dir, f"top_{self.eval_config.top_k}_anomalies.csv")
        topk_df.to_csv(topk_path, index=False)
        logging.info(f"Top-K 结果已保存: {topk_path}")
        
        # 保存模型
        if self.tabular_detector:
            model_path = os.path.join(output_dir, "tabular_model.pkl")
            self.tabular_detector.save(model_path)
        
        if self.graph_detector:
            model_path = os.path.join(output_dir, "graph_model.pt")
            self.graph_detector.save(model_path)
        
        # 保存配置
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            # 简化配置输出
            config_dict = {
                "fusion_strategy": self.fusion_config.strategy,
                "tabular_models": self.tabular_config.use_models,
                "graph_hidden": self.graph_config.hidden_channels,
                "graph_epochs": self.train_config.epochs,
                "top_k": self.eval_config.top_k
            }
            json.dump(config_dict, f, indent=2)
        logging.info(f"配置已保存: {config_path}")
    
    def run(self, data_path: str, output_dir: str = "output"):
        """运行完整流水线"""
        start_time = datetime.now()
        logging.info(f"\n开始 Fusion 异常检测流水线 - {start_time}")
        
        try:
            self.load_data(data_path)
            self.preprocess()
            self.train_tabular_model()
            self.train_graph_model()
            self.fuse_scores()
            self.evaluate()
            self.save_results(output_dir)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"\n流水线完成. 总耗时: {duration:.1f} 秒")
            
        except Exception as e:
            logging.error(f"流水线执行失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Fusion 异常检测")
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="输入数据路径 (CSV)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="output",
        help="输出目录"
    )
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="gated",
        choices=["gated", "weighted", "rank", "consistent"],
        help="融合策略"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="图模型训练轮数"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=1000,
        help="输出 Top-K 异常"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = FusionMainConfig()
    config.fusion.strategy = args.strategy
    config.train.epochs = args.epochs
    config.evaluation.top_k = args.top_k
    
    # 创建并运行流水线
    pipeline = FusionPipeline(config)
    pipeline.run(args.data, args.output)


if __name__ == "__main__":
    main()
