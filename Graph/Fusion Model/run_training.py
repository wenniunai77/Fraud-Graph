"""
模型训练主脚本
读取预处理完成的图数据和表格特征
训练图模型和表格模型，进行融合和评估
"""
import argparse
import logging
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Optional, Dict, Any

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import TrainingMainConfig
from models import TabularAnomalyDetector, GraphAnomalyDetector
from fusion import create_fusion_strategy, analyze_fusion, print_fusion_report
from evaluation import UnsupervisedEvaluator
from visualization import (
    # 样式设置
    setup_style,
    # 模型性能
    plot_training_curves,
    plot_model_comparison,
    plot_score_statistics,
    # 融合分析
    plot_fusion_overview,
    plot_fusion_weights_distribution,
    plot_model_agreement,
    # 特征贡献
    plot_feature_importance,
    plot_model_contribution,
    # 异常分布
    plot_score_distributions,
    plot_anomaly_scatter,
    plot_topk_analysis,
    # 仪表板
    create_comprehensive_report
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class TrainingPipeline:
    """训练流水线"""
    
    def __init__(self, config: TrainingMainConfig):
        self.config = config
        self.tabular_config = config.tabular_model
        self.graph_config = config.graph_model
        self.train_config = config.train
        self.fusion_config = config.fusion
        self.eval_config = config.evaluation
        
        # 组件
        self.tabular_detector: Optional[TabularAnomalyDetector] = None
        self.graph_detector: Optional[GraphAnomalyDetector] = None
        self.fusion_strategy = None
        self.evaluator: Optional[UnsupervisedEvaluator] = None
        
        # 数据
        self.df: Optional[pd.DataFrame] = None
        self.tabular_features: Optional[np.ndarray] = None
        self.graph_data = None
        self.meta_info: Optional[Dict[str, Any]] = None
        
        # 结果
        self.tabular_scores: Optional[np.ndarray] = None
        self.graph_scores: Optional[np.ndarray] = None
        self.fused_scores: Optional[np.ndarray] = None
        self.fusion_result = None
        self.graph_train_losses = None
    
    def load_preprocessed_data(self):
        """加载预处理完成的数据"""
        logging.info("=" * 60)
        logging.info("步骤 1: 加载预处理数据")
        logging.info("=" * 60)
        
        processed_dir = self.config.processed_data_dir
        
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"预处理数据目录不存在: {processed_dir}")
        
        # 1. 加载图数据
        graph_path = self.config.get_graph_data_path()
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"图数据文件不存在: {graph_path}")
        self.graph_data = torch.load(graph_path)
        logging.info(f"图数据已加载: {self.graph_data.num_nodes:,} 节点, "
                    f"{self.graph_data.edge_index.shape[1]:,} 边")
        
        # 2. 加载表格特征
        tabular_path = self.config.get_tabular_features_path()
        if not os.path.exists(tabular_path):
            raise FileNotFoundError(f"表格特征文件不存在: {tabular_path}")
        self.tabular_features = np.load(tabular_path)
        logging.info(f"表格特征已加载: {self.tabular_features.shape}")
        
        # 3. 加载原始数据（用于评估）
        raw_data_path = self.config.get_raw_data_path()
        if os.path.exists(raw_data_path):
            self.df = pd.read_pickle(raw_data_path)
            logging.info(f"原始数据已加载: {len(self.df):,} 行")
        else:
            logging.warning(f"原始数据文件不存在: {raw_data_path}")
            logging.warning("评估功能将受限")
        
        # 4. 加载元信息
        meta_path = self.config.get_meta_info_path()
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta_info = json.load(f)
            logging.info("元信息已加载")
        
        logging.info("预处理数据加载完成")
    
    def train_tabular_model(self):
        """训练表格模型"""
        logging.info("=" * 60)
        logging.info("步骤 2: 训练表格模型")
        logging.info("=" * 60)
        
        self.tabular_detector = TabularAnomalyDetector(self.tabular_config)
        self.tabular_detector.fit(self.tabular_features)
        
        # 获取分数
        self.tabular_scores = self.tabular_detector.predict_fusion_score(
            self.tabular_features
        )
        
        logging.info(f"表格模型训练完成")
        logging.info(f"表格分数范围: [{self.tabular_scores.min():.4f}, {self.tabular_scores.max():.4f}]")
    
    def train_graph_model(self):
        """训练图模型"""
        logging.info("=" * 60)
        logging.info("步骤 3: 训练图模型")
        logging.info("=" * 60)
        
        device = "cuda" if torch.cuda.is_available() and self.config.device >= 0 else "cpu"
        logging.info(f"使用设备: {device}")
        
        self.graph_detector = GraphAnomalyDetector(
            self.graph_config,
            self.train_config,
            device=device
        )
        
        # 训练并保存训练历史
        self.graph_train_losses = self.graph_detector.train(self.graph_data)
        
        # 获取边级别分数
        self.graph_scores = self.graph_detector.predict_scores(
            self.graph_data,
            level="edge",
            strategy="max"
        )
        
        logging.info(f"图模型训练完成")
        logging.info(f"图分数范围: [{self.graph_scores.min():.4f}, {self.graph_scores.max():.4f}]")
    
    def fuse_scores(self):
        """融合分数"""
        logging.info("=" * 60)
        logging.info("步骤 4: 融合分数")
        logging.info("=" * 60)
        
        # 确保分数长度一致
        min_len = min(len(self.tabular_scores), len(self.graph_scores))
        tabular_scores = self.tabular_scores[:min_len]
        graph_scores = self.graph_scores[:min_len]
        
        # 获取节点度数（用于门控融合）
        node_degrees = None
        if hasattr(self.graph_data, 'edge_index'):
            edge_index = self.graph_data.edge_index.cpu().numpy()
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
        
        logging.info(f"融合完成")
        logging.info(f"融合分数范围: [{self.fused_scores.min():.4f}, {self.fused_scores.max():.4f}]")
    
    def evaluate(self):
        """评估"""
        logging.info("=" * 60)
        logging.info("步骤 5: 无标签评估")
        logging.info("=" * 60)
        
        if self.df is None:
            logging.warning("原始数据不可用，跳过评估")
            return None
        
        self.evaluator = UnsupervisedEvaluator(self.eval_config)
        
        # 添加弱规则
        col_idx = self.config.col_idx
        payment_amount_idx = col_idx.payment_amount
        
        if payment_amount_idx < len(self.df.columns):
            amount_col = self.df.columns[payment_amount_idx]
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
    
    def visualize_results(self):
        """可视化结果"""
        if not self.config.visualize:
            logging.info("可视化已禁用，跳过")
            return
        
        logging.info("=" * 60)
        logging.info("步骤 6: 可视化")
        logging.info("=" * 60)
        
        vis_dir = os.path.join(self.config.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 设置可视化样式
        setup_style()
        
        # 获取节点度数
        node_degrees = None
        if hasattr(self.graph_data, 'edge_index'):
            edge_index = self.graph_data.edge_index.cpu().numpy()
            from collections import Counter
            n_nodes = self.graph_data.x.size(0)
            degree_counter = Counter(edge_index[0].tolist() + edge_index[1].tolist())
            node_degrees = np.array([degree_counter.get(i, 0) for i in range(n_nodes)])
            node_degrees = node_degrees[:len(self.fused_scores)]
        
        # 准备统一长度的分数
        n_samples = len(self.fused_scores)
        graph_scores = self.graph_scores[:n_samples]
        tabular_scores = self.tabular_scores[:n_samples]
        
        # ============= 1. 模型性能可视化 =============
        logging.info("绘制模型性能图...")
        
        # 1.1 训练曲线
        if self.graph_train_losses:
            plot_training_curves(
                graph_losses=self.graph_train_losses,
                save_path=os.path.join(vis_dir, "training_curves.png")
            )
        
        # 1.2 模型对比
        plot_model_comparison(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "model_comparison.png")
        )
        
        # 1.3 分数统计
        plot_score_statistics(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            save_path=os.path.join(vis_dir, "score_statistics.png")
        )
        
        # ============= 2. 融合分析可视化 =============
        logging.info("绘制融合分析图...")
        
        # 2.1 融合概览
        plot_fusion_overview(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            fusion_weights=self.fusion_result.fusion_weights,
            strategy=self.fusion_config.strategy,
            save_path=os.path.join(vis_dir, "fusion_overview.png")
        )
        
        # 2.2 融合权重分布
        if self.fusion_result.fusion_weights is not None:
            plot_fusion_weights_distribution(
                fusion_weights=self.fusion_result.fusion_weights,
                node_degrees=node_degrees,
                save_path=os.path.join(vis_dir, "fusion_weights_distribution.png")
            )
        
        # 2.3 模型一致性分析
        plot_model_agreement(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "model_agreement.png")
        )
        
        # ============= 3. 特征贡献可视化 =============
        logging.info("绘制特征贡献图...")
        
        # 3.1 特征重要性（如有特征名称）
        feature_names = None
        if self.meta_info:
            # 尝试从不同位置获取特征名称
            if 'tabular_info' in self.meta_info and 'feature_names' in self.meta_info['tabular_info']:
                feature_names = self.meta_info['tabular_info']['feature_names']
            elif 'feature_names' in self.meta_info:
                feature_names = self.meta_info['feature_names']
        
        if feature_names and self.tabular_features is not None:
            plot_feature_importance(
                tabular_features=self.tabular_features[:n_samples],
                tabular_scores=tabular_scores,
                feature_names=feature_names,
                save_path=os.path.join(vis_dir, "feature_importance.png")
            )
        
        # 3.2 模型贡献分析
        plot_model_contribution(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            fusion_weights=self.fusion_result.fusion_weights,
            node_degrees=node_degrees,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "model_contribution.png")
        )
        
        # ============= 4. 异常分布可视化 =============
        logging.info("绘制异常分布图...")
        
        # 4.1 分数分布
        plot_score_distributions(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            save_path=os.path.join(vis_dir, "score_distributions.png")
        )
        
        # 4.2 异常散点图
        plot_anomaly_scatter(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "anomaly_scatter.png")
        )
        
        # 4.3 Top-K 分析
        plot_topk_analysis(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            k_values=[100, 200, 500, 1000, 2000],
            save_path=os.path.join(vis_dir, "topk_analysis.png")
        )
        
        # ============= 5. 综合报告 =============
        logging.info("创建综合报告...")
        
        # 准备评估报告
        evaluation_report = None
        if self.df is not None:
            evaluator = UnsupervisedEvaluator(self.eval_config)
            df_subset = self.df.iloc[:n_samples]
            evaluation_report = evaluator.evaluate(
                df=df_subset,
                scores=self.fused_scores,
                top_k=self.eval_config.top_k
            )
        
        create_comprehensive_report(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            output_dir=vis_dir,
            graph_train_losses=self.graph_train_losses,
            fusion_weights=self.fusion_result.fusion_weights,
            node_degrees=node_degrees,
            fusion_strategy=self.fusion_config.strategy,
            top_k=self.eval_config.top_k
        )
        
        logging.info(f"所有可视化结果已保存到: {vis_dir}")
    
    def save_results(self):
        """保存结果"""
        logging.info("=" * 60)
        logging.info("步骤 7: 保存结果")
        logging.info("=" * 60)
        
        self.config.ensure_dirs()
        output_dir = self.config.output_dir
        
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
        if self.df is not None:
            topk_idx = np.argsort(-self.fused_scores)[:self.eval_config.top_k]
            topk_df = self.df.iloc[topk_idx].copy()
            topk_df["fused_score"] = self.fused_scores[topk_idx]
            topk_df["rank"] = range(1, len(topk_idx) + 1)
            
            topk_path = os.path.join(output_dir, f"top_{self.eval_config.top_k}_anomalies.csv")
            topk_df.to_csv(topk_path, index=False)
            logging.info(f"Top-K 结果已保存: {topk_path}")
        
        # 保存模型
        if self.config.save_model:
            if self.tabular_detector:
                model_path = os.path.join(output_dir, "tabular_model.pkl")
                self.tabular_detector.save(model_path)
                logging.info(f"表格模型已保存: {model_path}")
            
            if self.graph_detector:
                model_path = os.path.join(output_dir, "graph_model.pt")
                self.graph_detector.save(model_path)
                logging.info(f"图模型已保存: {model_path}")
        
        # 保存配置
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            config_dict = {
                "fusion_strategy": self.fusion_config.strategy,
                "tabular_model_type": self.tabular_config.model_type,
                "graph_hidden": self.graph_config.hidden_channels,
                "graph_epochs": self.train_config.epochs,
                "top_k": self.eval_config.top_k,
                "processed_data_dir": self.config.processed_data_dir
            }
            json.dump(config_dict, f, indent=2)
        logging.info(f"配置已保存: {config_path}")
    
    def run(self):
        """运行完整训练流水线"""
        start_time = datetime.now()
        logging.info(f"\n开始训练流水线 - {start_time}")
        
        try:
            self.load_preprocessed_data()
            self.train_tabular_model()
            self.train_graph_model()
            self.fuse_scores()
            self.evaluate()
            self.visualize_results()
            self.save_results()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"\n训练流水线完成. 总耗时: {duration:.1f} 秒")
            
        except Exception as e:
            logging.error(f"训练流水线执行失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型训练 - 图模型与表格模型融合")
    
    parser.add_argument(
        "--processed-data",
        type=str,
        required=True,
        help="预处理数据目录"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
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
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="禁用可视化"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU 设备号 (-1 表示 CPU)"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingMainConfig()
    config.processed_data_dir = args.processed_data
    config.output_dir = args.output
    config.fusion.strategy = args.strategy
    config.train.epochs = args.epochs
    config.evaluation.top_k = args.top_k
    config.visualize = not args.no_visualize
    config.device = args.device
    
    # 创建并运行流水线
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
