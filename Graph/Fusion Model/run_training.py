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
        # 兼容 PyTorch: 参数名是 weights_only（部分老版本不支持该参数）
        try:
            self.graph_data = torch.load(graph_path, weights_only=False)
        except TypeError:
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
        
        # 获取 per-edge 的节点度数（用于门控融合）
        # 使用 min(out_degree(src), in_degree(dst))，与边级分数对齐
        edge_degrees = None
        if hasattr(self.graph_data, 'edge_index'):
            # 优先使用原始边索引（不含自环）
            if hasattr(self.graph_data, 'original_edge_index'):
                edge_index = self.graph_data.original_edge_index.cpu().numpy()
            else:
                edge_index = self.graph_data.edge_index.cpu().numpy()
            
            num_nodes = self.graph_data.num_nodes
            # 计算每个节点的 out-degree 和 in-degree
            out_degree = np.bincount(edge_index[0], minlength=num_nodes)
            in_degree = np.bincount(edge_index[1], minlength=num_nodes)
            
            # 每条边取 min(src 的 out-degree, dst 的 in-degree)
            src_nodes = edge_index[0][:min_len]
            dst_nodes = edge_index[1][:min_len]
            edge_degrees = np.minimum(out_degree[src_nodes], in_degree[dst_nodes])
            
            logging.info(f"边级度数计算完成: min={edge_degrees.min()}, max={edge_degrees.max()}, "
                        f"mean={edge_degrees.mean():.2f}")
        
        # 创建融合策略
        self.fusion_strategy = create_fusion_strategy(self.fusion_config)
        
        # 融合
        self.fusion_result = self.fusion_strategy.fuse(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            node_degrees=edge_degrees
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
        
        # 导入 matplotlib 并设置非交互模式
        import matplotlib
        matplotlib.use('Agg')  # 确保使用非交互式后端
        import matplotlib.pyplot as plt
        
        vis_dir = os.path.join(self.config.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 设置可视化样式
        setup_style()
        
        # 准备统一长度的分数
        n_samples = len(self.fused_scores)
        graph_scores = self.graph_scores[:n_samples]
        tabular_scores = self.tabular_scores[:n_samples]
        
        # 获取 per-edge 度数（与 fuse_scores 保持一致）
        edge_degrees = None
        if hasattr(self.graph_data, 'edge_index'):
            if hasattr(self.graph_data, 'original_edge_index'):
                edge_index = self.graph_data.original_edge_index.cpu().numpy()
            else:
                edge_index = self.graph_data.edge_index.cpu().numpy()
            
            num_nodes = self.graph_data.num_nodes
            out_degree = np.bincount(edge_index[0], minlength=num_nodes)
            in_degree = np.bincount(edge_index[1], minlength=num_nodes)
            
            src_nodes = edge_index[0][:n_samples]
            dst_nodes = edge_index[1][:n_samples]
            edge_degrees = np.minimum(out_degree[src_nodes], in_degree[dst_nodes])
        
        # ============= 1. 模型性能可视化 =============
        logging.info("绘制模型性能图 (1/5)...")
        
        # 1.1 训练曲线
        if self.graph_train_losses:
            plot_training_curves(
                graph_losses=self.graph_train_losses,
                save_path=os.path.join(vis_dir, "training_curves.png")
            )
            plt.close('all')  # 关闭所有图形
            logging.info("  ✓ 训练曲线已保存")
        
        # 1.2 模型对比
        plot_model_comparison(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "model_comparison.png")
        )
        plt.close('all')
        logging.info("  ✓ 模型对比已保存")
        
        # 1.3 分数统计
        plot_score_statistics(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            save_path=os.path.join(vis_dir, "score_statistics.png")
        )
        plt.close('all')
        logging.info("  ✓ 分数统计已保存")
        
        # ============= 2. 融合分析可视化 =============
        logging.info("绘制融合分析图 (2/5)...")
        
        # 2.1 融合概览
        plot_fusion_overview(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            fusion_weights=self.fusion_result.fusion_weights,
            strategy=self.fusion_config.strategy,
            save_path=os.path.join(vis_dir, "fusion_overview.png")
        )
        plt.close('all')
        logging.info("  ✓ 融合概览已保存")
        
        # 2.2 融合权重分布
        if self.fusion_result.fusion_weights is not None:
            plot_fusion_weights_distribution(
                fusion_weights=self.fusion_result.fusion_weights,
                node_degrees=edge_degrees,
                save_path=os.path.join(vis_dir, "fusion_weights_distribution.png")
            )
            plt.close('all')
            logging.info("  ✓ 融合权重分布已保存")
        
        # 2.3 模型一致性分析
        plot_model_agreement(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "model_agreement.png")
        )
        plt.close('all')
        logging.info("  ✓ 模型一致性已保存")
        
        # ============= 3. 特征贡献可视化 =============
        logging.info("绘制特征贡献图 (3/5)...")
        
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
            plt.close('all')
            logging.info("  ✓ 特征重要性已保存")
        
        # 3.2 模型贡献分析
        plot_model_contribution(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            fusion_weights=self.fusion_result.fusion_weights,
            node_degrees=edge_degrees,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "model_contribution.png")
        )
        plt.close('all')
        logging.info("  ✓ 模型贡献已保存")
        
        # ============= 4. 异常分布可视化 =============
        logging.info("绘制异常分布图 (4/5)...")
        
        # 4.1 分数分布
        plot_score_distributions(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            save_path=os.path.join(vis_dir, "score_distributions.png")
        )
        plt.close('all')
        logging.info("  ✓ 分数分布已保存")
        
        # 4.2 异常散点图
        plot_anomaly_scatter(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            top_k=self.eval_config.top_k,
            save_path=os.path.join(vis_dir, "anomaly_scatter.png")
        )
        plt.close('all')
        logging.info("  ✓ 异常散点图已保存")
        
        # 4.3 Top-K 分析
        plot_topk_analysis(
            graph_scores=graph_scores,
            tabular_scores=tabular_scores,
            fused_scores=self.fused_scores,
            k_values=[100, 200, 500, 1000, 2000],
            save_path=os.path.join(vis_dir, "topk_analysis.png")
        )
        plt.close('all')
        logging.info("  ✓ Top-K分析已保存")
        
        # ============= 5. 综合报告 =============
        logging.info("创建综合报告 (5/5)...")
        
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
            node_degrees=edge_degrees,
            fusion_strategy=self.fusion_config.strategy,
            top_k=self.eval_config.top_k
        )
        plt.close('all')
        logging.info("  ✓ 综合报告已保存")
        
        logging.info("=" * 60)
        logging.info(f"✓ 所有可视化结果已保存到: {vis_dir}")
        logging.info("=" * 60)
    
    def save_results(self):
        """保存结果（增强版：支持可视化脚本读取）"""
        logging.info("=" * 60)
        logging.info("步骤 7: 保存结果")
        logging.info("=" * 60)
        
        self.config.ensure_dirs()
        output_dir = self.config.output_dir
        
        # ============= 1. 保存完整训练结果 (用于可视化脚本) =============
        n_samples = len(self.fused_scores)
        
        # 获取 per-edge 度数（与 fuse_scores 保持一致）
        edge_degrees = None
        if hasattr(self.graph_data, 'edge_index'):
            if hasattr(self.graph_data, 'original_edge_index'):
                edge_index = self.graph_data.original_edge_index.cpu().numpy()
            else:
                edge_index = self.graph_data.edge_index.cpu().numpy()
            
            num_nodes = self.graph_data.num_nodes
            out_degree = np.bincount(edge_index[0], minlength=num_nodes)
            in_degree = np.bincount(edge_index[1], minlength=num_nodes)
            
            src_nodes = edge_index[0][:n_samples]
            dst_nodes = edge_index[1][:n_samples]
            edge_degrees = np.minimum(out_degree[src_nodes], in_degree[dst_nodes])
        
        # 获取特征名称
        feature_names = None
        if self.meta_info:
            if 'tabular_info' in self.meta_info and 'feature_names' in self.meta_info['tabular_info']:
                feature_names = self.meta_info['tabular_info']['feature_names']
            elif 'feature_names' in self.meta_info:
                feature_names = self.meta_info['feature_names']
        
        # 准备评估报告
        evaluation_report = None
        if self.df is not None and hasattr(self, 'evaluator') and self.evaluator is not None:
            df_subset = self.df.iloc[:n_samples]
            try:
                evaluation_report = self.evaluator.evaluate(
                    df=df_subset,
                    scores=self.fused_scores,
                    top_k=self.eval_config.top_k
                )
            except Exception as e:
                logging.warning(f"生成评估报告失败: {e}")
        
        # 保存完整结果到 pickle (用于可视化脚本)
        training_results = {
            # 分数
            'graph_scores': self.graph_scores[:n_samples],
            'tabular_scores': self.tabular_scores[:n_samples],
            'fused_scores': self.fused_scores,
            'fusion_weights': self.fusion_result.fusion_weights if self.fusion_result else None,
            
            # 训练历史
            'graph_train_losses': self.graph_train_losses,
            'tabular_train_info': self.tabular_detector.get_training_info() if hasattr(self.tabular_detector, 'get_training_info') else None,
            
            # 融合信息
            'fusion_strategy': self.fusion_config.strategy,
            'fusion_config': {
                'strategy': self.fusion_config.strategy,
                'degree_threshold': self.fusion_config.degree_threshold,
                'alpha_high': self.fusion_config.alpha_high,
                'alpha_low': self.fusion_config.alpha_low,
                'fusion_alpha': self.fusion_config.fusion_alpha,
            },
            
            # 数据信息
            'node_degrees': edge_degrees,  # per-edge degrees (min of src out-deg and dst in-deg)
            'feature_names': feature_names,
            'tabular_features': self.tabular_features[:n_samples] if self.tabular_features is not None else None,
            
            # 评估报告
            'evaluation_report': evaluation_report,
            
            # 元信息
            'meta_info': self.meta_info,
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
        }
        
        results_pkl_path = os.path.join(output_dir, "training_results.pkl")
        with open(results_pkl_path, 'wb') as f:
            pickle.dump(training_results, f)
        logging.info(f"完整训练结果已保存: {results_pkl_path}")
        
        # ============= 2. 保存 CSV 分数 (向后兼容) =============
        results_df = pd.DataFrame({
            "graph_score": self.graph_scores[:n_samples],
            "tabular_score": self.tabular_scores[:n_samples],
            "fused_score": self.fused_scores
        })
        
        if self.fusion_result.fusion_weights is not None:
            results_df["fusion_weight"] = self.fusion_result.fusion_weights
        
        results_path = os.path.join(output_dir, "fusion_scores.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"分数 CSV 已保存: {results_path}")
        
        # ============= 3. 保存 Top-K =============
        if self.df is not None:
            topk_idx = np.argsort(-self.fused_scores)[:self.eval_config.top_k]
            topk_df = self.df.iloc[topk_idx].copy()
            topk_df["fused_score"] = self.fused_scores[topk_idx]
            topk_df["rank"] = range(1, len(topk_idx) + 1)
            
            topk_path = os.path.join(output_dir, f"top_{self.eval_config.top_k}_anomalies.csv")
            topk_df.to_csv(topk_path, index=False)
            logging.info(f"Top-K 结果已保存: {topk_path}")
        
        # ============= 4. 保存模型 =============
        if self.config.save_model:
            if self.tabular_detector:
                model_path = os.path.join(output_dir, "tabular_model.pkl")
                self.tabular_detector.save(model_path)
                logging.info(f"表格模型已保存: {model_path}")
            
            if self.graph_detector:
                model_path = os.path.join(output_dir, "graph_model.pt")
                self.graph_detector.save(model_path)
                logging.info(f"图模型已保存: {model_path}")
        
        # ============= 5. 保存配置 =============
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            config_dict = {
                "fusion_strategy": self.fusion_config.strategy,
                "tabular_model_type": self.tabular_config.model_type,
                "graph_hidden": self.graph_config.hidden_channels,
                "graph_layers": self.graph_config.num_layers,
                "graph_epochs": self.train_config.epochs,
                "graph_lr": self.train_config.lr,
                "mask_rate": self.graph_config.mask_rate,
                "top_k": self.eval_config.top_k,
                "processed_data_dir": self.config.processed_data_dir,
                "timestamp": datetime.now().isoformat()
            }
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logging.info(f"配置已保存: {config_path}")
        
        # ============= 6. 保存文字报告 =============
        self._save_text_report(output_dir, evaluation_report)
    
    def _save_text_report(self, output_dir: str, evaluation_report: Optional[Dict] = None):
        """保存文字报告"""
        report_path = os.path.join(output_dir, "training_report.txt")
        
        n_samples = len(self.fused_scores)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("                    模型训练报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # 数据概览
            f.write("-" * 70 + "\n")
            f.write("1. 数据概览\n")
            f.write("-" * 70 + "\n")
            f.write(f"  样本数量: {n_samples:,}\n")
            if self.graph_data is not None:
                f.write(f"  图节点数: {self.graph_data.num_nodes:,}\n")
                f.write(f"  图边数:   {self.graph_data.edge_index.shape[1]:,}\n")
            if self.tabular_features is not None:
                f.write(f"  表格特征维度: {self.tabular_features.shape[1]}\n")
            f.write("\n")
            
            # 模型配置
            f.write("-" * 70 + "\n")
            f.write("2. 模型配置\n")
            f.write("-" * 70 + "\n")
            f.write(f"  [图模型 - GraphMAE]\n")
            f.write(f"    隐藏层维度: {self.graph_config.hidden_channels}\n")
            f.write(f"    输出维度:   {self.graph_config.out_channels}\n")
            f.write(f"    层数:       {self.graph_config.num_layers}\n")
            f.write(f"    注意力头数: {self.graph_config.num_heads}\n")
            f.write(f"    Mask Rate:  {self.graph_config.mask_rate}\n")
            f.write(f"    训练轮数:   {self.train_config.epochs}\n")
            f.write(f"    学习率:     {self.train_config.lr}\n")
            f.write(f"\n")
            f.write(f"  [表格模型]\n")
            f.write(f"    模型类型: {self.tabular_config.model_type}\n")
            f.write("\n")
            
            # 训练结果
            f.write("-" * 70 + "\n")
            f.write("3. 训练结果\n")
            f.write("-" * 70 + "\n")
            if self.graph_train_losses:
                f.write(f"  [图模型训练]\n")
                f.write(f"    初始 Loss:  {self.graph_train_losses[0]:.6f}\n")
                f.write(f"    最终 Loss:  {self.graph_train_losses[-1]:.6f}\n")
                f.write(f"    最低 Loss:  {min(self.graph_train_losses):.6f}\n")
                f.write(f"    实际轮数:   {len(self.graph_train_losses)}\n")
            f.write("\n")
            
            # 分数统计
            f.write("-" * 70 + "\n")
            f.write("4. 分数统计\n")
            f.write("-" * 70 + "\n")
            f.write(f"  [图模型分数]\n")
            f.write(f"    范围: [{self.graph_scores.min():.4f}, {self.graph_scores.max():.4f}]\n")
            f.write(f"    均值: {self.graph_scores.mean():.4f}\n")
            f.write(f"    标准差: {self.graph_scores.std():.4f}\n")
            f.write(f"\n")
            f.write(f"  [表格模型分数]\n")
            f.write(f"    范围: [{self.tabular_scores.min():.4f}, {self.tabular_scores.max():.4f}]\n")
            f.write(f"    均值: {self.tabular_scores.mean():.4f}\n")
            f.write(f"    标准差: {self.tabular_scores.std():.4f}\n")
            f.write(f"\n")
            f.write(f"  [融合分数]\n")
            f.write(f"    范围: [{self.fused_scores.min():.4f}, {self.fused_scores.max():.4f}]\n")
            f.write(f"    均值: {self.fused_scores.mean():.4f}\n")
            f.write(f"    标准差: {self.fused_scores.std():.4f}\n")
            f.write("\n")
            
            # 融合策略
            f.write("-" * 70 + "\n")
            f.write("5. 融合策略\n")
            f.write("-" * 70 + "\n")
            f.write(f"  策略类型: {self.fusion_config.strategy}\n")
            if self.fusion_config.strategy == "gated":
                f.write(f"  度数阈值: {self.fusion_config.degree_threshold}\n")
                f.write(f"  高度权重 (alpha_high): {self.fusion_config.alpha_high}\n")
                f.write(f"  低度权重 (alpha_low):  {self.fusion_config.alpha_low}\n")
            elif self.fusion_config.strategy == "weighted":
                f.write(f"  融合权重 (alpha): {self.fusion_config.fusion_alpha}\n")
            if self.fusion_result and self.fusion_result.fusion_weights is not None:
                weights = self.fusion_result.fusion_weights
                f.write(f"  图模型平均权重: {weights.mean():.4f}\n")
            f.write("\n")
            
            # 评估结果
            if evaluation_report:
                f.write("-" * 70 + "\n")
                f.write("6. 评估结果\n")
                f.write("-" * 70 + "\n")
                if 'score_statistics' in evaluation_report:
                    stats = evaluation_report['score_statistics']
                    f.write(f"  Top-K: {self.eval_config.top_k}\n")
                if 'weak_rule_results' in evaluation_report:
                    f.write(f"  弱规则匹配:\n")
                    for rule_name, result in evaluation_report['weak_rule_results'].items():
                        f.write(f"    {rule_name}: {result.get('top_k_ratio', 0)*100:.2f}%\n")
                f.write("\n")
            
            # 输出文件
            f.write("-" * 70 + "\n")
            f.write("7. 输出文件\n")
            f.write("-" * 70 + "\n")
            f.write(f"  training_results.pkl   - 完整训练结果 (用于可视化)\n")
            f.write(f"  fusion_scores.csv      - 分数 CSV\n")
            f.write(f"  top_*.csv              - Top-K 异常样本\n")
            f.write(f"  training_config.json   - 训练配置\n")
            f.write(f"  training_report.txt    - 本报告\n")
            if self.config.save_model:
                f.write(f"  graph_model.pt         - 图模型\n")
                f.write(f"  tabular_model.pkl      - 表格模型\n")
            f.write("\n")
            
            # 可视化提示
            f.write("-" * 70 + "\n")
            f.write("8. 可视化\n")
            f.write("-" * 70 + "\n")
            f.write("  运行以下命令生成可视化图表:\n")
            f.write(f"  python run_visualization.py --results {output_dir}\n")
            f.write("\n")
            f.write("  可调整参数:\n")
            f.write("    --top-k <N>       调整 Top-K 数量\n")
            f.write("    --output <DIR>    指定输出目录\n")
            f.write("    --only <TYPES>    只生成特定图表\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("                    报告结束\n")
            f.write("=" * 70 + "\n")
        
        logging.info(f"文字报告已保存: {report_path}")
    
    def run(self, skip_visualization: bool = False):
        """运行完整训练流水线
        
        Args:
            skip_visualization: 是否跳过可视化步骤 (可后续通过 run_visualization.py 生成)
        """
        start_time = datetime.now()
        logging.info(f"\n开始训练流水线 - {start_time}")
        
        try:
            self.load_preprocessed_data()
            self.train_tabular_model()
            self.train_graph_model()
            self.fuse_scores()
            self.evaluate()
            self.save_results()  # 保存结果移到可视化之前
            
            if not skip_visualization:
                self.visualize_results()
            else:
                logging.info("=" * 60)
                logging.info("可视化已跳过")
                logging.info(f"运行以下命令生成可视化: python run_visualization.py --results {self.config.output_dir}")
                logging.info("=" * 60)
            
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
        help="[已弃用] 使用 --skip-visualization"
    )
    
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="跳过可视化 (可后续通过 run_visualization.py 生成)"
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
    config.visualize = not (args.no_visualize or args.skip_visualization)
    config.device = args.device
    
    # 创建并运行流水线
    pipeline = TrainingPipeline(config)
    pipeline.run(skip_visualization=args.skip_visualization or args.no_visualize)


if __name__ == "__main__":
    main()
