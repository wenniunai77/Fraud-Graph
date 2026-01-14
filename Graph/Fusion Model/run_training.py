"""
模型训练主脚本
读取预处理完成的图数据和表格特征
训练图模型和表格模型，进行融合和评估
支持单 seed 或多 seed 模式，自动输出稳定性分析
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
from typing import Optional, Dict, Any, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import TrainingMainConfig
from models import TabularAnomalyDetector, GraphAnomalyDetector
from fusion import create_fusion_strategy, analyze_fusion, print_fusion_report
from evaluation import UnsupervisedEvaluator
from utils import set_seed, compute_topk_overlap  # 新增：导入统一的 seed 固化工具

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
        
        # P3 修复: 严格检查分数长度一致性（不再悄悄截断）
        if len(self.tabular_scores) != len(self.graph_scores):
            error_msg = (
                f"分数长度不一致！\n"
                f"  表格分数: {len(self.tabular_scores)}\n"
                f"  图分数: {len(self.graph_scores)}\n"
                f"这可能是边过滤逻辑不一致导致的数据对齐错误。\n"
                f"请检查:\n"
                f"  1. 图侧和表格侧是否使用了相同的边数据\n"
                f"  2. 是否有不一致的过滤/采样逻辑\n"
                f"  3. original_edge_index 和 edge_index 的使用是否正确"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        tabular_scores = self.tabular_scores
        graph_scores = self.graph_scores
        num_edges = len(tabular_scores)
        
        logging.info(f"分数对齐检查通过: {num_edges} 条边")
        
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
            src_nodes = edge_index[0][:num_edges]
            dst_nodes = edge_index[1][:num_edges]
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
        
        # 评估
        df_subset = self.df.iloc[:len(self.fused_scores)]
        report = self.evaluator.evaluate(
            df=df_subset,
            scores=self.fused_scores,
            top_k=self.eval_config.top_k
        )
        
        # 打印报告 (EvaluationReport 对象)
        self.evaluator.print_report(report)
        
        # 返回字典格式 (用于保存)
        return report.to_dict()
    
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
                eval_result = self.evaluator.evaluate(
                    df=df_subset,
                    scores=self.fused_scores,
                    top_k=self.eval_config.top_k
                )
                # 转换为字典格式以便序列化和后续访问
                evaluation_report = eval_result.to_dict()
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
            
            # 添加分数和排名信息
            topk_df["rank"] = range(1, len(topk_idx) + 1)
            topk_df["fused_score"] = self.fused_scores[topk_idx]
            topk_df["graph_score"] = self.graph_scores[topk_idx]
            topk_df["tabular_score"] = self.tabular_scores[topk_idx]
            
            # 添加融合权重（门控融合时可用）
            if self.fusion_result.fusion_weights is not None:
                topk_df["fusion_weight"] = self.fusion_result.fusion_weights[topk_idx]
            
            # 添加节点度数代理（活跃度指标，用于解释为什么更信图/表）
            if hasattr(self.graph_data, 'edge_index'):
                if hasattr(self.graph_data, 'original_edge_index'):
                    edge_index = self.graph_data.original_edge_index.cpu().numpy()
                else:
                    edge_index = self.graph_data.edge_index.cpu().numpy()
                
                num_nodes = self.graph_data.num_nodes
                out_degree = np.bincount(edge_index[0], minlength=num_nodes)
                in_degree = np.bincount(edge_index[1], minlength=num_nodes)
                
                src_nodes = edge_index[0][topk_idx]
                dst_nodes = edge_index[1][topk_idx]
                node_degree_proxy = np.minimum(out_degree[src_nodes], in_degree[dst_nodes])
                
                topk_df["node_degree_proxy"] = node_degree_proxy
            
            topk_path = os.path.join(output_dir, f"top_{self.eval_config.top_k}_anomalies.csv")
            topk_df.to_csv(topk_path, index=False)
            logging.info(f"Top-K 结果已保存: {topk_path}")
            logging.info(f"  包含字段: {list(topk_df.columns)}")
        
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
            self.save_results()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"\n训练流水线完成. 总耗时: {duration:.1f} 秒")
            logging.info("=" * 60)
            logging.info("提示: 运行以下命令生成可视化")
            logging.info(f"  python run_visualization.py --results {self.config.output_dir}")
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"训练流水线执行失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型训练 - 图模型与表格模型融合")
    
    parser.add_argument(
        "--processed-data",
        type=str,
        default="./processed_data",
        help="预处理数据目录 (默认: ./processed_data)"
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
    config.device = args.device
    
    # ==================== 多 Seed 模式 ====================
    if config.enable_multi_seed and len(config.seeds) > 1:
        logging.info("=" * 80)
        logging.info(f"🔄 多 Seed 模式: 将运行 {len(config.seeds)} 个 seed")
        logging.info(f"Seeds: {config.seeds}")
        logging.info("=" * 80)
        
        run_multi_seed_training(config)
    
    # ==================== 单 Seed 模式 ====================
    else:
        # P2 修复: 统一设置随机种子（确保可复现）
        set_seed(config.seed)
        logging.info(f"随机种子已设置: {config.seed}")
        
        # 创建并运行流水线
        pipeline = TrainingPipeline(config)
        pipeline.run()


def run_multi_seed_training(config: TrainingMainConfig):
    """
    多 Seed 训练并输出稳定性分析
    
    Args:
        config: 训练配置
    """
    seeds = config.seeds
    n_seeds = len(seeds)
    
    # 存储每个 seed 的结果
    all_fused_scores = []
    all_graph_scores = []
    all_tabular_scores = []
    all_topk_indices = []
    
    # 为每个 seed 创建独立输出目录
    base_output_dir = config.output_dir
    seed_output_dirs = []
    
    for i, seed in enumerate(seeds):
        logging.info("\n" + "=" * 80)
        logging.info(f"🌱 运行 Seed {i+1}/{n_seeds}: {seed}")
        logging.info("=" * 80)
        
        # 设置随机种子
        set_seed(seed)
        
        # 创建该 seed 的输出目录
        seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)
        seed_output_dirs.append(seed_output_dir)
        
        # 临时修改配置的输出目录
        original_output_dir = config.output_dir
        config.output_dir = seed_output_dir
        
        # 运行训练
        pipeline = TrainingPipeline(config)
        pipeline.run()
        
        # 恢复原始输出目录
        config.output_dir = original_output_dir
        
        # 保存该 seed 的分数
        all_fused_scores.append(pipeline.fused_scores)
        all_graph_scores.append(pipeline.graph_scores)
        all_tabular_scores.append(pipeline.tabular_scores)
        
        # 保存 Top-K 索引
        topk_idx = np.argsort(-pipeline.fused_scores)[:config.evaluation.top_k]
        all_topk_indices.append(set(topk_idx))
        
        logging.info(f"✅ Seed {seed} 完成，结果已保存到: {seed_output_dir}")
    
    # ==================== 生成稳定性分析 ====================
    logging.info("\n" + "=" * 80)
    logging.info("📊 生成跨 Seed 稳定性分析")
    logging.info("=" * 80)
    
    analyze_multi_seed_results(
        config=config,
        seeds=seeds,
        all_fused_scores=all_fused_scores,
        all_graph_scores=all_graph_scores,
        all_tabular_scores=all_tabular_scores,
        all_topk_indices=all_topk_indices,
        output_dir=base_output_dir
    )
    
    logging.info("\n" + "=" * 80)
    logging.info(f"✅ 所有 {n_seeds} 个 Seed 训练完成！")
    logging.info(f"📂 结果保存在: {base_output_dir}")
    logging.info("=" * 80)


def analyze_multi_seed_results(
    config: TrainingMainConfig,
    seeds: List[int],
    all_fused_scores: List[np.ndarray],
    all_graph_scores: List[np.ndarray],
    all_tabular_scores: List[np.ndarray],
    all_topk_indices: List[set],
    output_dir: str
):
    """
    分析多 seed 结果的稳定性
    
    Args:
        config: 训练配置
        seeds: seed 列表
        all_fused_scores: 所有 seed 的融合分数
        all_graph_scores: 所有 seed 的图分数
        all_tabular_scores: 所有 seed 的表格分数
        all_topk_indices: 所有 seed 的 Top-K 索引集合
        output_dir: 输出目录
    """
    n_seeds = len(seeds)
    k_values = config.evaluation.top_k_values
    
    # ==================== 1. 计算 Jaccard 相似度 ====================
    logging.info("计算 Top-K Jaccard 相似度...")
    
    jaccard_results = {
        'fused': compute_topk_overlap(all_fused_scores[0], all_fused_scores[1], k_values) if n_seeds >= 2 else {},
        'graph': compute_topk_overlap(all_graph_scores[0], all_graph_scores[1], k_values) if n_seeds >= 2 else {},
        'tabular': compute_topk_overlap(all_tabular_scores[0], all_tabular_scores[1], k_values) if n_seeds >= 2 else {}
    }
    
    # 计算所有配对的平均相似度
    jaccard_matrix_fused = np.zeros((n_seeds, n_seeds))
    jaccard_matrix_graph = np.zeros((n_seeds, n_seeds))
    jaccard_matrix_tabular = np.zeros((n_seeds, n_seeds))
    
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            # 使用主 K 值计算
            k = config.evaluation.top_k
            
            # 融合分数
            topk_i = set(np.argsort(-all_fused_scores[i])[:k])
            topk_j = set(np.argsort(-all_fused_scores[j])[:k])
            jaccard_fused = len(topk_i & topk_j) / k
            jaccard_matrix_fused[i, j] = jaccard_fused
            jaccard_matrix_fused[j, i] = jaccard_fused
            
            # 图分数
            topk_i = set(np.argsort(-all_graph_scores[i])[:k])
            topk_j = set(np.argsort(-all_graph_scores[j])[:k])
            jaccard_graph = len(topk_i & topk_j) / k
            jaccard_matrix_graph[i, j] = jaccard_graph
            jaccard_matrix_graph[j, i] = jaccard_graph
            
            # 表格分数
            topk_i = set(np.argsort(-all_tabular_scores[i])[:k])
            topk_j = set(np.argsort(-all_tabular_scores[j])[:k])
            jaccard_tabular = len(topk_i & topk_j) / k
            jaccard_matrix_tabular[i, j] = jaccard_tabular
            jaccard_matrix_tabular[j, i] = jaccard_tabular
    
    # 保存 Jaccard 矩阵
    jaccard_df = pd.DataFrame({
        'seed_i': [],
        'seed_j': [],
        'jaccard_fused': [],
        'jaccard_graph': [],
        'jaccard_tabular': []
    })
    
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            jaccard_df = pd.concat([jaccard_df, pd.DataFrame({
                'seed_i': [seeds[i]],
                'seed_j': [seeds[j]],
                'jaccard_fused': [jaccard_matrix_fused[i, j]],
                'jaccard_graph': [jaccard_matrix_graph[i, j]],
                'jaccard_tabular': [jaccard_matrix_tabular[i, j]]
            })], ignore_index=True)
    
    jaccard_path = os.path.join(output_dir, "jaccard_similarity.csv")
    jaccard_df.to_csv(jaccard_path, index=False)
    logging.info(f"✅ Jaccard 相似度已保存: {jaccard_path}")
    
    # 计算平均相似度（只取上三角）
    avg_jaccard_fused = jaccard_matrix_fused[np.triu_indices(n_seeds, k=1)].mean()
    avg_jaccard_graph = jaccard_matrix_graph[np.triu_indices(n_seeds, k=1)].mean()
    avg_jaccard_tabular = jaccard_matrix_tabular[np.triu_indices(n_seeds, k=1)].mean()
    
    logging.info(f"平均 Jaccard 相似度 (Top-{config.evaluation.top_k}):")
    logging.info(f"  融合模型: {avg_jaccard_fused:.3f}")
    logging.info(f"  图模型:   {avg_jaccard_graph:.3f}")
    logging.info(f"  表格模型: {avg_jaccard_tabular:.3f}")
    
    # ==================== 2. 找出稳定检测的节点 ====================
    logging.info("\n查找稳定检测的异常节点...")
    
    # 所有 seed 的 Top-K 交集
    stable_nodes = set.intersection(*all_topk_indices)
    logging.info(f"稳定节点数量 (所有 {n_seeds} 个 seed 都识别): {len(stable_nodes)}")
    
    # 至少被 n-1 个 seed 识别的节点
    node_counts = {}
    for topk_set in all_topk_indices:
        for node in topk_set:
            node_counts[node] = node_counts.get(node, 0) + 1
    
    high_confidence_nodes = {node for node, count in node_counts.items() if count >= n_seeds - 1}
    logging.info(f"高置信节点 (至少 {n_seeds-1}/{n_seeds} 个 seed 识别): {len(high_confidence_nodes)}")
    
    # 保存稳定节点列表
    stable_df = pd.DataFrame({
        'node_index': list(stable_nodes),
        'detection_count': n_seeds
    })
    
    stable_path = os.path.join(output_dir, "stable_anomalies.csv")
    stable_df.to_csv(stable_path, index=False)
    logging.info(f"✅ 稳定节点列表已保存: {stable_path}")
    
    # ==================== 3. 生成汇总报告 ====================
    summary = {
        'n_seeds': n_seeds,
        'seeds': seeds,
        'top_k': config.evaluation.top_k,
        'jaccard_similarity': {
            'fused': {
                'mean': float(avg_jaccard_fused),
                'std': float(jaccard_matrix_fused[np.triu_indices(n_seeds, k=1)].std()),
                'min': float(jaccard_matrix_fused[np.triu_indices(n_seeds, k=1)].min()),
                'max': float(jaccard_matrix_fused[np.triu_indices(n_seeds, k=1)].max())
            },
            'graph': {
                'mean': float(avg_jaccard_graph),
                'std': float(jaccard_matrix_graph[np.triu_indices(n_seeds, k=1)].std()),
                'min': float(jaccard_matrix_graph[np.triu_indices(n_seeds, k=1)].min()),
                'max': float(jaccard_matrix_graph[np.triu_indices(n_seeds, k=1)].max())
            },
            'tabular': {
                'mean': float(avg_jaccard_tabular),
                'std': float(jaccard_matrix_tabular[np.triu_indices(n_seeds, k=1)].std()),
                'min': float(jaccard_matrix_tabular[np.triu_indices(n_seeds, k=1)].min()),
                'max': float(jaccard_matrix_tabular[np.triu_indices(n_seeds, k=1)].max())
            }
        },
        'stable_nodes': {
            'all_seeds': len(stable_nodes),
            'high_confidence': len(high_confidence_nodes),
            'ratio': len(stable_nodes) / config.evaluation.top_k if config.evaluation.top_k > 0 else 0.0
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(output_dir, "multi_seed_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ 汇总报告已保存: {summary_path}")
    
    # ==================== 4. 打印汇总表格 ====================
    logging.info("\n" + "=" * 60)
    logging.info("📊 多 Seed 稳定性汇总")
    logging.info("=" * 60)
    logging.info(f"运行 Seeds: {seeds}")
    logging.info(f"Top-K: {config.evaluation.top_k}")
    logging.info("")
    logging.info("Jaccard 相似度 (Mean ± Std):")
    logging.info(f"  融合模型: {avg_jaccard_fused:.3f} ± {jaccard_matrix_fused[np.triu_indices(n_seeds, k=1)].std():.3f}")
    logging.info(f"  图模型:   {avg_jaccard_graph:.3f} ± {jaccard_matrix_graph[np.triu_indices(n_seeds, k=1)].std():.3f}")
    logging.info(f"  表格模型: {avg_jaccard_tabular:.3f} ± {jaccard_matrix_tabular[np.triu_indices(n_seeds, k=1)].std():.3f}")
    logging.info("")
    logging.info("稳定异常节点:")
    logging.info(f"  所有 seed 交集: {len(stable_nodes)} ({len(stable_nodes)/config.evaluation.top_k*100:.1f}%)")
    logging.info(f"  高置信 (≥{n_seeds-1}/{n_seeds}): {len(high_confidence_nodes)} ({len(high_confidence_nodes)/config.evaluation.top_k*100:.1f}%)")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
