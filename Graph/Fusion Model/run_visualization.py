"""
可视化脚本
读取训练保存的结果，独立进行可视化
支持调整参数重新生成图表，无需重新训练模型
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
from typing import Optional, Dict, Any, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置 matplotlib 非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    plot_fusion_weights_analysis,
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


class TrainingResults:
    """训练结果容器"""
    
    def __init__(self):
        # 分数
        self.graph_scores: Optional[np.ndarray] = None
        self.tabular_scores: Optional[np.ndarray] = None
        self.fused_scores: Optional[np.ndarray] = None
        self.fusion_weights: Optional[np.ndarray] = None
        
        # 训练历史
        self.graph_train_losses: Optional[List[float]] = None
        self.tabular_train_info: Optional[Dict] = None
        
        # 融合信息
        self.fusion_strategy: Optional[str] = None
        self.fusion_config: Optional[Dict] = None
        
        # 数据信息
        self.node_degrees: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.tabular_features: Optional[np.ndarray] = None
        
        # 评估报告
        self.evaluation_report: Optional[Dict] = None
        
        # 元信息
        self.meta_info: Optional[Dict] = None
        self.training_config: Optional[Dict] = None
        self.timestamp: Optional[str] = None
    
    @classmethod
    def load(cls, results_dir: str) -> "TrainingResults":
        """从目录加载训练结果"""
        results = cls()
        
        # 1. 加载分数 (必需)
        scores_path = os.path.join(results_dir, "training_results.pkl")
        if os.path.exists(scores_path):
            with open(scores_path, 'rb') as f:
                data = pickle.load(f)
            
            results.graph_scores = data.get('graph_scores')
            results.tabular_scores = data.get('tabular_scores')
            results.fused_scores = data.get('fused_scores')
            results.fusion_weights = data.get('fusion_weights')
            results.graph_train_losses = data.get('graph_train_losses')
            results.tabular_train_info = data.get('tabular_train_info')
            results.node_degrees = data.get('node_degrees')
            results.feature_names = data.get('feature_names')
            results.tabular_features = data.get('tabular_features')
            results.fusion_strategy = data.get('fusion_strategy')
            results.fusion_config = data.get('fusion_config')
            results.evaluation_report = data.get('evaluation_report')
            results.meta_info = data.get('meta_info')
            results.timestamp = data.get('timestamp')
            
            logging.info(f"从 {scores_path} 加载训练结果成功")
        else:
            # 尝试从 CSV 加载 (向后兼容)
            csv_path = os.path.join(results_dir, "fusion_scores.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                results.graph_scores = df['graph_score'].values
                results.tabular_scores = df['tabular_score'].values
                results.fused_scores = df['fused_score'].values
                if 'fusion_weight' in df.columns:
                    results.fusion_weights = df['fusion_weight'].values
                logging.info(f"从 {csv_path} 加载分数成功 (向后兼容模式)")
            else:
                raise FileNotFoundError(f"找不到训练结果: {scores_path} 或 {csv_path}")
        
        # 2. 加载训练配置
        config_path = os.path.join(results_dir, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                results.training_config = json.load(f)
            if results.fusion_strategy is None:
                results.fusion_strategy = results.training_config.get('fusion_strategy')
        
        return results


class VisualizationPipeline:
    """可视化流水线"""
    
    def __init__(
        self,
        results: TrainingResults,
        output_dir: str,
        top_k: int = 1000,
        dpi: int = 150,
        figsize_scale: float = 1.0
    ):
        self.results = results
        self.output_dir = output_dir
        self.top_k = top_k
        self.dpi = dpi
        self.figsize_scale = figsize_scale
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_all(self):
        """运行所有可视化"""
        setup_style()
        
        logging.info("=" * 60)
        logging.info("开始生成可视化")
        logging.info("=" * 60)
        
        self.plot_model_performance()
        self.plot_fusion_analysis()
        self.plot_feature_contribution()
        self.plot_anomaly_distribution()
        self.create_report()
        
        logging.info("=" * 60)
        logging.info(f"✓ 所有可视化结果已保存到: {self.output_dir}")
        logging.info("=" * 60)
    
    def plot_model_performance(self):
        """模型性能可视化"""
        logging.info("绘制模型性能图 (1/5)...")
        
        r = self.results
        
        # 1. 训练曲线
        if r.graph_train_losses:
            plot_training_curves(
                graph_losses=r.graph_train_losses,
                save_path=os.path.join(self.output_dir, "training_curves.png")
            )
            plt.close('all')
            logging.info("  ✓ 训练曲线已保存")
        
        # 2. 模型对比
        if r.graph_scores is not None and r.tabular_scores is not None:
            plot_model_comparison(
                graph_scores=r.graph_scores,
                tabular_scores=r.tabular_scores,
                fused_scores=r.fused_scores,
                top_k=self.top_k,
                save_path=os.path.join(self.output_dir, "model_comparison.png")
            )
            plt.close('all')
            logging.info("  ✓ 模型对比已保存")
        
        # 3. 分数统计
        if r.fused_scores is not None:
            plot_score_statistics(
                graph_scores=r.graph_scores,
                tabular_scores=r.tabular_scores,
                fused_scores=r.fused_scores,
                save_path=os.path.join(self.output_dir, "score_statistics.png")
            )
            plt.close('all')
            logging.info("  ✓ 分数统计已保存")
    
    def plot_fusion_analysis(self):
        """融合分析可视化"""
        logging.info("绘制融合分析图 (2/5)...")
        
        r = self.results
        
        # 1. 融合概览
        plot_fusion_overview(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            fusion_weights=r.fusion_weights,
            strategy=r.fusion_strategy or "unknown",
            save_path=os.path.join(self.output_dir, "fusion_overview.png")
        )
        plt.close('all')
        logging.info("  ✓ 融合概览已保存")
        
        # 2. 融合权重分布
        if r.fusion_weights is not None:
            plot_fusion_weights_distribution(
                fusion_weights=r.fusion_weights,
                node_degrees=r.node_degrees,
                save_path=os.path.join(self.output_dir, "fusion_weights_distribution.png")
            )
            plt.close('all')
            logging.info("  ✓ 融合权重分布已保存")
            
            # 2b. 融合权重分析
            if r.node_degrees is not None:
                try:
                    plot_fusion_weights_analysis(
                        fusion_weights=r.fusion_weights,
                        node_degrees=r.node_degrees,
                        fused_scores=r.fused_scores,
                        save_path=os.path.join(self.output_dir, "fusion_weights_analysis.png")
                    )
                    plt.close('all')
                    logging.info("  ✓ 融合权重分析已保存")
                except Exception as e:
                    logging.warning(f"  ⚠ 融合权重分析失败: {e}")
        
        # 3. 模型一致性分析
        plot_model_agreement(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            top_k=self.top_k,
            save_path=os.path.join(self.output_dir, "model_agreement.png")
        )
        plt.close('all')
        logging.info("  ✓ 模型一致性已保存")
    
    def plot_feature_contribution(self):
        """特征贡献可视化"""
        logging.info("绘制特征贡献图 (3/5)...")
        
        r = self.results
        
        # 1. 特征重要性
        if r.feature_names and r.tabular_features is not None:
            plot_feature_importance(
                tabular_features=r.tabular_features,
                tabular_scores=r.tabular_scores,
                feature_names=r.feature_names,
                save_path=os.path.join(self.output_dir, "feature_importance.png")
            )
            plt.close('all')
            logging.info("  ✓ 特征重要性已保存")
        else:
            logging.info("  - 跳过特征重要性 (缺少特征信息)")
        
        # 2. 模型贡献分析
        plot_model_contribution(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            fusion_weights=r.fusion_weights,
            node_degrees=r.node_degrees,
            top_k=self.top_k,
            save_path=os.path.join(self.output_dir, "model_contribution.png")
        )
        plt.close('all')
        logging.info("  ✓ 模型贡献已保存")
    
    def plot_anomaly_distribution(self):
        """异常分布可视化"""
        logging.info("绘制异常分布图 (4/5)...")
        
        r = self.results
        
        # 1. 分数分布
        plot_score_distributions(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            save_path=os.path.join(self.output_dir, "score_distributions.png")
        )
        plt.close('all')
        logging.info("  ✓ 分数分布已保存")
        
        # 2. 异常散点图
        plot_anomaly_scatter(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            top_k=self.top_k,
            save_path=os.path.join(self.output_dir, "anomaly_scatter.png")
        )
        plt.close('all')
        logging.info("  ✓ 异常散点图已保存")
        
        # 3. Top-K 分析
        plot_topk_analysis(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            k_values=[100, 200, 500, 1000, 2000],
            save_path=os.path.join(self.output_dir, "topk_analysis.png")
        )
        plt.close('all')
        logging.info("  ✓ Top-K分析已保存")
    
    def create_report(self):
        """创建综合报告"""
        logging.info("创建综合报告 (5/5)...")
        
        r = self.results
        
        create_comprehensive_report(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            output_dir=self.output_dir,
            graph_train_losses=r.graph_train_losses,
            fusion_weights=r.fusion_weights,
            node_degrees=r.node_degrees,
            fusion_strategy=r.fusion_strategy or "unknown",
            top_k=self.top_k
        )
        plt.close('all')
        logging.info("  ✓ 综合报告已保存")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="可视化脚本 - 读取训练结果生成图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数可视化
  python run_visualization.py --results ./output
  
  # 自定义 Top-K 和输出目录
  python run_visualization.py --results ./output --top-k 500 --output ./new_viz
  
  # 只生成特定类型的图表
  python run_visualization.py --results ./output --only performance fusion
"""
    )
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="训练结果目录 (包含 training_results.pkl 或 fusion_scores.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可视化输出目录 (默认: <results>/visualizations_<timestamp>)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Top-K 异常数量 (默认: 1000)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="图像 DPI (默认: 150)"
    )
    
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=["performance", "fusion", "feature", "distribution", "report"],
        help="只生成指定类型的图表"
    )
    
    args = parser.parse_args()
    
    # 加载训练结果
    logging.info(f"加载训练结果: {args.results}")
    results = TrainingResults.load(args.results)
    
    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.results, f"visualizations_{timestamp}")
    
    # 创建可视化流水线
    pipeline = VisualizationPipeline(
        results=results,
        output_dir=output_dir,
        top_k=args.top_k,
        dpi=args.dpi
    )
    
    # 运行可视化
    if args.only:
        setup_style()
        if "performance" in args.only:
            pipeline.plot_model_performance()
        if "fusion" in args.only:
            pipeline.plot_fusion_analysis()
        if "feature" in args.only:
            pipeline.plot_feature_contribution()
        if "distribution" in args.only:
            pipeline.plot_anomaly_distribution()
        if "report" in args.only:
            pipeline.create_report()
        logging.info(f"✓ 选定的可视化结果已保存到: {output_dir}")
    else:
        pipeline.run_all()


if __name__ == "__main__":
    main()
