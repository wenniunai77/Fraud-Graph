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
        """运行所有可视化 - 统一使用结构化输出"""
        setup_style()
        
        logging.info("=" * 60)
        logging.info("开始生成可视化")
        logging.info("=" * 60)
        
        # 直接调用 dashboard 的综合报告，避免重复生成
        self.create_report()
        
        logging.info("=" * 60)
        logging.info(f"✓ 所有可视化结果已保存到: {self.output_dir}")
        logging.info("=" * 60)
    
    def create_report(self):
        """
        创建综合报告 - 统一的结构化输出
        生成所有可视化图表，按类别组织到子目录中
        """
        r = self.results
        
        create_comprehensive_report(
            graph_scores=r.graph_scores,
            tabular_scores=r.tabular_scores,
            fused_scores=r.fused_scores,
            output_dir=self.output_dir,
            graph_train_losses=r.graph_train_losses,
            fusion_weights=r.fusion_weights,
            node_degrees=r.node_degrees,
            tabular_features=r.tabular_features,
            feature_names=r.feature_names,
            fusion_strategy=r.fusion_strategy or "unknown",
            top_k=self.top_k
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="可视化脚本 - 读取训练结果生成图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数可视化（推荐，生成结构化报告）
  python run_visualization.py --results ./output
  
  # 自定义 Top-K 和输出目录
  python run_visualization.py --results ./output --top-k 500 --output ./new_viz
  
  # 只生成特定类型的图表（高级用法）
  python run_visualization.py --results ./output --only performance fusion
  
注意:
  默认会生成结构化的可视化报告，所有图表按类别组织在子目录中:
  - 1_model_performance/    模型性能
  - 2_fusion_analysis/      融合分析  
  - 3_feature_contribution/ 特征贡献
  - 4_anomaly_distribution/ 异常分布
  - summary.png             总体摘要
"""
    )
    
    parser.add_argument(
        "--results",
        type=str,
        default="./output",
        help="训练结果目录 (默认: ./output, 包含 training_results.pkl 或 fusion_scores.csv)"
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
