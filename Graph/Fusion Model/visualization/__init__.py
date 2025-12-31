"""
可视化模块
按主题组织的可视化功能

主题：
1. 模型效果 (model_performance) - 总体效果、训练曲线、模型对比
2. 融合分析 (fusion_analysis) - 融合策略、权重分布、融合效果
3. 特征贡献 (feature_contribution) - 特征重要性、模型贡献度
4. 异常分布 (anomaly_distribution) - 正常/异常节点分布、分数分布
"""

from .model_performance import (
    plot_training_curves,
    plot_model_comparison,
    plot_score_statistics
)

from .fusion_analysis import (
    plot_fusion_overview,
    plot_fusion_weights_analysis,
    plot_fusion_weights_distribution,  # 别名，兼容旧代码
    plot_model_agreement
)

from .feature_contribution import (
    plot_feature_importance,
    plot_model_contribution
)

from .anomaly_distribution import (
    plot_score_distributions,
    plot_anomaly_scatter,
    plot_topk_analysis
)

from .dashboard import create_comprehensive_report

from .utils import setup_style

__all__ = [
    # 样式设置
    "setup_style",
    # 模型效果
    "plot_training_curves",
    "plot_model_comparison", 
    "plot_score_statistics",
    # 融合分析
    "plot_fusion_overview",
    "plot_fusion_weights_analysis",
    "plot_fusion_weights_distribution",  # 别名
    "plot_model_agreement",
    # 特征贡献
    "plot_feature_importance",
    "plot_model_contribution",
    # 异常分布
    "plot_score_distributions",
    "plot_anomaly_scatter",
    "plot_topk_analysis",
    # 综合报告
    "create_comprehensive_report"
]
