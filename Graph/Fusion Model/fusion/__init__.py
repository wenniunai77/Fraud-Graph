"""
融合策略模块
"""
from .fusion import (
    FusionStrategy,
    FusionResult,  # 添加: 缺失的 FusionResult 导出
    GatedFusion,
    WeightedFusion,
    RankFusion,
    ConsistentFusion,
    EnsembleFusion,  # 添加: 缺失的 EnsembleFusion 导出
    create_fusion_strategy,
    analyze_fusion,      # 添加: run_fusion.py 需要的函数
    print_fusion_report  # 添加: run_fusion.py 需要的函数
)

__all__ = [
    "FusionStrategy",
    "FusionResult",
    "GatedFusion",
    "WeightedFusion",
    "RankFusion",
    "ConsistentFusion",
    "EnsembleFusion",
    "create_fusion_strategy",
    "analyze_fusion",
    "print_fusion_report"
]
