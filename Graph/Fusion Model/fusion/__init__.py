"""
融合策略模块
"""
from .fusion import (
    FusionStrategy,
    GatedFusion,
    WeightedFusion,
    RankFusion,
    ConsistentFusion,
    create_fusion_strategy
)

__all__ = [
    "FusionStrategy",
    "GatedFusion",
    "WeightedFusion",
    "RankFusion",
    "ConsistentFusion",
    "create_fusion_strategy"
]
