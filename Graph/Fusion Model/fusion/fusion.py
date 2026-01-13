"""
融合策略实现
基于 Fusion.md 中的方法论实现多种融合策略
"""
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats

from configs import FusionConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


@dataclass
class FusionResult:
    """融合结果"""
    fused_scores: np.ndarray
    graph_scores: np.ndarray
    tabular_scores: np.ndarray
    fusion_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class FusionStrategy(ABC):
    """融合策略基类"""
    
    @abstractmethod
    def fuse(
        self,
        graph_scores: np.ndarray,
        tabular_scores: np.ndarray,
        **kwargs
    ) -> FusionResult:
        """融合两路异常分数"""
        pass
    
    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """归一化分数到 [0, 1]"""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s > 1e-8:
            return (scores - min_s) / (max_s - min_s)
        return np.zeros_like(scores)


class GatedFusion(FusionStrategy):
    """
    门控融合（基于活跃度）
    
    对于活跃节点（高度数）：更信任图模型
    对于冷启动节点（低度数）：更信任表格模型
    
    公式: score_fused = α * score_graph + (1-α) * score_tabular
    其中 α 根据节点活跃度动态调整
    """
    
    def __init__(
        self,
        config: FusionConfig,
        alpha_high: Optional[float] = None,
        alpha_low: Optional[float] = None,
        degree_threshold: Optional[int] = None,
        use_hard_threshold: Optional[bool] = None,
        sigmoid_steepness: Optional[float] = None
    ):
        self.config = config
        # 从 config 读取参数，允许显式传入覆盖
        self.alpha_high = alpha_high if alpha_high is not None else config.alpha_high
        self.alpha_low = alpha_low if alpha_low is not None else config.alpha_low
        self.degree_threshold = degree_threshold if degree_threshold is not None else config.degree_threshold
        self.use_hard_threshold = use_hard_threshold if use_hard_threshold is not None else config.use_hard_threshold
        self.sigmoid_steepness = sigmoid_steepness if sigmoid_steepness is not None else config.sigmoid_steepness
    
    def fuse(
        self,
        graph_scores: np.ndarray,
        tabular_scores: np.ndarray,
        node_degrees: Optional[np.ndarray] = None,
        **kwargs
    ) -> FusionResult:
        """
        门控融合
        
        Args:
            graph_scores: 图模型分数
            tabular_scores: 表格模型分数
            node_degrees: 节点度数（用于门控）
        """
        # 归一化：门控建议在 rank/quantile 空间里做（更鲁棒）
        # 兼容旧行为：默认仍使用 min-max
        score_space = getattr(self.config, "gated_score_space", "minmax")
        score_space = (score_space or "minmax").lower()

        if score_space == "rank":
            n = len(graph_scores)
            # rankdata 返回 1..n，这里归一化到 (0,1]；再减去 1/n 变为 [0,1)
            # 不用 min-max，避免重尾分布被极值拉伸
            g_norm = (stats.rankdata(graph_scores) - 1) / max(n - 1, 1)
            t_norm = (stats.rankdata(tabular_scores) - 1) / max(n - 1, 1)
            logging.info("GatedFusion 使用 rank 分数空间进行门控融合")
        elif score_space == "minmax":
            g_norm = self.normalize_scores(graph_scores)
            t_norm = self.normalize_scores(tabular_scores)
        else:
            raise ValueError(f"Unknown gated_score_space: {score_space}. Use 'minmax' or 'rank'.")

        n = len(graph_scores)
        
        if node_degrees is not None:
            # 基于度数计算动态权重
            # 统计度数分布
            logging.info(f"度数分布: min={node_degrees.min()}, max={node_degrees.max()}, "
                        f"median={np.median(node_degrees):.1f}, mean={node_degrees.mean():.1f}")
            logging.info(f"度数阈值: {self.degree_threshold}")
            
            # 统计活跃/非活跃边数量（基于阈值）
            inactive_count = (node_degrees < self.degree_threshold).sum()
            active_count = (node_degrees >= self.degree_threshold).sum()
            logging.info(f"边活跃度分类 (基于 min(src_out_deg, dst_in_deg)): "
                        f"非活跃(<{self.degree_threshold})={inactive_count} ({inactive_count/n*100:.1f}%), "
                        f"活跃(≥{self.degree_threshold})={active_count} ({active_count/n*100:.1f}%)")
            
            if self.use_hard_threshold:
                # 硬阈值模式：二元分类
                alpha = np.where(
                    node_degrees < self.degree_threshold,
                    self.alpha_low,   # 度数 < 阈值: 非活跃边，用低权重
                    self.alpha_high   # 度数 >= 阈值: 活跃边，用高权重
                )
                logging.info(f"使用硬阈值模式: 非活跃α={self.alpha_low}, 活跃α={self.alpha_high}")
            else:
                # 平滑过渡模式：以 degree_threshold 为拐点的 sigmoid 平滑
                # 公式: α = α_low + (α_high - α_low) * sigmoid(k * (deg - threshold))
                # k 控制平滑陡峭程度，从 config 读取
                from scipy.special import expit  # 数值稳定的 sigmoid
                x = node_degrees.astype(np.float64) - self.degree_threshold
                sigmoid = expit(self.sigmoid_steepness * x)
                alpha = self.alpha_low + (self.alpha_high - self.alpha_low) * sigmoid
                logging.info(f"使用平滑过渡模式: 拐点={self.degree_threshold}, "
                           f"陡峭度={self.sigmoid_steepness}, α 范围约 [{self.alpha_low:.2f}, {self.alpha_high:.2f}]")
        else:
            # 如果没有度数信息，使用固定权重
            alpha = np.full(n, (self.alpha_high + self.alpha_low) / 2)
        
        # 融合
        fused_scores = alpha * g_norm + (1 - alpha) * t_norm
        
        logging.info(f"门控融合完成. α 范围: [{alpha.min():.3f}, {alpha.max():.3f}]")
        
        return FusionResult(
            # rank 空间的 fused_scores 本身就在 [0,1]，这里保持一致性不再二次 min-max
            fused_scores=fused_scores if score_space == "rank" else self.normalize_scores(fused_scores),
            graph_scores=g_norm,
            tabular_scores=t_norm,
            fusion_weights=alpha,
            metadata={
                "strategy": "gated",
                "degree_threshold": self.degree_threshold,
                "gated_score_space": score_space,
            }
        )


class WeightedFusion(FusionStrategy):
    """
    加权融合
    
    简单加权平均: score_fused = α * score_graph + (1-α) * score_tabular
    """
    
    def __init__(
        self,
        config: FusionConfig,
        alpha: float = 0.5
    ):
        self.config = config
        self.alpha = alpha  # 图模型权重
    
    def fuse(
        self,
        graph_scores: np.ndarray,
        tabular_scores: np.ndarray,
        **kwargs
    ) -> FusionResult:
        """加权融合"""
        g_norm = self.normalize_scores(graph_scores)
        t_norm = self.normalize_scores(tabular_scores)
        
        fused_scores = self.alpha * g_norm + (1 - self.alpha) * t_norm
        
        logging.info(f"加权融合完成. α = {self.alpha}")
        
        return FusionResult(
            fused_scores=self.normalize_scores(fused_scores),
            graph_scores=g_norm,
            tabular_scores=t_norm,
            fusion_weights=np.full(len(graph_scores), self.alpha),
            metadata={"strategy": "weighted", "alpha": self.alpha}
        )


class RankFusion(FusionStrategy):
    """
    排名融合
    
    将分数转换为排名，再进行融合
    更鲁棒，不受分数分布影响
    """
    
    def __init__(
        self,
        config: FusionConfig,
        alpha: float = 0.5
    ):
        self.config = config
        self.alpha = alpha
    
    def fuse(
        self,
        graph_scores: np.ndarray,
        tabular_scores: np.ndarray,
        **kwargs
    ) -> FusionResult:
        """排名融合"""
        n = len(graph_scores)
        
        # 转换为排名（0-1之间）
        g_rank = stats.rankdata(graph_scores) / n
        t_rank = stats.rankdata(tabular_scores) / n
        
        # 融合排名
        fused_rank = self.alpha * g_rank + (1 - self.alpha) * t_rank
        
        logging.info(f"排名融合完成. α = {self.alpha}")
        
        return FusionResult(
            fused_scores=self.normalize_scores(fused_rank),
            graph_scores=g_rank,
            tabular_scores=t_rank,
            fusion_weights=np.full(n, self.alpha),
            metadata={"strategy": "rank", "alpha": self.alpha}
        )


class ConsistentFusion(FusionStrategy):
    """
    一致性融合
    
    当两个模型都给出高分时增强信号
    当两个模型不一致时降低置信度
    
    公式: score_fused = sqrt(score_graph * score_tabular) * agreement_factor
    """
    
    def __init__(
        self,
        config: FusionConfig,
        consistency_weight: float = 0.3
    ):
        self.config = config
        self.consistency_weight = consistency_weight
    
    def fuse(
        self,
        graph_scores: np.ndarray,
        tabular_scores: np.ndarray,
        **kwargs
    ) -> FusionResult:
        """一致性融合"""
        g_norm = self.normalize_scores(graph_scores)
        t_norm = self.normalize_scores(tabular_scores)
        
        # 几何平均
        geometric_mean = np.sqrt(g_norm * t_norm + 1e-8)
        
        # 一致性度量（两个分数越接近，一致性越高）
        consistency = 1 - np.abs(g_norm - t_norm)
        
        # 融合：几何平均 + 一致性加成
        fused_scores = geometric_mean * (1 + self.consistency_weight * consistency)
        
        logging.info(f"一致性融合完成. 平均一致性: {consistency.mean():.3f}")
        
        return FusionResult(
            fused_scores=self.normalize_scores(fused_scores),
            graph_scores=g_norm,
            tabular_scores=t_norm,
            fusion_weights=consistency,
            metadata={"strategy": "consistent", "mean_consistency": float(consistency.mean())}
        )


class EnsembleFusion(FusionStrategy):
    """
    集成融合
    
    组合多种融合策略的结果
    """
    
    def __init__(
        self,
        config: FusionConfig,
        strategies: List[FusionStrategy],
        weights: Optional[List[float]] = None
    ):
        self.config = config
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
    
    def fuse(
        self,
        graph_scores: np.ndarray,
        tabular_scores: np.ndarray,
        **kwargs
    ) -> FusionResult:
        """集成融合"""
        all_fused = []
        
        for strategy in self.strategies:
            result = strategy.fuse(graph_scores, tabular_scores, **kwargs)
            all_fused.append(result.fused_scores)
        
        # 加权平均
        fused_scores = np.zeros_like(graph_scores)
        for i, (scores, weight) in enumerate(zip(all_fused, self.weights)):
            fused_scores += weight * scores
        
        logging.info(f"集成融合完成. 策略数: {len(self.strategies)}")
        
        return FusionResult(
            fused_scores=self.normalize_scores(fused_scores),
            graph_scores=self.normalize_scores(graph_scores),
            tabular_scores=self.normalize_scores(tabular_scores),
            fusion_weights=np.array(self.weights),
            metadata={"strategy": "ensemble", "n_strategies": len(self.strategies)}
        )


def create_fusion_strategy(config: FusionConfig) -> FusionStrategy:
    """
    工厂函数：根据配置创建融合策略
    """
    strategy_type = config.strategy
    
    if strategy_type == "gated":
        return GatedFusion(
            config=config,
            alpha_high=config.alpha_high,
            alpha_low=config.alpha_low,
            degree_threshold=config.degree_threshold,
            use_hard_threshold=config.use_hard_threshold  # 新增参数
        )
    elif strategy_type == "weighted":
        return WeightedFusion(
            config=config,
            alpha=config.fusion_alpha
        )
    elif strategy_type == "rank":
        return RankFusion(
            config=config,
            alpha=config.fusion_alpha
        )
    elif strategy_type == "consistent":
        return ConsistentFusion(
            config=config,
            consistency_weight=config.consistency_weight
        )
    else:
        raise ValueError(f"未知的融合策略: {strategy_type}")


# ==================== 融合分析工具 ====================

def analyze_fusion(result: FusionResult, top_k: int = 1000) -> Dict:
    """
    分析融合结果
    
    Args:
        result: 融合结果
        top_k: 分析 top-k 结果
    
    Returns:
        分析报告字典
    """
    n = len(result.fused_scores)
    
    # 排名
    fused_rank = np.argsort(-result.fused_scores)
    graph_rank = np.argsort(-result.graph_scores)
    tabular_rank = np.argsort(-result.tabular_scores)
    
    # Top-K 集合
    fused_topk = set(fused_rank[:top_k])
    graph_topk = set(graph_rank[:top_k])
    tabular_topk = set(tabular_rank[:top_k])
    
    # 重叠分析
    graph_overlap = len(fused_topk & graph_topk) / top_k
    tabular_overlap = len(fused_topk & tabular_topk) / top_k
    both_overlap = len(fused_topk & graph_topk & tabular_topk) / top_k
    
    # 排名相关性
    graph_corr = stats.spearmanr(result.fused_scores, result.graph_scores)[0]
    tabular_corr = stats.spearmanr(result.fused_scores, result.tabular_scores)[0]
    g_t_corr = stats.spearmanr(result.graph_scores, result.tabular_scores)[0]
    
    # 分数分布
    report = {
        "n_samples": n,
        "top_k": top_k,
        "overlap": {
            "fused_graph": graph_overlap,
            "fused_tabular": tabular_overlap,
            "fused_both": both_overlap
        },
        "correlation": {
            "fused_graph": float(graph_corr),
            "fused_tabular": float(tabular_corr),
            "graph_tabular": float(g_t_corr)
        },
        "score_stats": {
            "fused": {
                "mean": float(result.fused_scores.mean()),
                "std": float(result.fused_scores.std()),
                "median": float(np.median(result.fused_scores)),
                "p95": float(np.percentile(result.fused_scores, 95)),
                "p99": float(np.percentile(result.fused_scores, 99))
            },
            "graph": {
                "mean": float(result.graph_scores.mean()),
                "std": float(result.graph_scores.std())
            },
            "tabular": {
                "mean": float(result.tabular_scores.mean()),
                "std": float(result.tabular_scores.std())
            }
        },
        "metadata": result.metadata
    }
    
    return report


def print_fusion_report(report: Dict):
    """打印融合分析报告"""
    print("\n" + "=" * 60)
    print("融合分析报告")
    print("=" * 60)
    
    print(f"\n样本数: {report['n_samples']:,}")
    print(f"Top-K: {report['top_k']}")
    
    print(f"\n策略: {report['metadata'].get('strategy', 'unknown')}")
    
    print("\n--- Top-K 重叠率 ---")
    print(f"  融合 ∩ 图模型: {report['overlap']['fused_graph']:.1%}")
    print(f"  融合 ∩ 表格模型: {report['overlap']['fused_tabular']:.1%}")
    print(f"  三者交集: {report['overlap']['fused_both']:.1%}")
    
    print("\n--- Spearman 相关性 ---")
    print(f"  融合 vs 图模型: {report['correlation']['fused_graph']:.3f}")
    print(f"  融合 vs 表格模型: {report['correlation']['fused_tabular']:.3f}")
    print(f"  图模型 vs 表格模型: {report['correlation']['graph_tabular']:.3f}")
    
    print("\n--- 融合分数分布 ---")
    stats = report['score_stats']['fused']
    print(f"  均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    print(f"  P95: {stats['p95']:.4f}")
    print(f"  P99: {stats['p99']:.4f}")
    
    print("\n" + "=" * 60)
