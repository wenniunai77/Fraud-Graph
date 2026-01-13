"""
无标签评估模块
实现 Fusion.md 中的三种无标签评估方法
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from scipy import stats

from configs import EvaluationConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


@dataclass
class StabilityResult:
    """稳定性评估结果"""
    jaccard_scores: Dict[int, float]  # k -> Jaccard@k
    mean_jaccard: float
    std_jaccard: float
    n_runs: int


@dataclass
class EvaluationReport:
    """完整评估报告"""
    stability: Optional[StabilityResult]
    score_distribution: Dict
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'stability': {
                'jaccard_scores': self.stability.jaccard_scores,
                'mean_jaccard': self.stability.mean_jaccard,
                'std_jaccard': self.stability.std_jaccard,
                'n_runs': self.stability.n_runs
            } if self.stability else None,
            'score_statistics': self.score_distribution,
            'metadata': self.metadata
        }


class StabilityEvaluator:
    """
    稳定性评估器
    
    核心思想：对模型多次随机初始化训练，
    比较 Top-K 异常检测结果的 Jaccard 相似度
    高相似度 → 模型结果稳定可靠
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.k_values = config.stability_k_values
    
    def compute_jaccard(self, set1: set, set2: set) -> float:
        """计算 Jaccard 相似度"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def evaluate(
        self,
        score_runs: List[np.ndarray]
    ) -> StabilityResult:
        """
        评估多次运行结果的稳定性
        
        Args:
            score_runs: 多次运行的分数列表
        
        Returns:
            StabilityResult
        """
        n_runs = len(score_runs)
        if n_runs < 2:
            logging.warning("稳定性评估需要至少2次运行结果")
            return StabilityResult(
                jaccard_scores={k: 0.0 for k in self.k_values},
                mean_jaccard=0.0,
                std_jaccard=0.0,
                n_runs=n_runs
            )
        
        jaccard_scores = defaultdict(list)
        
        for k in self.k_values:
            # 获取每次运行的 Top-K 集合
            topk_sets = []
            for scores in score_runs:
                topk_idx = np.argsort(-scores)[:k]
                topk_sets.append(set(topk_idx))
            
            # 计算所有配对的 Jaccard
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    jaccard = self.compute_jaccard(topk_sets[i], topk_sets[j])
                    jaccard_scores[k].append(jaccard)
        
        # 汇总结果
        result_jaccard = {}
        all_jaccard = []
        
        for k, scores in jaccard_scores.items():
            mean_k = np.mean(scores)
            result_jaccard[k] = mean_k
            all_jaccard.extend(scores)
        
        logging.info(f"稳定性评估完成. 平均 Jaccard: {np.mean(all_jaccard):.3f}")
        
        return StabilityResult(
            jaccard_scores=result_jaccard,
            mean_jaccard=float(np.mean(all_jaccard)),
            std_jaccard=float(np.std(all_jaccard)),
            n_runs=n_runs
        )


class ScoreDistributionAnalyzer:
    """
    分数分布分析器
    
    分析异常分数的分布特性，判断模型是否有区分度
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def analyze(
        self,
        scores: np.ndarray,
        name: str = "scores"
    ) -> Dict:
        """
        分析分数分布
        
        Args:
            scores: 异常分数
            name: 分数名称
        
        Returns:
            分析结果字典
        """
        result = {
            "name": name,
            "n_samples": len(scores),
            "statistics": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "range": float(np.max(scores) - np.min(scores)),
            },
            "percentiles": {
                f"p{p}": float(np.percentile(scores, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        # P3 修复: 对 skewness 和 kurtosis 做 NaN 兜底
        try:
            skewness = float(stats.skew(scores))
            if np.isnan(skewness) or np.isinf(skewness):
                skewness = 0.0
        except:
            skewness = 0.0
        
        try:
            kurtosis_val = float(stats.kurtosis(scores))
            if np.isnan(kurtosis_val) or np.isinf(kurtosis_val):
                kurtosis_val = 0.0
        except:
            kurtosis_val = 0.0
        
        result["statistics"]["skewness"] = skewness
        result["statistics"]["kurtosis"] = kurtosis_val
        
        # 计算分数分离度
        p95 = result["percentiles"]["p95"]
        p50 = result["percentiles"]["p50"]
        p5 = result["percentiles"]["p5"]
        
        # 分离度指标：高分与中位数之间的差距
        if result["statistics"]["std"] > 0:
            separation = (p95 - p50) / result["statistics"]["std"]
        else:
            separation = 0.0
        
        result["separation_score"] = float(separation)
        
        # P3 修复: 尾部权重计算加除零保护
        top5_mask = scores >= p95
        eps = 1e-10
        scores_sum = scores.sum()
        
        if scores_sum <= eps or top5_mask.sum() == 0:
            tail_weight = 0.0
        else:
            tail_weight = scores[top5_mask].sum() / scores_sum
        
        result["tail_weight"] = float(tail_weight)
        
        logging.info(
            f"分布分析 [{name}]: "
            f"均值={result['statistics']['mean']:.4f}, "
            f"标准差={result['statistics']['std']:.4f}, "
            f"分离度={separation:.2f}"
        )
        
        return result


class UnsupervisedEvaluator:
    """
    无监督评估器（整合）
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.stability_evaluator = StabilityEvaluator(config)
        self.distribution_analyzer = ScoreDistributionAnalyzer(config)
    
    def evaluate(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        score_runs: Optional[List[np.ndarray]] = None,
        top_k: int = 1000
    ) -> EvaluationReport:
        """
        完整评估
        
        Args:
            df: 数据 DataFrame
            scores: 最终异常分数
            score_runs: 多次运行的分数（用于稳定性评估）
            top_k: Top-K 数量
        
        Returns:
            完整评估报告
        """
        logging.info("=" * 50)
        logging.info("开始无监督评估")
        logging.info("=" * 50)
        
        # 1. 稳定性评估
        stability_result = None
        if score_runs is not None and len(score_runs) >= 2:
            stability_result = self.stability_evaluator.evaluate(score_runs)
        
        # 2. 分数分布分析
        distribution_result = self.distribution_analyzer.analyze(scores, "fused_scores")
        
        report = EvaluationReport(
            stability=stability_result,
            score_distribution=distribution_result,
            metadata={
                "top_k": top_k,
                "n_samples": len(scores)
            }
        )
        
        logging.info("无监督评估完成")
        
        return report
    
    def print_report(self, report: EvaluationReport):
        """打印评估报告"""
        print("\n" + "=" * 70)
        print("无标签评估报告")
        print("=" * 70)
        
        print(f"\n样本数: {report.metadata['n_samples']:,}")
        print(f"Top-K: {report.metadata['top_k']}")
        
        # 稳定性
        if report.stability:
            print("\n--- 稳定性评估 ---")
            print(f"  运行次数: {report.stability.n_runs}")
            print(f"  平均 Jaccard: {report.stability.mean_jaccard:.3f} ± {report.stability.std_jaccard:.3f}")
            for k, jaccard in report.stability.jaccard_scores.items():
                print(f"    Jaccard@{k}: {jaccard:.3f}")
        
        # 分数分布
        print("\n--- 分数分布 ---")
        dist = report.score_distribution
        stats = dist["statistics"]
        print(f"  均值: {stats['mean']:.4f}")
        print(f"  标准差: {stats['std']:.4f}")
        print(f"  偏度: {stats['skewness']:.2f}")
        print(f"  峰度: {stats['kurtosis']:.2f}")
        print(f"  分离度: {dist['separation_score']:.2f}")
        print(f"  尾部权重: {dist['tail_weight']:.1%}")
        
        print("\n--- 关键百分位数 ---")
        for k, v in dist["percentiles"].items():
            if k in ["p50", "p90", "p95", "p99"]:
                print(f"  {k}: {v:.4f}")
        
        print("\n" + "=" * 70)
