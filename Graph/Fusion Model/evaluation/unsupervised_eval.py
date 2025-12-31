"""
无标签评估模块
实现 Fusion.md 中的三种无标签评估方法
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
from collections import defaultdict
from scipy import stats

if TYPE_CHECKING:
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
class WeakRuleResult:
    """弱规则评估结果"""
    rule_name: str
    enrichment_ratio: float  # Top-K 中满足规则的比例 / 全集比例
    topk_hit_rate: float     # Top-K 中满足规则的比例
    population_rate: float   # 全集中满足规则的比例
    lift: float              # 提升度


@dataclass
class EvaluationReport:
    """完整评估报告"""
    stability: Optional[StabilityResult]
    weak_rules: List[WeakRuleResult]
    score_distribution: Dict
    metadata: Dict


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


class WeakRuleEvaluator:
    """
    弱规则评估器
    
    核心思想：使用领域知识定义"弱标签"规则，
    评估异常检测结果是否与这些规则一致
    
    例如：
    - 大额交易更可能是异常
    - 首次交易对手更可疑
    - 高频交易更可能是异常
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.rules: List[Tuple[str, Callable]] = []
    
    def add_rule(self, name: str, rule_func: Callable[[pd.DataFrame], np.ndarray]):
        """
        添加弱规则
        
        Args:
            name: 规则名称
            rule_func: 规则函数，输入 DataFrame，返回布尔数组
        """
        self.rules.append((name, rule_func))
        logging.info(f"添加弱规则: {name}")
    
    def add_amount_rule(self, df: pd.DataFrame, amount_col: str, percentile: float = 99):
        """添加大额交易规则"""
        threshold = np.percentile(df[amount_col], percentile)
        
        def rule(data: pd.DataFrame) -> np.ndarray:
            return (data[amount_col] >= threshold).values
        
        self.add_rule(f"large_amount_p{percentile}", rule)
    
    def add_frequency_rule(
        self, 
        df: pd.DataFrame, 
        entity_col: str, 
        threshold_percentile: float = 95
    ):
        """添加高频交易规则"""
        freq = df[entity_col].value_counts()
        threshold = np.percentile(freq.values, threshold_percentile)
        high_freq_entities = set(freq[freq >= threshold].index)
        
        def rule(data: pd.DataFrame) -> np.ndarray:
            return data[entity_col].isin(high_freq_entities).values
        
        self.add_rule(f"high_freq_{entity_col}", rule)
    
    def evaluate(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        top_k: int = 1000
    ) -> List[WeakRuleResult]:
        """
        评估弱规则
        
        Args:
            df: 数据 DataFrame
            scores: 异常分数
            top_k: Top-K 数量
        
        Returns:
            弱规则评估结果列表
        """
        if len(self.rules) == 0:
            logging.warning("没有定义弱规则")
            return []
        
        n = len(scores)
        topk_idx = np.argsort(-scores)[:top_k]
        topk_mask = np.zeros(n, dtype=bool)
        topk_mask[topk_idx] = True
        
        results = []
        
        for name, rule_func in self.rules:
            try:
                rule_mask = rule_func(df)
                
                # 全集中满足规则的比例
                population_rate = rule_mask.sum() / n
                
                # Top-K 中满足规则的比例
                topk_hit_rate = rule_mask[topk_idx].sum() / top_k
                
                # 富集比率
                if population_rate > 0:
                    enrichment_ratio = topk_hit_rate / population_rate
                else:
                    enrichment_ratio = 0.0
                
                # 提升度 (Lift)
                lift = enrichment_ratio
                
                result = WeakRuleResult(
                    rule_name=name,
                    enrichment_ratio=float(enrichment_ratio),
                    topk_hit_rate=float(topk_hit_rate),
                    population_rate=float(population_rate),
                    lift=float(lift)
                )
                results.append(result)
                
                logging.info(
                    f"弱规则 [{name}]: "
                    f"命中率={topk_hit_rate:.1%}, "
                    f"基准={population_rate:.1%}, "
                    f"提升度={lift:.2f}x"
                )
                
            except Exception as e:
                logging.error(f"弱规则 [{name}] 评估失败: {e}")
        
        return results


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
                "skewness": float(stats.skew(scores)),
                "kurtosis": float(stats.kurtosis(scores))
            },
            "percentiles": {
                f"p{p}": float(np.percentile(scores, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        }
        
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
        
        # 尾部权重（高于 P95 的样本所占分数比例）
        top5_mask = scores >= p95
        if top5_mask.sum() > 0:
            tail_weight = scores[top5_mask].sum() / scores.sum()
        else:
            tail_weight = 0.0
        
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
        self.weak_rule_evaluator = WeakRuleEvaluator(config)
        self.distribution_analyzer = ScoreDistributionAnalyzer(config)
    
    def add_weak_rule(self, name: str, rule_func: Callable):
        """添加弱规则"""
        self.weak_rule_evaluator.add_rule(name, rule_func)
    
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
        
        # 2. 弱规则评估
        weak_rule_results = self.weak_rule_evaluator.evaluate(df, scores, top_k)
        
        # 3. 分数分布分析
        distribution_result = self.distribution_analyzer.analyze(scores, "fused_scores")
        
        report = EvaluationReport(
            stability=stability_result,
            weak_rules=weak_rule_results,
            score_distribution=distribution_result,
            metadata={
                "top_k": top_k,
                "n_samples": len(scores),
                "n_rules": len(weak_rule_results)
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
        
        # 弱规则
        if report.weak_rules:
            print("\n--- 弱规则评估 ---")
            for rule in report.weak_rules:
                print(f"  [{rule.rule_name}]")
                print(f"    Top-K 命中率: {rule.topk_hit_rate:.1%}")
                print(f"    基准命中率: {rule.population_rate:.1%}")
                print(f"    提升度: {rule.lift:.2f}x")
        
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


def create_default_weak_rules(
    df: pd.DataFrame,
    amount_col: str = "payment_amount",
    entity_cols: Optional[List[str]] = None
) -> WeakRuleEvaluator:
    """
    创建默认弱规则集
    
    Args:
        df: 数据 DataFrame
        amount_col: 金额列名
        entity_cols: 实体列名列表
    
    Returns:
        配置好的 WeakRuleEvaluator
    """
    from configs import EvaluationConfig
    
    config = EvaluationConfig()
    evaluator = WeakRuleEvaluator(config)
    
    # 大额交易规则
    if amount_col in df.columns:
        for p in [95, 99]:
            threshold = np.percentile(df[amount_col], p)
            
            def make_rule(thresh):
                def rule(data: pd.DataFrame) -> np.ndarray:
                    return (data[amount_col] >= thresh).values
                return rule
            
            evaluator.add_rule(f"large_amount_p{p}", make_rule(threshold))
    
    # 高频实体规则
    if entity_cols:
        for col in entity_cols:
            if col in df.columns:
                freq = df[col].value_counts()
                threshold = np.percentile(freq.values, 95)
                high_freq = set(freq[freq >= threshold].index)
                
                def make_freq_rule(entities, column):
                    def rule(data: pd.DataFrame) -> np.ndarray:
                        return data[column].isin(entities).values
                    return rule
                
                evaluator.add_rule(f"high_freq_{col}", make_freq_rule(high_freq, col))
    
    return evaluator
