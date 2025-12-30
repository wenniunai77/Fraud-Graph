"""
工具函数模块
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """获取计算设备"""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logging.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logging.info("使用 CPU")
        return device
    except ImportError:
        return "cpu"


def save_results(
    results: Dict,
    output_dir: str,
    prefix: str = "results"
):
    """保存结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    
    logging.info(f"结果已保存: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict:
    """从 JSON 加载结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_topk_overlap(
    scores1: np.ndarray,
    scores2: np.ndarray,
    k_values: List[int] = [100, 500, 1000]
) -> Dict[int, float]:
    """
    计算两组分数的 Top-K 重叠率
    
    Args:
        scores1: 第一组分数
        scores2: 第二组分数
        k_values: K 值列表
    
    Returns:
        K -> 重叠率 字典
    """
    result = {}
    
    for k in k_values:
        topk1 = set(np.argsort(-scores1)[:k])
        topk2 = set(np.argsort(-scores2)[:k])
        overlap = len(topk1 & topk2) / k
        result[k] = overlap
    
    return result


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    分数归一化
    
    Args:
        scores: 原始分数
        method: 归一化方法 ('minmax', 'zscore', 'rank')
    
    Returns:
        归一化后的分数
    """
    if method == "minmax":
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s > 1e-8:
            return (scores - min_s) / (max_s - min_s)
        return np.zeros_like(scores)
    
    elif method == "zscore":
        mean_s = scores.mean()
        std_s = scores.std()
        if std_s > 1e-8:
            return (scores - mean_s) / std_s
        return np.zeros_like(scores)
    
    elif method == "rank":
        from scipy import stats
        return stats.rankdata(scores) / len(scores)
    
    else:
        raise ValueError(f"未知归一化方法: {method}")


def compute_score_statistics(scores: np.ndarray) -> Dict:
    """计算分数统计量"""
    from scipy import stats as scipy_stats
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "skewness": float(scipy_stats.skew(scores)),
        "kurtosis": float(scipy_stats.kurtosis(scores)),
        "percentiles": {
            f"p{p}": float(np.percentile(scores, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
    }


def export_topk_to_csv(
    df: pd.DataFrame,
    scores: np.ndarray,
    output_path: str,
    k: int = 1000,
    score_name: str = "anomaly_score"
):
    """
    导出 Top-K 异常到 CSV
    
    Args:
        df: 原始数据
        scores: 异常分数
        output_path: 输出路径
        k: Top-K 数量
        score_name: 分数列名
    """
    topk_idx = np.argsort(-scores)[:k]
    
    result_df = df.iloc[topk_idx].copy()
    result_df[score_name] = scores[topk_idx]
    result_df["rank"] = range(1, k + 1)
    
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Top-{k} 异常已导出: {output_path}")


def merge_scores(
    score_dicts: List[Dict[str, np.ndarray]],
    weights: Optional[List[float]] = None,
    method: str = "weighted"
) -> np.ndarray:
    """
    合并多组分数
    
    Args:
        score_dicts: 分数字典列表，每个字典包含 {'name': scores}
        weights: 权重列表
        method: 合并方法 ('weighted', 'max', 'mean', 'rank')
    
    Returns:
        合并后的分数
    """
    all_scores = []
    for d in score_dicts:
        for name, scores in d.items():
            all_scores.append(normalize_scores(scores))
    
    n = len(all_scores)
    if weights is None:
        weights = [1.0 / n] * n
    
    if method == "weighted":
        result = np.zeros_like(all_scores[0])
        for scores, w in zip(all_scores, weights):
            result += w * scores
        return result
    
    elif method == "max":
        return np.max(np.stack(all_scores), axis=0)
    
    elif method == "mean":
        return np.mean(np.stack(all_scores), axis=0)
    
    elif method == "rank":
        rank_sum = np.zeros_like(all_scores[0])
        from scipy import stats
        for scores in all_scores:
            rank_sum += stats.rankdata(scores)
        return rank_sum / n
    
    else:
        raise ValueError(f"未知合并方法: {method}")


class Timer:
    """计时器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        if self.name:
            logging.info(f"[{self.name}] 耗时: {duration:.2f} 秒")
    
    @property
    def duration(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def format_number(n: float, precision: int = 4) -> str:
    """格式化数字输出"""
    if abs(n) >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def print_separator(title: str = "", char: str = "=", width: int = 60):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(char * padding + f" {title} " + char * padding)
    else:
        print(char * width)
