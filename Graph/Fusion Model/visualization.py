"""
可视化模块
提供融合异常检测结果的可视化功能
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


def plot_score_distributions(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    绘制分数分布对比图
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 图模型分数分布
    axes[0].hist(graph_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Graph Model Scores')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.percentile(graph_scores, 95), color='red', linestyle='--', label='P95')
    axes[0].legend()
    
    # 表格模型分数分布
    axes[1].hist(tabular_scores, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Tabular Model Scores')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.percentile(tabular_scores, 95), color='red', linestyle='--', label='P95')
    axes[1].legend()
    
    # 融合分数分布
    axes[2].hist(fused_scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[2].set_title('Fused Scores')
    axes[2].set_xlabel('Score')
    axes[2].set_ylabel('Frequency')
    axes[2].axvline(np.percentile(fused_scores, 95), color='red', linestyle='--', label='P95')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"分布图已保存: {save_path}")
    
    plt.show()


def plot_score_scatter(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: Optional[np.ndarray] = None,
    top_k: int = 1000,
    save_path: Optional[str] = None
):
    """
    绘制分数散点图
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数（用于着色）
        top_k: 高亮 Top-K
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 获取 Top-K 索引
    if fused_scores is not None:
        topk_idx = set(np.argsort(-fused_scores)[:top_k])
        colors = ['red' if i in topk_idx else 'blue' for i in range(len(graph_scores))]
        alphas = [0.8 if i in topk_idx else 0.1 for i in range(len(graph_scores))]
    else:
        colors = 'blue'
        alphas = 0.3
    
    scatter = ax.scatter(
        graph_scores, 
        tabular_scores, 
        c=colors,
        alpha=0.3,
        s=10
    )
    
    ax.set_xlabel('Graph Model Score', fontsize=12)
    ax.set_ylabel('Tabular Model Score', fontsize=12)
    ax.set_title(f'Score Scatter (Top-{top_k} in Red)', fontsize=14)
    
    # 添加对角线
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"散点图已保存: {save_path}")
    
    plt.show()


def plot_topk_overlap(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    k_values: List[int] = [100, 200, 500, 1000, 2000],
    save_path: Optional[str] = None
):
    """
    绘制 Top-K 重叠率曲线
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        k_values: K 值列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    overlaps_graph = []
    overlaps_tabular = []
    overlaps_both = []
    
    for k in k_values:
        graph_topk = set(np.argsort(-graph_scores)[:k])
        tabular_topk = set(np.argsort(-tabular_scores)[:k])
        fused_topk = set(np.argsort(-fused_scores)[:k])
        
        overlaps_graph.append(len(fused_topk & graph_topk) / k)
        overlaps_tabular.append(len(fused_topk & tabular_topk) / k)
        overlaps_both.append(len(fused_topk & graph_topk & tabular_topk) / k)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, overlaps_graph, 'b-o', label='Fused ∩ Graph', linewidth=2)
    ax.plot(k_values, overlaps_tabular, 'g-s', label='Fused ∩ Tabular', linewidth=2)
    ax.plot(k_values, overlaps_both, 'r-^', label='All Three', linewidth=2)
    
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Overlap Rate', fontsize=12)
    ax.set_title('Top-K Overlap Analysis', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"重叠率图已保存: {save_path}")
    
    plt.show()


def plot_fusion_weights(
    weights: np.ndarray,
    node_degrees: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    绘制融合权重分布
    
    Args:
        weights: 融合权重 (α)
        node_degrees: 节点度数
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 权重分布
    axes[0].hist(weights, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0].set_title('Fusion Weight (α) Distribution')
    axes[0].set_xlabel('α (Graph Weight)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.3f}')
    axes[0].legend()
    
    # 权重 vs 度数（如果有）
    if node_degrees is not None:
        axes[1].scatter(node_degrees, weights, alpha=0.3, s=5)
        axes[1].set_xlabel('Node Degree')
        axes[1].set_ylabel('Fusion Weight (α)')
        axes[1].set_title('Weight vs Degree (Gated Fusion)')
    else:
        axes[1].text(0.5, 0.5, 'Node degrees not available', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Weight vs Degree')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"权重图已保存: {save_path}")
    
    plt.show()


def plot_weak_rule_evaluation(
    rule_results: List[Dict],
    save_path: Optional[str] = None
):
    """
    绘制弱规则评估结果
    
    Args:
        rule_results: 弱规则评估结果列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    if not rule_results:
        logging.warning("没有弱规则结果可绘制")
        return
    
    names = [r.rule_name if hasattr(r, 'rule_name') else r['rule_name'] for r in rule_results]
    lifts = [r.lift if hasattr(r, 'lift') else r['lift'] for r in rule_results]
    hit_rates = [r.topk_hit_rate if hasattr(r, 'topk_hit_rate') else r['topk_hit_rate'] for r in rule_results]
    base_rates = [r.population_rate if hasattr(r, 'population_rate') else r['population_rate'] for r in rule_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 提升度
    colors = ['green' if l > 1 else 'red' for l in lifts]
    bars = axes[0].barh(names, lifts, color=colors, alpha=0.7)
    axes[0].axvline(x=1, color='black', linestyle='--', label='Baseline (1x)')
    axes[0].set_xlabel('Lift')
    axes[0].set_title('Weak Rule Lift')
    axes[0].legend()
    
    for bar, lift in zip(bars, lifts):
        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{lift:.2f}x', va='center')
    
    # 命中率对比
    x = np.arange(len(names))
    width = 0.35
    
    axes[1].bar(x - width/2, hit_rates, width, label='Top-K Hit Rate', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, base_rates, width, label='Population Rate', color='gray', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('Hit Rate vs Population Rate')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"弱规则评估图已保存: {save_path}")
    
    plt.show()


def plot_stability_analysis(
    jaccard_scores: Dict[int, float],
    save_path: Optional[str] = None
):
    """
    绘制稳定性分析结果
    
    Args:
        jaccard_scores: K -> Jaccard@K 字典
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    k_values = sorted(jaccard_scores.keys())
    scores = [jaccard_scores[k] for k in k_values]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(range(len(k_values)), scores, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'K={k}' for k in k_values])
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('Stability Analysis: Jaccard@K Across Runs')
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for i, score in enumerate(scores):
        ax.text(i, score + 0.02, f'{score:.3f}', ha='center', fontsize=10)
    
    # 添加平均线
    mean_jaccard = np.mean(scores)
    ax.axhline(y=mean_jaccard, color='red', linestyle='--', 
               label=f'Mean: {mean_jaccard:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"稳定性分析图已保存: {save_path}")
    
    plt.show()


def create_summary_dashboard(
    fusion_result,
    evaluation_report,
    save_path: Optional[str] = None
):
    """
    创建综合仪表板
    
    Args:
        fusion_result: 融合结果
        evaluation_report: 评估报告
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 分数分布 (3 子图)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(fusion_result.graph_scores, bins=30, alpha=0.7, label='Graph')
    ax1.hist(fusion_result.tabular_scores, bins=30, alpha=0.7, label='Tabular')
    ax1.set_title('Score Distributions')
    ax1.legend()
    
    # 2. 融合分数分布
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(fusion_result.fused_scores, bins=30, color='purple', alpha=0.7)
    ax2.axvline(np.percentile(fusion_result.fused_scores, 95), 
                color='red', linestyle='--', label='P95')
    ax2.set_title('Fused Score Distribution')
    ax2.legend()
    
    # 3. 散点图
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(fusion_result.graph_scores, fusion_result.tabular_scores, 
                alpha=0.2, s=5, c=fusion_result.fused_scores, cmap='Reds')
    ax3.set_xlabel('Graph Score')
    ax3.set_ylabel('Tabular Score')
    ax3.set_title('Score Scatter')
    
    # 4. 弱规则提升度
    ax4 = fig.add_subplot(2, 3, 4)
    if evaluation_report.weak_rules:
        names = [r.rule_name for r in evaluation_report.weak_rules]
        lifts = [r.lift for r in evaluation_report.weak_rules]
        colors = ['green' if l > 1 else 'red' for l in lifts]
        ax4.barh(names, lifts, color=colors, alpha=0.7)
        ax4.axvline(x=1, color='black', linestyle='--')
        ax4.set_title('Weak Rule Lift')
    else:
        ax4.text(0.5, 0.5, 'No weak rules', ha='center', va='center')
    
    # 5. 融合权重分布
    ax5 = fig.add_subplot(2, 3, 5)
    if fusion_result.fusion_weights is not None:
        ax5.hist(fusion_result.fusion_weights, bins=30, color='orange', alpha=0.7)
        ax5.set_title(f'Fusion Weights (mean={fusion_result.fusion_weights.mean():.3f})')
    else:
        ax5.text(0.5, 0.5, 'No weights', ha='center', va='center')
    
    # 6. 统计信息
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    === Fusion Summary ===
    
    Strategy: {fusion_result.metadata.get('strategy', 'N/A')}
    Samples: {len(fusion_result.fused_scores):,}
    
    --- Score Statistics ---
    Mean: {fusion_result.fused_scores.mean():.4f}
    Std: {fusion_result.fused_scores.std():.4f}
    P95: {np.percentile(fusion_result.fused_scores, 95):.4f}
    P99: {np.percentile(fusion_result.fused_scores, 99):.4f}
    
    --- Distribution ---
    Separation: {evaluation_report.score_distribution.get('separation_score', 'N/A'):.2f}
    Tail Weight: {evaluation_report.score_distribution.get('tail_weight', 'N/A'):.1%}
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Fusion Anomaly Detection Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"仪表板已保存: {save_path}")
    
    plt.show()
