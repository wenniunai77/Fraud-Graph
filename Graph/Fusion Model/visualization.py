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


# ==================== 新增：模型贡献分析可视化 ====================

def plot_model_contribution_analysis(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    fusion_weights: Optional[np.ndarray] = None,
    top_k: int = 1000,
    save_path: Optional[str] = None
):
    """
    模型贡献分析图 - 分析图模型和表格模型对最终结果的贡献
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        fusion_weights: 融合权重（α值，图模型权重）
        top_k: Top-K 分析
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    n = len(graph_scores)
    topk_idx = np.argsort(-fused_scores)[:top_k]
    topk_mask = np.zeros(n, dtype=bool)
    topk_mask[topk_idx] = True
    
    # ========== 图1: 模型分数差异分析 ==========
    ax = axes[0, 0]
    score_diff = graph_scores - tabular_scores  # 正值=图模型更高，负值=表格模型更高
    
    ax.hist(score_diff[~topk_mask], bins=50, alpha=0.5, color='gray', label='Normal', density=True)
    ax.hist(score_diff[topk_mask], bins=50, alpha=0.7, color='red', label=f'Top-{top_k}', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Score Difference (Graph - Tabular)')
    ax.set_ylabel('Density')
    ax.set_title('Model Score Difference Distribution')
    ax.legend()
    ax.text(0.02, 0.98, '← Tabular Higher | Graph Higher →', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top')
    ax.grid(True, alpha=0.3)
    
    # ========== 图2: 主导模型分类饼图 ==========
    ax = axes[0, 1]
    
    # 对Top-K进行分类
    graph_dominant = (graph_scores[topk_idx] > tabular_scores[topk_idx]).sum()
    tabular_dominant = (graph_scores[topk_idx] < tabular_scores[topk_idx]).sum()
    equal_contribution = (np.abs(graph_scores[topk_idx] - tabular_scores[topk_idx]) < 0.1).sum()
    
    labels = ['Graph Dominant', 'Tabular Dominant', 'Similar']
    sizes = [graph_dominant, tabular_dominant, equal_contribution]
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    explode = (0.05, 0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title(f'Dominant Model in Top-{top_k} Anomalies')
    
    # ========== 图3: 分数贡献条形图 ==========
    ax = axes[0, 2]
    
    # 计算平均贡献
    if fusion_weights is not None:
        avg_graph_weight = fusion_weights.mean()
        avg_tabular_weight = 1 - avg_graph_weight
        
        # Top-K 中的平均贡献
        topk_graph_contrib = (fusion_weights[topk_idx] * graph_scores[topk_idx]).mean()
        topk_tabular_contrib = ((1 - fusion_weights[topk_idx]) * tabular_scores[topk_idx]).mean()
        
        categories = ['All Samples\n(Avg Weight)', f'Top-{top_k}\n(Avg Contribution)']
        graph_vals = [avg_graph_weight, topk_graph_contrib / (topk_graph_contrib + topk_tabular_contrib + 1e-8)]
        tabular_vals = [avg_tabular_weight, topk_tabular_contrib / (topk_graph_contrib + topk_tabular_contrib + 1e-8)]
    else:
        categories = ['Overall', f'Top-{top_k}']
        graph_vals = [graph_scores.mean(), graph_scores[topk_idx].mean()]
        tabular_vals = [tabular_scores.mean(), tabular_scores[topk_idx].mean()]
        # 归一化
        total_overall = graph_vals[0] + tabular_vals[0]
        total_topk = graph_vals[1] + tabular_vals[1]
        graph_vals = [graph_vals[0]/total_overall, graph_vals[1]/total_topk]
        tabular_vals = [tabular_vals[0]/total_overall, tabular_vals[1]/total_topk]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, graph_vals, width, label='Graph Model', color='#3498db')
    bars2 = ax.bar(x + width/2, tabular_vals, width, label='Tabular Model', color='#2ecc71')
    
    ax.set_ylabel('Contribution Ratio')
    ax.set_title('Model Contribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 添加数值标签
    for bar, val in zip(bars1, graph_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', fontsize=9)
    for bar, val in zip(bars2, tabular_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', fontsize=9)
    
    # ========== 图4: 四象限分析 ==========
    ax = axes[1, 0]
    
    # 使用百分位数作为阈值
    g_threshold = np.percentile(graph_scores, 90)
    t_threshold = np.percentile(tabular_scores, 90)
    
    # 分类
    q1 = (graph_scores >= g_threshold) & (tabular_scores >= t_threshold)  # 两者都高
    q2 = (graph_scores >= g_threshold) & (tabular_scores < t_threshold)   # 只有图高
    q3 = (graph_scores < g_threshold) & (tabular_scores >= t_threshold)   # 只有表格高
    q4 = (graph_scores < g_threshold) & (tabular_scores < t_threshold)    # 两者都低
    
    colors_scatter = np.array(['lightgray'] * n)
    colors_scatter[q1] = 'red'      # 两者都高 - 高风险
    colors_scatter[q2] = 'blue'     # 只有图高 - 关系异常
    colors_scatter[q3] = 'green'    # 只有表格高 - 交易异常
    colors_scatter[q4] = 'lightgray'  # 两者都低 - 正常
    
    # 先画普通点
    normal_mask = ~(q1 | q2 | q3)
    ax.scatter(graph_scores[normal_mask], tabular_scores[normal_mask], 
               c='lightgray', alpha=0.1, s=5, label='Normal')
    ax.scatter(graph_scores[q2], tabular_scores[q2], 
               c='blue', alpha=0.5, s=15, label=f'Graph Only ({q2.sum()})')
    ax.scatter(graph_scores[q3], tabular_scores[q3], 
               c='green', alpha=0.5, s=15, label=f'Tabular Only ({q3.sum()})')
    ax.scatter(graph_scores[q1], tabular_scores[q1], 
               c='red', alpha=0.7, s=20, label=f'Both High ({q1.sum()})')
    
    ax.axvline(g_threshold, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(t_threshold, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Graph Model Score')
    ax.set_ylabel('Tabular Model Score')
    ax.set_title('Four Quadrant Analysis (P90 Threshold)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 添加象限标签
    ax.text(0.95, 0.95, 'Q1: Both High\n(Highest Risk)', transform=ax.transAxes, 
            fontsize=8, ha='right', va='top', color='red')
    ax.text(0.95, 0.05, 'Q2: Graph Only\n(Relation Anomaly)', transform=ax.transAxes, 
            fontsize=8, ha='right', va='bottom', color='blue')
    ax.text(0.05, 0.95, 'Q3: Tabular Only\n(Transaction Anomaly)', transform=ax.transAxes, 
            fontsize=8, ha='left', va='top', color='green')
    
    # ========== 图5: Top-K来源分析 ==========
    ax = axes[1, 1]
    
    # 分析Top-K中各类型占比
    topk_q1 = q1[topk_idx].sum()
    topk_q2 = q2[topk_idx].sum()
    topk_q3 = q3[topk_idx].sum()
    topk_q4 = q4[topk_idx].sum()
    
    categories = ['Both High\n(Confirmed)', 'Graph Only\n(Relation)', 
                  'Tabular Only\n(Transaction)', 'Neither\n(Fusion Effect)']
    values = [topk_q1, topk_q2, topk_q3, topk_q4]
    colors = ['red', 'blue', 'green', 'gray']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(f'Top-{top_k} Anomaly Sources')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值和百分比
    for bar, val in zip(bars, values):
        pct = val / top_k * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val}\n({pct:.1f}%)', ha='center', fontsize=9)
    
    # ========== 图6: 模型互补性分析 ==========
    ax = axes[1, 2]
    
    # 计算排名
    graph_rank = np.argsort(np.argsort(-graph_scores))  # 图模型排名
    tabular_rank = np.argsort(np.argsort(-tabular_scores))  # 表格模型排名
    fused_rank = np.argsort(np.argsort(-fused_scores))  # 融合排名
    
    # 分析Top-K中的排名差异
    topk_graph_rank = graph_rank[topk_idx]
    topk_tabular_rank = tabular_rank[topk_idx]
    
    # 计算各模型独自发现的异常数量（只在该模型Top-K中）
    graph_topk_set = set(np.argsort(-graph_scores)[:top_k])
    tabular_topk_set = set(np.argsort(-tabular_scores)[:top_k])
    fused_topk_set = set(topk_idx)
    
    only_graph = len(fused_topk_set & graph_topk_set - tabular_topk_set)
    only_tabular = len(fused_topk_set & tabular_topk_set - graph_topk_set)
    both_found = len(fused_topk_set & graph_topk_set & tabular_topk_set)
    fusion_unique = len(fused_topk_set - graph_topk_set - tabular_topk_set)
    
    # 绘制维恩图风格的条形图
    categories = ['Found by\nGraph Only', 'Found by\nTabular Only', 
                  'Found by\nBoth', 'Fusion\nUnique']
    values = [only_graph, only_tabular, both_found, fusion_unique]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(f'Model Complementarity in Top-{top_k}')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        pct = val / top_k * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{val}\n({pct:.1f}%)', ha='center', fontsize=9)
    
    plt.suptitle('Model Contribution Analysis Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"模型贡献分析图已保存: {save_path}")
    
    plt.show()


def plot_degree_contribution_analysis(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    node_degrees: np.ndarray,
    fusion_weights: Optional[np.ndarray] = None,
    degree_threshold: int = 5,
    save_path: Optional[str] = None
):
    """
    度数-贡献分析图 - 分析不同活跃度下两个模型的贡献
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        node_degrees: 节点度数
        fusion_weights: 融合权重
        degree_threshold: 活跃度阈值
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n = len(graph_scores)
    
    # 分组
    inactive_mask = node_degrees < degree_threshold
    active_mask = node_degrees >= degree_threshold
    
    # ========== 图1: 度数分布与模型权重 ==========
    ax = axes[0, 0]
    
    ax2 = ax.twinx()
    
    # 度数直方图
    bins = np.logspace(0, np.log10(node_degrees.max() + 1), 30)
    ax.hist(node_degrees, bins=bins, alpha=0.5, color='gray', label='Degree Distribution')
    ax.set_xscale('log')
    ax.set_xlabel('Node Degree (log scale)')
    ax.set_ylabel('Count', color='gray')
    ax.tick_params(axis='y', labelcolor='gray')
    
    # 如果有权重，绘制权重曲线
    if fusion_weights is not None:
        # 按度数分组计算平均权重
        unique_degrees = np.unique(node_degrees)
        avg_weights = []
        for d in unique_degrees:
            mask = node_degrees == d
            avg_weights.append(fusion_weights[mask].mean())
        
        ax2.scatter(unique_degrees, avg_weights, c='red', s=20, alpha=0.7, label='Avg α (Graph Weight)')
        ax2.set_ylabel('Graph Model Weight (α)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.axvline(degree_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={degree_threshold}')
    ax.legend(loc='upper left')
    ax.set_title('Degree Distribution & Fusion Weights')
    
    # ========== 图2: 活跃vs非活跃 模型分数对比 ==========
    ax = axes[0, 1]
    
    categories = ['Inactive\n(degree<{})'.format(degree_threshold), 
                  'Active\n(degree≥{})'.format(degree_threshold)]
    
    # 计算各组平均分
    inactive_graph_mean = graph_scores[inactive_mask].mean()
    inactive_tabular_mean = tabular_scores[inactive_mask].mean()
    active_graph_mean = graph_scores[active_mask].mean()
    active_tabular_mean = tabular_scores[active_mask].mean()
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [inactive_graph_mean, active_graph_mean], 
                   width, label='Graph Model', color='#3498db')
    bars2 = ax.bar(x + width/2, [inactive_tabular_mean, active_tabular_mean], 
                   width, label='Tabular Model', color='#2ecc71')
    
    ax.set_ylabel('Average Anomaly Score')
    ax.set_title('Average Scores by Activity Level')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{bar.get_height():.3f}', ha='center', fontsize=9)
    
    # ========== 图3: 模型分数相关性按度数分组 ==========
    ax = axes[1, 0]
    
    # 计算不同度数区间的相关性
    degree_bins = [1, 3, 5, 10, 20, 50, 100, node_degrees.max() + 1]
    correlations = []
    bin_labels = []
    
    for i in range(len(degree_bins) - 1):
        mask = (node_degrees >= degree_bins[i]) & (node_degrees < degree_bins[i + 1])
        if mask.sum() > 10:  # 至少10个样本
            corr = np.corrcoef(graph_scores[mask], tabular_scores[mask])[0, 1]
            correlations.append(corr)
            bin_labels.append(f'{degree_bins[i]}-{degree_bins[i+1]-1}')
    
    colors = ['#e74c3c' if c < 0.3 else '#f39c12' if c < 0.6 else '#2ecc71' for c in correlations]
    bars = ax.bar(bin_labels, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Degree Range')
    ax.set_ylabel('Correlation (Graph vs Tabular)')
    ax.set_title('Model Score Correlation by Degree')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加颜色图例
    ax.text(0.98, 0.02, 'Low corr: Models complement each other\nHigh corr: Models agree', 
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== 图4: 综合统计表格 ==========
    ax = axes[1, 1]
    ax.axis('off')
    
    # 计算统计信息
    inactive_count = inactive_mask.sum()
    active_count = active_mask.sum()
    
    # Top-K 分析
    top_k = min(1000, n // 10)
    topk_idx = np.argsort(-fused_scores)[:top_k]
    topk_inactive = inactive_mask[topk_idx].sum()
    topk_active = active_mask[topk_idx].sum()
    
    # 主导模型分析
    graph_dominant_inactive = (graph_scores[inactive_mask] > tabular_scores[inactive_mask]).sum()
    graph_dominant_active = (graph_scores[active_mask] > tabular_scores[active_mask]).sum()
    
    summary_text = f"""
    ══════════════════════════════════════════════════════════
                    DEGREE-BASED CONTRIBUTION ANALYSIS
    ══════════════════════════════════════════════════════════
    
    Degree Threshold: {degree_threshold}
    
    ┌─────────────────────────────────────────────────────────┐
    │  POPULATION DISTRIBUTION                                │
    ├─────────────────────────────────────────────────────────┤
    │  Inactive (degree < {degree_threshold}):    {inactive_count:>8,} ({inactive_count/n*100:>5.1f}%)         │
    │  Active   (degree ≥ {degree_threshold}):    {active_count:>8,} ({active_count/n*100:>5.1f}%)         │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │  TOP-{top_k} ANOMALY COMPOSITION                            │
    ├─────────────────────────────────────────────────────────┤
    │  From Inactive:  {topk_inactive:>6} ({topk_inactive/top_k*100:>5.1f}%)                       │
    │  From Active:    {topk_active:>6} ({topk_active/top_k*100:>5.1f}%)                       │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │  DOMINANT MODEL (where score is higher)                 │
    ├─────────────────────────────────────────────────────────┤
    │  Inactive nodes:                                        │
    │    Graph dominant:   {graph_dominant_inactive:>6} ({graph_dominant_inactive/inactive_count*100:>5.1f}%)               │
    │    Tabular dominant: {inactive_count - graph_dominant_inactive:>6} ({(inactive_count - graph_dominant_inactive)/inactive_count*100:>5.1f}%)               │
    │  Active nodes:                                          │
    │    Graph dominant:   {graph_dominant_active:>6} ({graph_dominant_active/active_count*100:>5.1f}%)               │
    │    Tabular dominant: {active_count - graph_dominant_active:>6} ({(active_count - graph_dominant_active)/active_count*100:>5.1f}%)               │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │  AVERAGE SCORES                                         │
    ├─────────────────────────────────────────────────────────┤
    │           │  Graph   │  Tabular  │  Fused   │          │
    │  Inactive │  {graph_scores[inactive_mask].mean():.4f}  │   {tabular_scores[inactive_mask].mean():.4f}  │  {fused_scores[inactive_mask].mean():.4f}  │          │
    │  Active   │  {graph_scores[active_mask].mean():.4f}  │   {tabular_scores[active_mask].mean():.4f}  │  {fused_scores[active_mask].mean():.4f}  │          │
    └─────────────────────────────────────────────────────────┘
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.suptitle('Degree-based Model Contribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"度数贡献分析图已保存: {save_path}")
    
    plt.show()


def plot_anomaly_source_heatmap(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    top_k: int = 1000,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    异常来源热力图 - 可视化不同分数组合下的异常分布
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        top_k: Top-K 分析
        n_bins: 分数分箱数
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    n = len(graph_scores)
    topk_idx = np.argsort(-fused_scores)[:top_k]
    topk_mask = np.zeros(n, dtype=bool)
    topk_mask[topk_idx] = True
    
    # 创建分箱边界
    g_bins = np.linspace(0, 1, n_bins + 1)
    t_bins = np.linspace(0, 1, n_bins + 1)
    
    # ========== 图1: 样本密度热力图 ==========
    ax = axes[0]
    
    density_matrix = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (graph_scores >= g_bins[i]) & (graph_scores < g_bins[i+1]) & \
                   (tabular_scores >= t_bins[j]) & (tabular_scores < t_bins[j+1])
            density_matrix[j, i] = mask.sum()
    
    im = ax.imshow(density_matrix, cmap='Blues', origin='lower', aspect='auto')
    ax.set_xlabel('Graph Model Score (binned)')
    ax.set_ylabel('Tabular Model Score (binned)')
    ax.set_title('Sample Density Heatmap')
    plt.colorbar(im, ax=ax, label='Count')
    
    # 添加刻度标签
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels([f'{g_bins[i]:.1f}' for i in range(n_bins)], fontsize=7)
    ax.set_yticklabels([f'{t_bins[i]:.1f}' for i in range(n_bins)], fontsize=7)
    
    # ========== 图2: Top-K 密度热力图 ==========
    ax = axes[1]
    
    topk_density = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (graph_scores >= g_bins[i]) & (graph_scores < g_bins[i+1]) & \
                   (tabular_scores >= t_bins[j]) & (tabular_scores < t_bins[j+1]) & topk_mask
            topk_density[j, i] = mask.sum()
    
    im = ax.imshow(topk_density, cmap='Reds', origin='lower', aspect='auto')
    ax.set_xlabel('Graph Model Score (binned)')
    ax.set_ylabel('Tabular Model Score (binned)')
    ax.set_title(f'Top-{top_k} Anomaly Density Heatmap')
    plt.colorbar(im, ax=ax, label='Count')
    
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels([f'{g_bins[i]:.1f}' for i in range(n_bins)], fontsize=7)
    ax.set_yticklabels([f'{t_bins[i]:.1f}' for i in range(n_bins)], fontsize=7)
    
    # ========== 图3: 富集度热力图 ==========
    ax = axes[2]
    
    # 计算富集度 = Top-K比例 / 总体比例
    enrichment = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (graph_scores >= g_bins[i]) & (graph_scores < g_bins[i+1]) & \
                   (tabular_scores >= t_bins[j]) & (tabular_scores < t_bins[j+1])
            total_in_bin = mask.sum()
            topk_in_bin = (mask & topk_mask).sum()
            
            if total_in_bin > 0:
                expected_ratio = top_k / n
                actual_ratio = topk_in_bin / total_in_bin
                enrichment[j, i] = actual_ratio / expected_ratio if expected_ratio > 0 else 0
            else:
                enrichment[j, i] = 0
    
    # 使用对数尺度，0值处理
    enrichment_log = np.log10(enrichment + 0.1)
    
    im = ax.imshow(enrichment_log, cmap='RdYlGn', origin='lower', aspect='auto',
                   vmin=-1, vmax=2)
    ax.set_xlabel('Graph Model Score (binned)')
    ax.set_ylabel('Tabular Model Score (binned)')
    ax.set_title(f'Enrichment Ratio Heatmap (log10 scale)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(Enrichment)')
    cbar.set_ticks([-1, 0, 1, 2])
    cbar.set_ticklabels(['0.1x', '1x', '10x', '100x'])
    
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels([f'{g_bins[i]:.1f}' for i in range(n_bins)], fontsize=7)
    ax.set_yticklabels([f'{t_bins[i]:.1f}' for i in range(n_bins)], fontsize=7)
    
    plt.suptitle('Anomaly Source Analysis Heatmaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"异常来源热力图已保存: {save_path}")
    
    plt.show()


def plot_training_comparison(
    graph_train_losses: List[float],
    tabular_train_info: Optional[Dict] = None,
    save_path: Optional[str] = None
):
    """
    训练过程对比图
    
    Args:
        graph_train_losses: 图模型训练loss
        tabular_train_info: 表格模型训练信息（可选）
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== 图1: GraphMAE 训练曲线 ==========
    ax = axes[0]
    
    epochs = range(1, len(graph_train_losses) + 1)
    ax.plot(epochs, graph_train_losses, 'b-', linewidth=2, label='Train Loss')
    ax.fill_between(epochs, graph_train_losses, alpha=0.2)
    
    # 标记最佳点
    best_epoch = np.argmin(graph_train_losses) + 1
    best_loss = min(graph_train_losses)
    ax.scatter([best_epoch], [best_loss], c='red', s=100, zorder=5, label=f'Best: {best_loss:.4f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (SCE)')
    ax.set_title('GraphMAE Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========== 图2: 模型训练统计 ==========
    ax = axes[1]
    ax.axis('off')
    
    # 统计信息
    final_loss = graph_train_losses[-1]
    improvement = (graph_train_losses[0] - final_loss) / graph_train_losses[0] * 100
    
    stats_text = f"""
    ══════════════════════════════════════════════
              TRAINING SUMMARY
    ══════════════════════════════════════════════
    
    GraphMAE Model:
    ──────────────────────────────────────────────
      Total Epochs:     {len(graph_train_losses)}
      Initial Loss:     {graph_train_losses[0]:.4f}
      Final Loss:       {final_loss:.4f}
      Best Loss:        {best_loss:.4f} (Epoch {best_epoch})
      Improvement:      {improvement:.1f}%
      Convergence:      {'Yes' if final_loss - best_loss < 0.01 else 'No (Early Stop)'}
    
    """
    
    if tabular_train_info:
        stats_text += f"""
    Tabular Models:
    ──────────────────────────────────────────────
      Isolation Forest: Fitted
      LOF:              Fitted  
      AutoEncoder:      {tabular_train_info.get('ae_epochs', 'N/A')} epochs
                        Final Loss: {tabular_train_info.get('ae_final_loss', 'N/A')}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Model Training Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"训练对比图已保存: {save_path}")
    
    plt.show()
