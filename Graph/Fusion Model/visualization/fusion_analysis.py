"""
主题 2: 融合分析可视化
- 融合效果概览
- 权重分布分析
- 模型一致性分析
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any
from .utils import setup_style, COLORS, save_figure, get_topk_mask

setup_style()


def plot_fusion_overview(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    fusion_weights: Optional[np.ndarray] = None,
    strategy: str = "gated",
    top_k: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    融合效果概览图
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        fusion_weights: 融合权重 (α)
        strategy: 融合策略名称
        top_k: Top-K 分析
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    n = len(graph_scores)
    topk_mask, topk_idx = get_topk_mask(fused_scores, top_k)
    
    # === 左上: 融合前后分数对比 ===
    ax = axes[0, 0]
    
    # 计算融合效果指标
    g_score_topk = graph_scores[topk_idx].mean()
    t_score_topk = tabular_scores[topk_idx].mean()
    f_score_topk = fused_scores[topk_idx].mean()
    
    g_score_all = graph_scores.mean()
    t_score_all = tabular_scores.mean()
    f_score_all = fused_scores.mean()
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [g_score_all, t_score_all, f_score_all], width, 
                  label='All Samples', color='lightgray', edgecolor='black')
    bars2 = ax.bar(x + width/2, [g_score_topk, t_score_topk, f_score_topk], width,
                  label=f'Top-{top_k}', color=[COLORS['graph'], COLORS['tabular'], COLORS['fused']])
    
    ax.set_ylabel('Average Score', fontsize=11)
    ax.set_title('Score Comparison: All vs Top-K', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Graph', 'Tabular', 'Fused'])
    ax.legend()
    
    # 添加数值
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # === 右上: 融合权重分布 ===
    ax = axes[0, 1]
    
    if fusion_weights is not None:
        ax.hist(fusion_weights, bins=50, color=COLORS['highlight'], alpha=0.7, edgecolor='black')
        ax.axvline(fusion_weights.mean(), color=COLORS['anomaly'], linestyle='--', linewidth=2,
                  label=f'Mean: {fusion_weights.mean():.3f}')
        ax.axvline(0.5, color='black', linestyle=':', linewidth=1.5, label='Equal Weight')
        ax.set_xlabel('α (Graph Weight)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Fusion Weight Distribution ({strategy})', fontsize=12, fontweight='bold')
        ax.legend()
        
        # 标注区域
        ax.fill_betweenx([0, ax.get_ylim()[1]], 0, 0.5, alpha=0.1, color=COLORS['tabular'])
        ax.fill_betweenx([0, ax.get_ylim()[1]], 0.5, 1, alpha=0.1, color=COLORS['graph'])
        ax.text(0.25, ax.get_ylim()[1]*0.9, 'Tabular\nDominant', ha='center', fontsize=9, color=COLORS['tabular'])
        ax.text(0.75, ax.get_ylim()[1]*0.9, 'Graph\nDominant', ha='center', fontsize=9, color=COLORS['graph'])
    else:
        ax.text(0.5, 0.5, f'Strategy: {strategy}\n(No dynamic weights)', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Fusion Weight Distribution', fontsize=12, fontweight='bold')
    
    # === 左下: 四象限分析 ===
    ax = axes[1, 0]
    
    g_threshold = np.percentile(graph_scores, 90)
    t_threshold = np.percentile(tabular_scores, 90)
    
    # 分类
    q1 = (graph_scores >= g_threshold) & (tabular_scores >= t_threshold)  # 两者都高
    q2 = (graph_scores >= g_threshold) & (tabular_scores < t_threshold)   # 只有图高
    q3 = (graph_scores < g_threshold) & (tabular_scores >= t_threshold)   # 只有表格高
    q4 = (graph_scores < g_threshold) & (tabular_scores < t_threshold)    # 两者都低
    
    # 绘制散点
    ax.scatter(graph_scores[q4], tabular_scores[q4], c=COLORS['normal'], alpha=0.1, s=5, label='Normal')
    ax.scatter(graph_scores[q2], tabular_scores[q2], c=COLORS['graph'], alpha=0.5, s=15, label=f'Graph High ({q2.sum()})')
    ax.scatter(graph_scores[q3], tabular_scores[q3], c=COLORS['tabular'], alpha=0.5, s=15, label=f'Tabular High ({q3.sum()})')
    ax.scatter(graph_scores[q1], tabular_scores[q1], c=COLORS['anomaly'], alpha=0.7, s=20, label=f'Both High ({q1.sum()})')
    
    ax.axvline(g_threshold, color=COLORS['graph'], linestyle='--', alpha=0.5)
    ax.axhline(t_threshold, color=COLORS['tabular'], linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Graph Score', fontsize=11)
    ax.set_ylabel('Tabular Score', fontsize=11)
    ax.set_title('Four Quadrant Analysis (P90 Threshold)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    # === 右下: 融合增益分析 ===
    ax = axes[1, 1]
    
    # 分析融合相对于单一模型的增益
    topk_q1 = q1[topk_idx].sum()
    topk_q2 = q2[topk_idx].sum()
    topk_q3 = q3[topk_idx].sum()
    topk_q4 = q4[topk_idx].sum()
    
    categories = ['Both Models\nAgree', 'Graph\nOnly', 'Tabular\nOnly', 'Fusion\nDiscovery']
    values = [topk_q1, topk_q2, topk_q3, topk_q4]
    colors = [COLORS['fused'], COLORS['graph'], COLORS['tabular'], COLORS['highlight']]
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Top-{top_k} Anomaly Sources', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val}\n({val/top_k*100:.1f}%)', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    
    plt.suptitle(f'Fusion Analysis Overview (Strategy: {strategy})', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_fusion_weights_analysis(
    fusion_weights: np.ndarray,
    node_degrees: np.ndarray,
    degree_threshold: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    融合权重与节点度数的关系分析
    
    Args:
        fusion_weights: 融合权重
        node_degrees: 节点度数
        degree_threshold: 活跃度阈值
        save_path: 保存路径
    """
    import logging

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- 输入标准化/对齐 ---
    fusion_weights = np.asarray(fusion_weights).reshape(-1)
    if node_degrees is None:
        node_degrees_arr: Optional[np.ndarray] = None
    else:
        node_degrees_arr = np.asarray(node_degrees).reshape(-1)

    if node_degrees_arr is not None and len(node_degrees_arr) != len(fusion_weights):
        min_len = min(len(node_degrees_arr), len(fusion_weights))
        logging.warning(
            "plot_fusion_weights_analysis: node_degrees(%d) 与 fusion_weights(%d) 长度不一致，已对齐到 %d（使用前 min_len）",
            len(node_degrees_arr),
            len(fusion_weights),
            min_len,
        )
        node_degrees_arr = node_degrees_arr[:min_len]
        fusion_weights = fusion_weights[:min_len]
    
    # === 左图: 权重 vs 度数散点图 ===
    ax = axes[0]

    if node_degrees_arr is None or len(node_degrees_arr) == 0:
        # 没有度数信息：降级为权重分布展示（避免训练流水线直接失败）
        ax.hist(fusion_weights, bins=50, color=COLORS['highlight'], alpha=0.7, edgecolor='white')
        ax.axvline(0.5, color='black', linestyle=':', linewidth=1.5)
        ax.set_xlabel('α (Graph Weight)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Fusion Weight Distribution (no node degrees)', fontsize=12, fontweight='bold')
    else:
        ax.scatter(node_degrees_arr, fusion_weights, alpha=0.3, s=10, c=COLORS['highlight'])
        ax.axvline(degree_threshold, color=COLORS['anomaly'], linestyle='--', linewidth=2,
                  label=f'Threshold: {degree_threshold}')
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Node Degree', fontsize=11)
        ax.set_ylabel('α (Graph Weight)', fontsize=11)
        ax.set_title('Fusion Weight vs Node Degree', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xscale('log')
    
    # === 中图: 分组对比 ===
    ax = axes[1]

    if node_degrees_arr is None or len(node_degrees_arr) == 0:
        inactive_weights = fusion_weights
        active_weights = np.array([], dtype=float)
    else:
        inactive_mask = node_degrees_arr < degree_threshold
        active_mask = ~inactive_mask

        inactive_weights = fusion_weights[inactive_mask]
        active_weights = fusion_weights[active_mask]
    
    positions = [1, 2]
    bp = ax.boxplot([inactive_weights, active_weights], positions=positions, patch_artist=True)
    
    colors_box = [COLORS['tabular'], COLORS['graph']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax.set_xticklabels([f'Inactive\n(degree<{degree_threshold})\nn={len(inactive_weights)}', 
                       f'Active\n(degree≥{degree_threshold})\nn={len(active_weights)}'])
    ax.set_ylabel('α (Graph Weight)', fontsize=11)
    ax.set_title('Weight Distribution by Activity', fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5)
    
    # === 右图: 度数区间权重分析 ===
    ax = axes[2]

    if node_degrees_arr is None or len(node_degrees_arr) == 0:
        # 无度数：直接画权重分位数，保持三联图完整
        qs = [50, 75, 90, 95, 99]
        q_vals = [np.percentile(fusion_weights, q) for q in qs]
        bars = ax.bar([f'P{q}' for q in qs], q_vals, color=COLORS['highlight'], edgecolor='black')
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, label='Equal Weight')
        ax.set_xlabel('Percentile', fontsize=11)
        ax.set_ylabel('α', fontsize=11)
        ax.set_title('Weight Percentiles', fontsize=12, fontweight='bold')
        ax.legend()
        for bar, val in zip(bars, q_vals):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    else:
        degree_bins = [1, 2, 5, 10, 20, 50, 100, int(node_degrees_arr.max()) + 1]
        mean_weights = []
        bin_labels = []

        for i in range(len(degree_bins) - 1):
            mask = (node_degrees_arr >= degree_bins[i]) & (node_degrees_arr < degree_bins[i+1])
            if mask.sum() > 0:
                mean_weights.append(fusion_weights[mask].mean())
                bin_labels.append(f'{degree_bins[i]}-{degree_bins[i+1]-1}')

        bars = ax.bar(bin_labels, mean_weights, color=COLORS['highlight'], edgecolor='black')
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, label='Equal Weight')
        ax.set_xlabel('Degree Range', fontsize=11)
        ax.set_ylabel('Mean α', fontsize=11)
        ax.set_title('Mean Weight by Degree Range', fontsize=12, fontweight='bold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Fusion Weight Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_model_agreement(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    top_k: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    模型一致性分析
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        top_k: Top-K 分析
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    n = len(graph_scores)
    
    # === 左图: 排名差异分布 ===
    ax = axes[0]
    
    graph_rank = np.argsort(np.argsort(-graph_scores))
    tabular_rank = np.argsort(np.argsort(-tabular_scores))
    rank_diff = np.abs(graph_rank - tabular_rank)
    
    topk_mask, _ = get_topk_mask(fused_scores, top_k)
    
    ax.hist(rank_diff[~topk_mask], bins=50, alpha=0.5, color=COLORS['normal'], 
           label='Normal', density=True)
    ax.hist(rank_diff[topk_mask], bins=50, alpha=0.7, color=COLORS['anomaly'], 
           label=f'Top-{top_k}', density=True)
    
    ax.set_xlabel('|Graph Rank - Tabular Rank|', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Ranking Disagreement Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    # === 中图: 一致性分类 ===
    ax = axes[1]
    
    # 定义一致性阈值
    score_diff = graph_scores - tabular_scores
    consistent_high = (graph_scores > np.percentile(graph_scores, 80)) & \
                     (tabular_scores > np.percentile(tabular_scores, 80))
    consistent_low = (graph_scores < np.percentile(graph_scores, 20)) & \
                    (tabular_scores < np.percentile(tabular_scores, 20))
    disagree = ~consistent_high & ~consistent_low
    
    categories = ['Agree High', 'Agree Low', 'Disagree']
    total_counts = [consistent_high.sum(), consistent_low.sum(), disagree.sum()]
    topk_counts = [consistent_high[topk_mask].sum(), consistent_low[topk_mask].sum(), disagree[topk_mask].sum()]
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, total_counts, width, label='All', color='lightgray', edgecolor='black')
    ax.bar(x + width/2, topk_counts, width, label=f'Top-{top_k}', 
          color=[COLORS['fused'], COLORS['normal'], COLORS['highlight']])
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Model Agreement Categories', fontsize=12, fontweight='bold')
    ax.legend()
    
    # === 右图: 分数差异分布 ===
    ax = axes[2]
    
    ax.hist(score_diff[~topk_mask], bins=50, alpha=0.5, color=COLORS['normal'], 
           label='Normal', density=True)
    ax.hist(score_diff[topk_mask], bins=50, alpha=0.7, color=COLORS['anomaly'], 
           label=f'Top-{top_k}', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Score Difference (Graph - Tabular)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Score Difference Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    ax.text(0.02, 0.98, '← Tabular Higher', transform=ax.transAxes, fontsize=9, va='top', color=COLORS['tabular'])
    ax.text(0.98, 0.98, 'Graph Higher →', transform=ax.transAxes, fontsize=9, va='top', ha='right', color=COLORS['graph'])
    
    plt.suptitle('Model Agreement Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


# 别名函数，保持向后兼容
plot_fusion_weights_distribution = plot_fusion_weights_analysis