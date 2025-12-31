"""
主题 4: 异常分布可视化
- 分数分布
- 异常散点图
- Top-K 分析
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from .utils import setup_style, COLORS, save_figure, get_topk_mask

setup_style()


def plot_score_distributions(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    top_k: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    分数分布对比图
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        top_k: Top-K 高亮
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    topk_mask, topk_idx = get_topk_mask(fused_scores, top_k)
    
    scores_list = [graph_scores, tabular_scores, fused_scores]
    names = ['Graph Model', 'Tabular Model', 'Fused']
    colors = [COLORS['graph'], COLORS['tabular'], COLORS['fused']]
    
    # === 第一行: 分数直方图 ===
    for ax, scores, name, color in zip(axes[0], scores_list, names, colors):
        # 全部数据
        ax.hist(scores, bins=50, alpha=0.5, color=COLORS['normal'], 
               label='All', density=True)
        # Top-K
        ax.hist(scores[topk_mask], bins=30, alpha=0.8, color=color, 
               label=f'Top-{top_k}', density=True)
        
        # 百分位线
        for p, ls in [(90, ':'), (95, '--'), (99, '-')]:
            val = np.percentile(scores, p)
            ax.axvline(val, color=COLORS['anomaly'], linestyle=ls, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold', color=color)
        ax.legend(loc='upper right')
        
        # 添加统计信息
        stats = f'μ={scores.mean():.3f}\nP95={np.percentile(scores, 95):.3f}'
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
               va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === 第二行: 正常 vs 异常对比 ===
    for ax, scores, name, color in zip(axes[1], scores_list, names, colors):
        normal_scores = scores[~topk_mask]
        anomaly_scores = scores[topk_mask]
        
        # 并排箱线图
        bp = ax.boxplot([normal_scores, anomaly_scores], 
                       labels=['Normal', f'Top-{top_k}'],
                       patch_artist=True)
        
        bp['boxes'][0].set_facecolor(COLORS['normal'])
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(color)
        bp['boxes'][1].set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{name}: Normal vs Anomaly', fontsize=11, fontweight='bold')
        
        # 计算分离度
        separation = (anomaly_scores.mean() - normal_scores.mean()) / \
                    (normal_scores.std() + 1e-8)
        ax.text(0.5, 0.02, f'Separation: {separation:.2f}σ', transform=ax.transAxes,
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Score Distribution Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_anomaly_scatter(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    node_degrees: Optional[np.ndarray] = None,
    top_k: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    异常散点图
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        node_degrees: 节点度数
        top_k: Top-K 分析
        save_path: 保存路径
    """
    import logging

    # --- 输入标准化/对齐：避免 boolean index mismatch ---
    graph_scores = np.asarray(graph_scores).reshape(-1)
    tabular_scores = np.asarray(tabular_scores).reshape(-1)
    fused_scores = np.asarray(fused_scores).reshape(-1)

    min_len = min(len(graph_scores), len(tabular_scores), len(fused_scores))
    if min_len == 0:
        raise ValueError("plot_anomaly_scatter: 输入分数为空")

    if len(graph_scores) != min_len or len(tabular_scores) != min_len or len(fused_scores) != min_len:
        logging.warning(
            "plot_anomaly_scatter: score 长度不一致 g=%d, t=%d, f=%d，已对齐到 min_len=%d",
            len(graph_scores),
            len(tabular_scores),
            len(fused_scores),
            min_len,
        )
        graph_scores = graph_scores[:min_len]
        tabular_scores = tabular_scores[:min_len]
        fused_scores = fused_scores[:min_len]

    if node_degrees is not None:
        node_degrees = np.asarray(node_degrees).reshape(-1)
        if len(node_degrees) != min_len:
            aligned_len = min(len(node_degrees), min_len)
            logging.warning(
                "plot_anomaly_scatter: node_degrees(%d) 与 scores(%d) 长度不一致，已对齐到 %d",
                len(node_degrees),
                min_len,
                aligned_len,
            )
            node_degrees = node_degrees[:aligned_len]
            graph_scores = graph_scores[:aligned_len]
            tabular_scores = tabular_scores[:aligned_len]
            fused_scores = fused_scores[:aligned_len]
            min_len = aligned_len

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    n = min_len
    topk_mask, topk_idx = get_topk_mask(fused_scores, top_k)
    
    # === 左图: Graph vs Tabular 散点 ===
    ax = axes[0]
    
    # 先画正常点
    ax.scatter(graph_scores[~topk_mask], tabular_scores[~topk_mask],
              c=COLORS['normal'], alpha=0.1, s=5, label='Normal')
    # 再画异常点
    scatter = ax.scatter(graph_scores[topk_mask], tabular_scores[topk_mask],
                        c=fused_scores[topk_mask], cmap='Reds', alpha=0.8, s=20,
                        edgecolor='white', linewidth=0.5, label=f'Top-{top_k}')
    
    # 添加对角线
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Graph Model Score', fontsize=11)
    ax.set_ylabel('Tabular Model Score', fontsize=11)
    ax.set_title('Model Score Scatter', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Fused Score')
    
    # === 中图: 分数 vs 节点度数 ===
    ax = axes[1]
    
    if node_degrees is not None:
        ax.scatter(node_degrees[~topk_mask], fused_scores[~topk_mask],
                  c=COLORS['normal'], alpha=0.1, s=5, label='Normal')
        ax.scatter(node_degrees[topk_mask], fused_scores[topk_mask],
                  c=COLORS['anomaly'], alpha=0.8, s=20, edgecolor='white', 
                  linewidth=0.5, label=f'Top-{top_k}')
        
        ax.set_xlabel('Node Degree', fontsize=11)
        ax.set_ylabel('Fused Score', fontsize=11)
        ax.set_title('Score vs Node Degree', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
    else:
        # 如果没有度数信息，显示分数排名
        ranks = np.argsort(np.argsort(-fused_scores))
        ax.scatter(ranks[~topk_mask], fused_scores[~topk_mask],
                  c=COLORS['normal'], alpha=0.1, s=5)
        ax.scatter(ranks[topk_mask], fused_scores[topk_mask],
                  c=COLORS['anomaly'], alpha=0.8, s=20)
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Fused Score', fontsize=11)
        ax.set_title('Score vs Rank', fontsize=12, fontweight='bold')
    
    # === 右图: 2D 密度图 ===
    ax = axes[2]
    
    # 使用2D直方图创建热力图效果
    h = ax.hist2d(graph_scores, tabular_scores, bins=50, cmap='Blues', density=True)
    plt.colorbar(h[3], ax=ax, label='Density')
    
    # 叠加 Top-K 点
    ax.scatter(graph_scores[topk_mask], tabular_scores[topk_mask],
              c=COLORS['anomaly'], alpha=0.6, s=10, edgecolor='white', linewidth=0.3,
              label=f'Top-{top_k}')
    
    ax.set_xlabel('Graph Model Score', fontsize=11)
    ax.set_ylabel('Tabular Model Score', fontsize=11)
    ax.set_title('Score Density with Top-K Overlay', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.suptitle('Anomaly Distribution Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_topk_analysis(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    node_degrees: Optional[np.ndarray] = None,
    k_values: List[int] = [100, 200, 500, 1000],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Top-K 详细分析
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        node_degrees: 节点度数
        k_values: K值列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    n = len(graph_scores)
    k_values = [k for k in k_values if k < n]
    
    # === 左上: 各K值下的分数阈值 ===
    ax = axes[0, 0]
    
    thresholds = {
        'Graph': [np.sort(graph_scores)[::-1][k-1] for k in k_values],
        'Tabular': [np.sort(tabular_scores)[::-1][k-1] for k in k_values],
        'Fused': [np.sort(fused_scores)[::-1][k-1] for k in k_values]
    }
    
    x = np.arange(len(k_values))
    width = 0.25
    
    ax.bar(x - width, thresholds['Graph'], width, label='Graph', color=COLORS['graph'])
    ax.bar(x, thresholds['Tabular'], width, label='Tabular', color=COLORS['tabular'])
    ax.bar(x + width, thresholds['Fused'], width, label='Fused', color=COLORS['fused'])
    
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Score Threshold', fontsize=11)
    ax.set_title('Score Threshold at Each K', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}' for k in k_values])
    ax.legend()
    
    # === 右上: 各K值下的重叠率变化 ===
    ax = axes[0, 1]
    
    overlaps = {'G∩F': [], 'T∩F': [], 'G∩T∩F': []}
    
    for k in k_values:
        g_topk = set(np.argsort(-graph_scores)[:k])
        t_topk = set(np.argsort(-tabular_scores)[:k])
        f_topk = set(np.argsort(-fused_scores)[:k])
        
        overlaps['G∩F'].append(len(g_topk & f_topk) / k)
        overlaps['T∩F'].append(len(t_topk & f_topk) / k)
        overlaps['G∩T∩F'].append(len(g_topk & t_topk & f_topk) / k)
    
    ax.plot(k_values, overlaps['G∩F'], 'o-', color=COLORS['graph'], 
           linewidth=2, markersize=8, label='Graph ∩ Fused')
    ax.plot(k_values, overlaps['T∩F'], 's-', color=COLORS['tabular'], 
           linewidth=2, markersize=8, label='Tabular ∩ Fused')
    ax.plot(k_values, overlaps['G∩T∩F'], '^-', color=COLORS['fused'], 
           linewidth=2, markersize=8, label='All Three')
    
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Overlap Rate', fontsize=11)
    ax.set_title('Overlap Rate vs K', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # === 左下: Top-K 节点度数分布 ===
    ax = axes[1, 0]
    
    if node_degrees is not None:
        for k, color in zip(k_values[:3], [COLORS['graph'], COLORS['tabular'], COLORS['fused']]):
            topk_idx = np.argsort(-fused_scores)[:k]
            topk_degrees = node_degrees[topk_idx]
            ax.hist(topk_degrees, bins=30, alpha=0.5, label=f'Top-{k}', color=color, density=True)
        
        ax.set_xlabel('Node Degree', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Node Degree Distribution in Top-K', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xscale('log')
    else:
        ax.text(0.5, 0.5, 'Node degree not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Node Degree Distribution in Top-K', fontsize=12, fontweight='bold')
    
    # === 右下: Top-K 统计摘要 ===
    ax = axes[1, 1]
    ax.axis('off')
    
    # 生成表格数据
    summary_data = []
    for k in k_values:
        g_topk = set(np.argsort(-graph_scores)[:k])
        t_topk = set(np.argsort(-tabular_scores)[:k])
        f_topk = set(np.argsort(-fused_scores)[:k])
        
        both = len(g_topk & t_topk & f_topk)
        only_g = len(f_topk & g_topk - t_topk)
        only_t = len(f_topk & t_topk - g_topk)
        fusion = len(f_topk - g_topk - t_topk)
        
        summary_data.append([k, both, only_g, only_t, fusion])
    
    summary = """
╔═══════════════════════════════════════════════════════════════╗
║                    TOP-K ANALYSIS SUMMARY                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║     K     │   Both   │  Graph  │ Tabular │  Fusion  │        ║
║           │  Models  │  Only   │  Only   │  Unique  │        ║
║   ────────┼──────────┼─────────┼─────────┼──────────┤        ║
"""
    
    for row in summary_data:
        k, both, only_g, only_t, fusion = row
        summary += f"║   {k:>5}  │   {both:>5}  │  {only_g:>5}  │  {only_t:>5}  │   {fusion:>5}  │        ║\n"
    
    summary += """║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Top-K Detailed Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig
