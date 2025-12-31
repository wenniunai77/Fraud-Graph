"""
主题 1: 模型效果可视化
- 训练曲线
- 模型对比
- 分数统计
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from .utils import setup_style, COLORS, save_figure, add_value_labels

setup_style()


def plot_training_curves(
    graph_losses: List[float],
    tabular_info: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制模型训练曲线
    
    Args:
        graph_losses: GraphMAE 训练 loss 列表
        tabular_info: 表格模型训练信息 (可选)
        save_path: 保存路径
    
    Returns:
        matplotlib Figure 对象
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === 左图: GraphMAE 训练曲线 ===
    ax = axes[0]
    epochs = range(1, len(graph_losses) + 1)
    
    ax.plot(epochs, graph_losses, color=COLORS['graph'], linewidth=2, label='Training Loss')
    ax.fill_between(epochs, graph_losses, alpha=0.2, color=COLORS['graph'])
    
    # 标记最佳点
    best_epoch = np.argmin(graph_losses) + 1
    best_loss = min(graph_losses)
    ax.scatter([best_epoch], [best_loss], c=COLORS['anomaly'], s=100, zorder=5, 
              edgecolor='white', linewidth=2, label=f'Best: {best_loss:.4f} @ Epoch {best_epoch}')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (SCE)', fontsize=11)
    ax.set_title('GraphMAE Training Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    # === 右图: 训练统计摘要 ===
    ax = axes[1]
    ax.axis('off')
    
    final_loss = graph_losses[-1]
    improvement = (graph_losses[0] - final_loss) / graph_losses[0] * 100
    converged = abs(final_loss - best_loss) < 0.01
    
    summary = f"""
╔══════════════════════════════════════════════════╗
║           TRAINING SUMMARY                       ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  GraphMAE Model                                  ║
║  ─────────────────────────────────────────────   ║
║  • Total Epochs:      {len(graph_losses):>6}                     ║
║  • Initial Loss:      {graph_losses[0]:>10.4f}                 ║
║  • Final Loss:        {final_loss:>10.4f}                 ║
║  • Best Loss:         {best_loss:>10.4f}  (Epoch {best_epoch:>3})     ║
║  • Improvement:       {improvement:>9.1f}%                 ║
║  • Converged:         {'Yes ✓' if converged else 'No (Early Stop)':>14}         ║
║                                                  ║
"""
    
    if tabular_info:
        summary += f"""║  Tabular Model                                   ║
║  ─────────────────────────────────────────────   ║
║  • Type:              {tabular_info.get('type', 'N/A'):>14}         ║
║  • Training Time:     {tabular_info.get('time', 'N/A'):>14}         ║
║                                                  ║
"""
    
    summary += """╚══════════════════════════════════════════════════╝"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Model Training Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_model_comparison(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    top_k: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制三个模型的对比分析
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        top_k: Top-K 分析
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    n = len(graph_scores)
    
    # 如果数据量太大，对正常点进行采样（但保留所有Top-K异常点）
    max_normal_points = 50000  # 最多显示5万个正常点
    
    # === 左上: 分数散点图 ===
    ax = axes[0, 0]
    
    # 获取 Top-K 索引
    topk_idx = np.argsort(-fused_scores)[:top_k]
    topk_mask = np.zeros(n, dtype=bool)
    topk_mask[topk_idx] = True
    
    # 对正常点采样
    normal_mask = ~topk_mask
    normal_indices = np.where(normal_mask)[0]
    
    if len(normal_indices) > max_normal_points:
        # 随机采样正常点
        np.random.seed(42)
        sampled_normal_idx = np.random.choice(normal_indices, max_normal_points, replace=False)
        # 使用整数数组索引
        graph_scores_normal = graph_scores[sampled_normal_idx]
        tabular_scores_normal = tabular_scores[sampled_normal_idx]
    else:
        # 使用布尔数组索引
        graph_scores_normal = graph_scores[normal_mask]
        tabular_scores_normal = tabular_scores[normal_mask]
    
    # 先画正常点（采样后的）
    if len(graph_scores_normal) > 0:
        ax.scatter(graph_scores_normal, tabular_scores_normal, 
                  c=COLORS['normal'], alpha=0.1, s=8, label='Normal', rasterized=True)
    
    # 再画异常点（全部）
    if topk_mask.sum() > 0:
        ax.scatter(graph_scores[topk_mask], tabular_scores[topk_mask], 
                  c=COLORS['anomaly'], alpha=0.8, s=20, label=f'Top-{top_k}',
                  edgecolors='white', linewidths=0.5, zorder=5)
    
    # 添加对角线
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x')
    
    ax.set_xlabel('Graph Model Score', fontsize=11)
    ax.set_ylabel('Tabular Model Score', fontsize=11)
    ax.set_title(f'Model Score Scatter (Top-{top_k} in Red)', fontsize=12, fontweight='bold')
    
    # === 右上: 相关性分析 ===
    ax = axes[0, 1]
    
    corr_gt = np.corrcoef(graph_scores, tabular_scores)[0, 1]
    corr_gf = np.corrcoef(graph_scores, fused_scores)[0, 1]
    corr_tf = np.corrcoef(tabular_scores, fused_scores)[0, 1]
    
    correlations = [corr_gt, corr_gf, corr_tf]
    labels = ['Graph vs\nTabular', 'Graph vs\nFused', 'Tabular vs\nFused']
    colors_bar = [COLORS['highlight'], COLORS['graph'], COLORS['tabular']]
    
    bars = ax.bar(labels, correlations, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Pearson Correlation', fontsize=11)
    ax.set_title('Model Correlation Analysis', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.2, 1.1)
    
    for bar, val in zip(bars, correlations):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # === 左下: Top-K 重叠率 ===
    ax = axes[1, 0]
    
    k_values = [50, 100, 200, 500, 1000, 2000]
    k_values = [k for k in k_values if k < n]
    
    overlaps_graph = []
    overlaps_tabular = []
    overlaps_all = []
    
    for k in k_values:
        g_topk = set(np.argsort(-graph_scores)[:k])
        t_topk = set(np.argsort(-tabular_scores)[:k])
        f_topk = set(np.argsort(-fused_scores)[:k])
        
        overlaps_graph.append(len(f_topk & g_topk) / k)
        overlaps_tabular.append(len(f_topk & t_topk) / k)
        overlaps_all.append(len(f_topk & g_topk & t_topk) / k)
    
    x = np.arange(len(k_values))
    width = 0.25
    
    ax.bar(x - width, overlaps_graph, width, label='Fused ∩ Graph', color=COLORS['graph'])
    ax.bar(x, overlaps_tabular, width, label='Fused ∩ Tabular', color=COLORS['tabular'])
    ax.bar(x + width, overlaps_all, width, label='All Three', color=COLORS['fused'])
    
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Overlap Rate', fontsize=11)
    ax.set_title('Top-K Overlap Analysis', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}' for k in k_values])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    # === 右下: 模型排名一致性 ===
    ax = axes[1, 1]
    
    # 分析 Top-K 中每个模型独立发现的异常
    graph_topk = set(np.argsort(-graph_scores)[:top_k])
    tabular_topk = set(np.argsort(-tabular_scores)[:top_k])
    fused_topk = set(np.argsort(-fused_scores)[:top_k])
    
    only_graph = len(fused_topk & graph_topk - tabular_topk)
    only_tabular = len(fused_topk & tabular_topk - graph_topk)
    both_found = len(fused_topk & graph_topk & tabular_topk)
    fusion_unique = len(fused_topk - graph_topk - tabular_topk)
    
    sizes = [both_found, only_graph, only_tabular, fusion_unique]
    labels = [f'Both\n({both_found})', f'Graph Only\n({only_graph})', 
              f'Tabular Only\n({only_tabular})', f'Fusion Unique\n({fusion_unique})']
    colors_pie = [COLORS['fused'], COLORS['graph'], COLORS['tabular'], COLORS['anomaly']]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, 
                                       autopct='%1.1f%%', startangle=90,
                                       explode=(0.02, 0.02, 0.02, 0.05))
    ax.set_title(f'Top-{top_k} Detection Sources', fontsize=12, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_score_statistics(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制分数统计信息
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scores_list = [graph_scores, tabular_scores, fused_scores]
    names = ['Graph Model', 'Tabular Model', 'Fused']
    colors = [COLORS['graph'], COLORS['tabular'], COLORS['fused']]
    
    for ax, scores, name, color in zip(axes, scores_list, names, colors):
        # 绘制箱线图和小提琴图的组合效果
        parts = ax.violinplot([scores], positions=[0], showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
        
        # 添加百分位线
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(scores, p)
            ax.axhline(val, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.35, val, f'P{p}: {val:.3f}', fontsize=9, va='center')
        
        ax.set_xlim(-0.5, 0.8)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold', color=color)
        ax.set_xticks([])
        
        # 添加统计信息
        stats_text = f"μ={scores.mean():.3f}\nσ={scores.std():.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Score Distribution Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig
