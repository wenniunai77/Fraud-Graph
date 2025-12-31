"""
综合报告仪表板
一键生成所有可视化
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from .model_performance import plot_training_curves, plot_model_comparison, plot_score_statistics
from .fusion_analysis import plot_fusion_overview, plot_fusion_weights_analysis, plot_model_agreement
from .feature_contribution import plot_feature_importance, plot_model_contribution
from .anomaly_distribution import plot_score_distributions, plot_anomaly_scatter, plot_topk_analysis
from .utils import setup_style, COLORS, save_figure

setup_style()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def create_comprehensive_report(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    output_dir: str,
    graph_train_losses: Optional[List[float]] = None,
    fusion_weights: Optional[np.ndarray] = None,
    node_degrees: Optional[np.ndarray] = None,
    tabular_features: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    fusion_strategy: str = "gated",
    top_k: int = 1000,
    show_plots: bool = False
):
    """
    生成综合可视化报告
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        output_dir: 输出目录
        graph_train_losses: 图模型训练loss
        fusion_weights: 融合权重
        node_degrees: 节点度数
        tabular_features: 表格特征
        feature_names: 特征名称
        fusion_strategy: 融合策略
        top_k: Top-K 分析
        show_plots: 是否显示图片
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not show_plots:
        plt.ioff()
    
    logging.info("=" * 60)
    logging.info("生成可视化报告")
    logging.info("=" * 60)
    
    # ========== 1. 模型效果 ==========
    logging.info("1/4 生成模型效果图...")
    model_dir = os.path.join(output_dir, "1_model_performance")
    os.makedirs(model_dir, exist_ok=True)
    
    # 1.1 训练曲线
    if graph_train_losses:
        fig = plot_training_curves(
            graph_losses=graph_train_losses,
            save_path=os.path.join(model_dir, "training_curves.png")
        )
        plt.close(fig)
    
    # 1.2 模型对比
    fig = plot_model_comparison(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        top_k=top_k,
        save_path=os.path.join(model_dir, "model_comparison.png")
    )
    plt.close(fig)
    
    # 1.3 分数统计
    fig = plot_score_statistics(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        save_path=os.path.join(model_dir, "score_statistics.png")
    )
    plt.close(fig)
    
    # ========== 2. 融合分析 ==========
    logging.info("2/4 生成融合分析图...")
    fusion_dir = os.path.join(output_dir, "2_fusion_analysis")
    os.makedirs(fusion_dir, exist_ok=True)
    
    # 2.1 融合概览
    fig = plot_fusion_overview(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        fusion_weights=fusion_weights,
        strategy=fusion_strategy,
        top_k=top_k,
        save_path=os.path.join(fusion_dir, "fusion_overview.png")
    )
    plt.close(fig)
    
    # 2.2 权重分析
    if fusion_weights is not None and node_degrees is not None:
        fig = plot_fusion_weights_analysis(
            fusion_weights=fusion_weights,
            node_degrees=node_degrees,
            save_path=os.path.join(fusion_dir, "fusion_weights_analysis.png")
        )
        plt.close(fig)
    
    # 2.3 模型一致性
    fig = plot_model_agreement(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        top_k=top_k,
        save_path=os.path.join(fusion_dir, "model_agreement.png")
    )
    plt.close(fig)
    
    # ========== 3. 特征贡献 ==========
    logging.info("3/4 生成特征贡献图...")
    feature_dir = os.path.join(output_dir, "3_feature_contribution")
    os.makedirs(feature_dir, exist_ok=True)
    
    # 3.1 特征重要性
    if tabular_features is not None:
        fig = plot_feature_importance(
            tabular_features=tabular_features,
            tabular_scores=tabular_scores,
            feature_names=feature_names,
            save_path=os.path.join(feature_dir, "feature_importance.png")
        )
        plt.close(fig)
    
    # 3.2 模型贡献
    fig = plot_model_contribution(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        fusion_weights=fusion_weights,
        node_degrees=node_degrees,
        top_k=top_k,
        save_path=os.path.join(feature_dir, "model_contribution.png")
    )
    plt.close(fig)
    
    # ========== 4. 异常分布 ==========
    logging.info("4/4 生成异常分布图...")
    dist_dir = os.path.join(output_dir, "4_anomaly_distribution")
    os.makedirs(dist_dir, exist_ok=True)
    
    # 4.1 分数分布
    fig = plot_score_distributions(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        top_k=top_k,
        save_path=os.path.join(dist_dir, "score_distributions.png")
    )
    plt.close(fig)
    
    # 4.2 异常散点
    fig = plot_anomaly_scatter(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        node_degrees=node_degrees,
        top_k=top_k,
        save_path=os.path.join(dist_dir, "anomaly_scatter.png")
    )
    plt.close(fig)
    
    # 4.3 Top-K 分析
    fig = plot_topk_analysis(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        node_degrees=node_degrees,
        k_values=[100, 200, 500, 1000, 2000],
        save_path=os.path.join(dist_dir, "topk_analysis.png")
    )
    plt.close(fig)
    
    # ========== 5. 生成摘要页 ==========
    _create_summary_page(
        graph_scores=graph_scores,
        tabular_scores=tabular_scores,
        fused_scores=fused_scores,
        fusion_weights=fusion_weights,
        fusion_strategy=fusion_strategy,
        top_k=top_k,
        save_path=os.path.join(output_dir, "summary.png")
    )
    
    if not show_plots:
        plt.ion()
    
    logging.info("=" * 60)
    logging.info(f"可视化报告已生成: {output_dir}")
    logging.info("  ├── 1_model_performance/    模型效果")
    logging.info("  ├── 2_fusion_analysis/      融合分析")
    logging.info("  ├── 3_feature_contribution/ 特征贡献")
    logging.info("  ├── 4_anomaly_distribution/ 异常分布")
    logging.info("  └── summary.png             摘要页")
    logging.info("=" * 60)


def _create_summary_page(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    fusion_weights: Optional[np.ndarray],
    fusion_strategy: str,
    top_k: int,
    save_path: str
):
    """创建摘要页"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    n = len(graph_scores)
    topk_idx = np.argsort(-fused_scores)[:top_k]
    topk_mask = np.zeros(n, dtype=bool)
    topk_mask[topk_idx] = True
    
    # === 1. 分数分布 ===
    ax = axes[0, 0]
    ax.hist(graph_scores, bins=40, alpha=0.5, label='Graph', color=COLORS['graph'], density=True)
    ax.hist(tabular_scores, bins=40, alpha=0.5, label='Tabular', color=COLORS['tabular'], density=True)
    ax.hist(fused_scores, bins=40, alpha=0.5, label='Fused', color=COLORS['fused'], density=True)
    ax.set_title('Score Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    
    # === 2. 散点图 ===
    ax = axes[0, 1]
    ax.scatter(graph_scores[~topk_mask], tabular_scores[~topk_mask], 
              c=COLORS['normal'], alpha=0.1, s=5)
    ax.scatter(graph_scores[topk_mask], tabular_scores[topk_mask], 
              c=COLORS['anomaly'], alpha=0.7, s=15)
    ax.set_xlabel('Graph Score')
    ax.set_ylabel('Tabular Score')
    ax.set_title(f'Score Scatter (Top-{top_k} in Red)', fontsize=12, fontweight='bold')
    
    # === 3. 融合权重 ===
    ax = axes[0, 2]
    if fusion_weights is not None:
        ax.hist(fusion_weights, bins=40, color=COLORS['highlight'], alpha=0.7, edgecolor='black')
        ax.axvline(fusion_weights.mean(), color=COLORS['anomaly'], linestyle='--', linewidth=2)
        ax.set_xlabel('α (Graph Weight)')
        ax.set_ylabel('Frequency')
    else:
        ax.text(0.5, 0.5, f'Strategy: {fusion_strategy}\n(Fixed weights)', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Fusion Weights', fontsize=12, fontweight='bold')
    
    # === 4. Top-K 来源 ===
    ax = axes[1, 0]
    g_topk = set(np.argsort(-graph_scores)[:top_k])
    t_topk = set(np.argsort(-tabular_scores)[:top_k])
    f_topk = set(topk_idx)
    
    both = len(f_topk & g_topk & t_topk)
    only_g = len(f_topk & g_topk - t_topk)
    only_t = len(f_topk & t_topk - g_topk)
    unique = len(f_topk - g_topk - t_topk)
    
    sizes = [both, only_g, only_t, unique]
    labels = ['Both', 'Graph', 'Tabular', 'Fusion']
    colors_pie = [COLORS['fused'], COLORS['graph'], COLORS['tabular'], COLORS['highlight']]
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Top-{top_k} Sources', fontsize=12, fontweight='bold')
    
    # === 5. 模型相关性 ===
    ax = axes[1, 1]
    corr_matrix = np.corrcoef([graph_scores, tabular_scores, fused_scores])
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Graph', 'Tabular', 'Fused'])
    ax.set_yticklabels(['Graph', 'Tabular', 'Fused'])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=12)
    ax.set_title('Model Correlations', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # === 6. 统计摘要 ===
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    ══════════════════════════════════
         FUSION DETECTION SUMMARY
    ══════════════════════════════════
    
    Total Samples:          {n:>10,}
    Top-K Analyzed:         {top_k:>10,}
    Fusion Strategy:        {fusion_strategy:>10}
    
    ─────────────────────────────────
    Score Statistics (Mean ± Std)
    ─────────────────────────────────
    Graph:    {graph_scores.mean():>7.4f} ± {graph_scores.std():.4f}
    Tabular:  {tabular_scores.mean():>7.4f} ± {tabular_scores.std():.4f}
    Fused:    {fused_scores.mean():>7.4f} ± {fused_scores.std():.4f}
    
    ─────────────────────────────────
    Top-K Detection Sources
    ─────────────────────────────────
    Both Models:    {both:>5} ({both/top_k*100:>5.1f}%)
    Graph Only:     {only_g:>5} ({only_g/top_k*100:>5.1f}%)
    Tabular Only:   {only_t:>5} ({only_t/top_k*100:>5.1f}%)
    Fusion Unique:  {unique:>5} ({unique/top_k*100:>5.1f}%)
    ══════════════════════════════════
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
    
    plt.suptitle('Fusion Anomaly Detection - Summary Report', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    plt.close(fig)
