"""
主题 3: 特征贡献可视化
- 特征重要性分析
- 模型贡献度分析
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from .utils import setup_style, COLORS, save_figure, get_topk_mask

setup_style()


def plot_feature_importance(
    tabular_features: np.ndarray,
    tabular_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    特征重要性分析
    
    基于与异常分数的相关性估计特征重要性
    
    Args:
        tabular_features: 表格特征矩阵 (n_samples, n_features)
        tabular_scores: 表格模型异常分数
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_features = tabular_features.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # 计算每个特征与异常分数的相关性
    correlations = []
    for i in range(n_features):
        corr = np.abs(np.corrcoef(tabular_features[:, i], tabular_scores)[0, 1])
        correlations.append(corr if not np.isnan(corr) else 0)
    
    correlations = np.array(correlations)
    
    # 排序
    sorted_idx = np.argsort(-correlations)[:top_n]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_corrs = correlations[sorted_idx]
    
    # === 左图: 特征重要性条形图 ===
    ax = axes[0]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_corrs)))
    bars = ax.barh(range(len(sorted_names)), sorted_corrs, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('|Correlation with Anomaly Score|', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, sorted_corrs)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # === 右图: 特征分组分析 ===
    ax = axes[1]
    
    # 按特征类型分组（假设特征名包含类型信息）
    feature_types = {}
    for i, name in enumerate(feature_names):
        if 'amount' in name.lower():
            ftype = 'Amount'
        elif 'time' in name.lower() or 'hour' in name.lower() or 'day' in name.lower():
            ftype = 'Time'
        elif 'encoded' in name.lower() or 'cat' in name.lower():
            ftype = 'Categorical'
        else:
            ftype = 'Other'
        
        if ftype not in feature_types:
            feature_types[ftype] = []
        feature_types[ftype].append(correlations[i])
    
    # 计算每组平均重要性
    type_names = list(feature_types.keys())
    type_means = [np.mean(feature_types[t]) for t in type_names]
    type_counts = [len(feature_types[t]) for t in type_names]
    
    colors_type = [COLORS['anomaly'], COLORS['graph'], COLORS['tabular'], COLORS['normal']][:len(type_names)]
    bars = ax.bar(type_names, type_means, color=colors_type, edgecolor='black')
    
    ax.set_ylabel('Mean |Correlation|', fontsize=11)
    ax.set_title('Feature Group Importance', fontsize=12, fontweight='bold')
    
    for bar, val, count in zip(bars, type_means, type_counts):
        ax.annotate(f'{val:.3f}\n(n={count})', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    
    plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig


def plot_model_contribution(
    graph_scores: np.ndarray,
    tabular_scores: np.ndarray,
    fused_scores: np.ndarray,
    fusion_weights: Optional[np.ndarray] = None,
    node_degrees: Optional[np.ndarray] = None,
    top_k: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    模型贡献度分析
    
    分析图模型和表格模型对检测结果的贡献
    
    Args:
        graph_scores: 图模型分数
        tabular_scores: 表格模型分数
        fused_scores: 融合分数
        fusion_weights: 融合权重
        node_degrees: 节点度数
        top_k: Top-K 分析
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    n = len(graph_scores)
    topk_mask, topk_idx = get_topk_mask(fused_scores, top_k)
    
    # === 左上: 模型贡献比例 ===
    ax = axes[0, 0]
    
    if fusion_weights is not None:
        # 有动态权重时的贡献
        graph_contrib_all = fusion_weights.mean()
        tabular_contrib_all = 1 - graph_contrib_all
        
        graph_contrib_topk = fusion_weights[topk_idx].mean()
        tabular_contrib_topk = 1 - graph_contrib_topk
    else:
        # 估算贡献（基于归一化分数）
        g_norm = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min() + 1e-8)
        t_norm = (tabular_scores - tabular_scores.min()) / (tabular_scores.max() - tabular_scores.min() + 1e-8)
        
        graph_contrib_all = g_norm.mean() / (g_norm.mean() + t_norm.mean())
        tabular_contrib_all = 1 - graph_contrib_all
        
        graph_contrib_topk = g_norm[topk_idx].mean() / (g_norm[topk_idx].mean() + t_norm[topk_idx].mean())
        tabular_contrib_topk = 1 - graph_contrib_topk
    
    categories = ['All Samples', f'Top-{top_k}']
    graph_vals = [graph_contrib_all, graph_contrib_topk]
    tabular_vals = [tabular_contrib_all, tabular_contrib_topk]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, graph_vals, width, label='Graph Model', color=COLORS['graph'])
    bars2 = ax.bar(x + width/2, tabular_vals, width, label='Tabular Model', color=COLORS['tabular'])
    
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5)
    ax.set_ylabel('Contribution Ratio', fontsize=11)
    ax.set_title('Model Contribution Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1)
    
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    # === 右上: 主导模型分布 ===
    ax = axes[0, 1]
    
    # 判断哪个模型占主导
    graph_dominant = graph_scores > tabular_scores
    
    topk_graph_dominant = graph_dominant[topk_idx].sum()
    topk_tabular_dominant = (~graph_dominant[topk_idx]).sum()
    
    labels = [f'Graph Dominant\n({topk_graph_dominant})', f'Tabular Dominant\n({topk_tabular_dominant})']
    sizes = [topk_graph_dominant, topk_tabular_dominant]
    colors_pie = [COLORS['graph'], COLORS['tabular']]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90, 
                                       explode=(0.03, 0.03))
    ax.set_title(f'Dominant Model in Top-{top_k}', fontsize=12, fontweight='bold')
    
    # === 左下: 按度数分组的贡献 ===
    ax = axes[1, 0]
    
    if node_degrees is not None:
        # 分组分析
        degree_groups = [
            ('Low (1-2)', (node_degrees >= 1) & (node_degrees <= 2)),
            ('Medium (3-10)', (node_degrees >= 3) & (node_degrees <= 10)),
            ('High (11-50)', (node_degrees >= 11) & (node_degrees <= 50)),
            ('Very High (>50)', node_degrees > 50)
        ]
        
        group_names = []
        graph_contribs = []
        tabular_contribs = []
        
        for name, mask in degree_groups:
            if mask.sum() > 0:
                group_names.append(name)
                if fusion_weights is not None:
                    g_contrib = fusion_weights[mask].mean()
                else:
                    g_norm = graph_scores[mask]
                    t_norm = tabular_scores[mask]
                    g_contrib = g_norm.mean() / (g_norm.mean() + t_norm.mean() + 1e-8)
                graph_contribs.append(g_contrib)
                tabular_contribs.append(1 - g_contrib)
        
        x = np.arange(len(group_names))
        width = 0.35
        
        ax.bar(x - width/2, graph_contribs, width, label='Graph', color=COLORS['graph'])
        ax.bar(x + width/2, tabular_contribs, width, label='Tabular', color=COLORS['tabular'])
        
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5)
        ax.set_ylabel('Contribution Ratio', fontsize=11)
        ax.set_title('Model Contribution by Node Degree', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Node degree not available', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Model Contribution by Node Degree', fontsize=12, fontweight='bold')
    
    # === 右下: 贡献统计摘要 ===
    ax = axes[1, 1]
    ax.axis('off')
    
    # 计算统计信息
    graph_topk_set = set(np.argsort(-graph_scores)[:top_k])
    tabular_topk_set = set(np.argsort(-tabular_scores)[:top_k])
    fused_topk_set = set(topk_idx)
    
    only_graph = len(fused_topk_set & graph_topk_set - tabular_topk_set)
    only_tabular = len(fused_topk_set & tabular_topk_set - graph_topk_set)
    both_found = len(fused_topk_set & graph_topk_set & tabular_topk_set)
    fusion_unique = len(fused_topk_set - graph_topk_set - tabular_topk_set)
    
    summary = f"""
╔══════════════════════════════════════════════════════╗
║           MODEL CONTRIBUTION SUMMARY                 ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Top-{top_k} Detection Analysis                          ║
║  ──────────────────────────────────────────────────  ║
║  • Found by Both Models:     {both_found:>5} ({both_found/top_k*100:>5.1f}%)         ║
║  • Found by Graph Only:      {only_graph:>5} ({only_graph/top_k*100:>5.1f}%)         ║
║  • Found by Tabular Only:    {only_tabular:>5} ({only_tabular/top_k*100:>5.1f}%)         ║
║  • Fusion Discovery:         {fusion_unique:>5} ({fusion_unique/top_k*100:>5.1f}%)         ║
║                                                      ║
║  Contribution Weights                                ║
║  ──────────────────────────────────────────────────  ║
║  • Graph Model (All):        {graph_contrib_all:>9.1%}              ║
║  • Graph Model (Top-K):      {graph_contrib_topk:>9.1%}              ║
║  • Tabular Model (All):      {tabular_contrib_all:>9.1%}              ║
║  • Tabular Model (Top-K):    {tabular_contrib_topk:>9.1%}              ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Model Contribution Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig
