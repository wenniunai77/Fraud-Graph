"""
通用工具和样式设置
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# 设置中文字体和样式
def setup_style():
    """设置绘图样式"""
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 150
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3

# 配色方案
COLORS = {
    'graph': '#3498db',      # 蓝色 - 图模型
    'tabular': '#2ecc71',    # 绿色 - 表格模型
    'fused': '#9b59b6',      # 紫色 - 融合结果
    'anomaly': '#e74c3c',    # 红色 - 异常
    'normal': '#95a5a6',     # 灰色 - 正常
    'highlight': '#f39c12',  # 橙色 - 高亮
    'background': '#ecf0f1'  # 浅灰 - 背景
}

def save_figure(fig, save_path: Optional[str], tight: bool = True):
    """保存图片"""
    if save_path:
        if tight:
            fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(save_path, facecolor='white')
        print(f"图片已保存: {save_path}")

def add_value_labels(ax, bars, fmt='{:.2f}', fontsize=8):
    """为柱状图添加数值标签"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=fontsize)

def get_topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """获取 Top-K 掩码"""
    topk_idx = np.argsort(-scores)[:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[topk_idx] = True
    return mask, topk_idx
