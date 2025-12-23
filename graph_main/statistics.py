"""
描述性统计模块
用于数据探索和图结构统计分析
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import degree
except ImportError:
    pass

try:
    import dgl
except ImportError:
    pass

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class GraphStatistics:
    """
    图结构统计分析类
    """
    
    def __init__(self, data, loader=None):
        """
        Args:
            data: PyG Data对象或DGL Graph对象
            loader: DataLoader实例（可选，用于访问原始数据）
        """
        self.data = data
        self.loader = loader
        self.is_pyg = hasattr(data, 'edge_index')
        self.is_dgl = hasattr(data, 'num_nodes') and hasattr(data, 'num_edges')
        
    def get_basic_stats(self) -> Dict:
        """获取基本统计信息"""
        stats = {}
        
        if self.is_pyg:
            stats['num_nodes'] = self.data.num_nodes
            stats['num_edges_with_loops'] = self.data.edge_index.shape[1]
            stats['num_original_edges'] = self.data.original_edge_index.shape[1] if hasattr(self.data, 'original_edge_index') else stats['num_edges_with_loops']
            stats['num_node_features'] = self.data.x.shape[1]
        elif self.is_dgl:
            stats['num_nodes'] = self.data.num_nodes()
            stats['num_edges'] = self.data.num_edges()
            stats['num_node_features'] = self.data.ndata['feat'].shape[1] if 'feat' in self.data.ndata else 0
        
        return stats
    
    def compute_degree_stats(self) -> Dict:
        """计算节点度数统计"""
        stats = {}
        
        if self.is_pyg:
            edge_index = self.data.original_edge_index if hasattr(self.data, 'original_edge_index') else self.data.edge_index
            in_degree = degree(edge_index[1], num_nodes=self.data.num_nodes).cpu().numpy()
            out_degree = degree(edge_index[0], num_nodes=self.data.num_nodes).cpu().numpy()
        elif self.is_dgl:
            in_degree = self.data.in_degrees().cpu().numpy()
            out_degree = self.data.out_degrees().cpu().numpy()
        
        total_degree = in_degree + out_degree
        
        stats['in_degree'] = {
            'mean': float(np.mean(in_degree)),
            'std': float(np.std(in_degree)),
            'min': float(np.min(in_degree)),
            'max': float(np.max(in_degree)),
            'median': float(np.median(in_degree))
        }
        
        stats['out_degree'] = {
            'mean': float(np.mean(out_degree)),
            'std': float(np.std(out_degree)),
            'min': float(np.min(out_degree)),
            'max': float(np.max(out_degree)),
            'median': float(np.median(out_degree))
        }
        
        stats['total_degree'] = {
            'mean': float(np.mean(total_degree)),
            'std': float(np.std(total_degree)),
            'min': float(np.min(total_degree)),
            'max': float(np.max(total_degree)),
            'median': float(np.median(total_degree))
        }
        
        return stats, (in_degree, out_degree, total_degree)
    
    def compute_graph_density(self) -> float:
        """计算图密度"""
        if self.is_pyg:
            num_nodes = self.data.num_nodes
            num_edges = self.data.original_edge_index.shape[1] if hasattr(self.data, 'original_edge_index') else self.data.edge_index.shape[1]
        elif self.is_dgl:
            num_nodes = self.data.num_nodes()
            num_edges = self.data.num_edges()
        
        max_possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        return density
    
    def compute_node_feature_stats(self) -> Dict:
        """计算节点特征统计"""
        if self.is_pyg:
            features = self.data.x.cpu().numpy()
        elif self.is_dgl:
            features = self.data.ndata['feat'].cpu().numpy() if 'feat' in self.data.ndata else None
        
        if features is None:
            return {}
        
        stats = {
            'shape': features.shape,
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'has_nan': bool(np.isnan(features).any()),
            'nan_count': int(np.isnan(features).sum()),
            'zero_ratio': float((features == 0).sum() / features.size)
        }
        
        return stats
    
    def analyze_account_types(self) -> Dict:
        """分析账户类型分布"""
        if self.loader is None or self.loader.df is None:
            return {}
        
        df = self.loader.df
        config = self.loader.config
        
        src_col = config.src_col
        dst_col = config.dst_col
        
        src_accounts = set(df.iloc[:, src_col].astype(str).unique())
        dst_accounts = set(df.iloc[:, dst_col].astype(str).unique())
        
        both_accounts = src_accounts & dst_accounts
        only_src = src_accounts - dst_accounts
        only_dst = dst_accounts - src_accounts
        
        return {
            'source_only_accounts': len(only_src),
            'dest_only_accounts': len(only_dst),
            'bidirectional_accounts': len(both_accounts),
            'total_unique_accounts': len(src_accounts | dst_accounts)
        }
    
    def analyze_categorical_distribution(self) -> Dict:
        """分析类别特征分布"""
        if self.loader is None or self.loader.df is None:
            return {}
        
        df = self.loader.df
        config = self.loader.config
        
        distributions = {}
        
        for col_idx in config.categorical_cols:
            col_data = df.iloc[:, col_idx].astype(str)
            value_counts = col_data.value_counts()
            
            distributions[f'col_{col_idx}'] = {
                'unique_values': int(col_data.nunique()),
                'top_5': value_counts.head(5).to_dict(),
                'missing_ratio': float(col_data.isna().mean())
            }
        
        return distributions
    
    def analyze_numerical_distribution(self) -> Dict:
        """分析数值特征分布"""
        if self.loader is None or self.loader.df is None:
            return {}
        
        df = self.loader.df
        config = self.loader.config
        
        distributions = {}
        
        for col_idx in config.numerical_cols:
            col_data = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
            
            distributions[f'col_{col_idx}'] = {
                'mean': float(col_data.mean()) if not col_data.isna().all() else None,
                'std': float(col_data.std()) if not col_data.isna().all() else None,
                'min': float(col_data.min()) if not col_data.isna().all() else None,
                'max': float(col_data.max()) if not col_data.isna().all() else None,
                'median': float(col_data.median()) if not col_data.isna().all() else None,
                'quantiles': {
                    '25%': float(col_data.quantile(0.25)) if not col_data.isna().all() else None,
                    '50%': float(col_data.quantile(0.50)) if not col_data.isna().all() else None,
                    '75%': float(col_data.quantile(0.75)) if not col_data.isna().all() else None,
                    '95%': float(col_data.quantile(0.95)) if not col_data.isna().all() else None,
                    '99%': float(col_data.quantile(0.99)) if not col_data.isna().all() else None,
                },
                'missing_ratio': float(col_data.isna().mean())
            }
        
        return distributions
    
    def print_full_report(self):
        """打印完整统计报告"""
        print("=" * 80)
        print("图结构统计报告 (Graph Structure Statistics Report)")
        print("=" * 80)
        
        # 基本统计
        basic_stats = self.get_basic_stats()
        print(f"\n📊 基本统计 (Basic Statistics):")
        for key, value in basic_stats.items():
            print(f"  ✓ {key}: {value:,}" if isinstance(value, int) else f"  ✓ {key}: {value}")
        
        # 图密度
        density = self.compute_graph_density()
        print(f"\n🔗 图密度 (Graph Density):")
        print(f"  ✓ Density: {density:.10f} ({density*100:.8f}%)")
        
        # 度数统计
        degree_stats, degrees = self.compute_degree_stats()
        print(f"\n📈 节点度数统计 (Node Degree Statistics):")
        for degree_type, stats in degree_stats.items():
            print(f"\n  {degree_type}:")
            for stat_name, value in stats.items():
                print(f"    - {stat_name}: {value:.2f}")
        
        # 节点特征统计
        feature_stats = self.compute_node_feature_stats()
        if feature_stats:
            print(f"\n📊 节点特征统计 (Node Feature Statistics):")
            for key, value in feature_stats.items():
                print(f"  ✓ {key}: {value}")
        
        # 账户类型分析
        account_stats = self.analyze_account_types()
        if account_stats:
            print(f"\n👥 账户类型分析 (Account Type Analysis):")
            for key, value in account_stats.items():
                print(f"  ✓ {key}: {value:,}")
        
        # 类别特征分布
        cat_dist = self.analyze_categorical_distribution()
        if cat_dist:
            print(f"\n📋 类别特征分布 (Categorical Feature Distribution):")
            for col_name, stats in cat_dist.items():
                print(f"\n  {col_name}:")
                print(f"    - Unique values: {stats['unique_values']}")
                print(f"    - Missing ratio: {stats['missing_ratio']:.4f}")
                print(f"    - Top 5: {list(stats['top_5'].keys())[:5]}")
        
        # 数值特征分布
        num_dist = self.analyze_numerical_distribution()
        if num_dist:
            print(f"\n📊 数值特征分布 (Numerical Feature Distribution):")
            for col_name, stats in num_dist.items():
                print(f"\n  {col_name}:")
                print(f"    - Mean: {stats['mean']}")
                print(f"    - Std: {stats['std']}")
                print(f"    - Min: {stats['min']}, Max: {stats['max']}")
                print(f"    - Median: {stats['median']}")
                print(f"    - Missing ratio: {stats['missing_ratio']:.4f}")
        
        print("\n" + "=" * 80)
        print("统计报告完成 (Report Complete)")
        print("=" * 80)
        
        return {
            'basic': basic_stats,
            'density': density,
            'degree': degree_stats,
            'features': feature_stats,
            'accounts': account_stats,
            'categorical': cat_dist,
            'numerical': num_dist
        }


def generate_statistics_report(data, loader=None) -> Dict:
    """
    生成统计报告的便捷函数
    
    Args:
        data: 图数据对象
        loader: DataLoader实例（可选）
    
    Returns:
        包含所有统计信息的字典
    """
    stats = GraphStatistics(data, loader)
    return stats.print_full_report()
