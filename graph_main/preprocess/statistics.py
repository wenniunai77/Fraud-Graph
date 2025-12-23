"""
统计分析模块
负责对图数据进行描述性统计
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional
from collections import Counter

from config import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class GraphStatistics:
    """
    图统计分析器
    负责计算和输出图的各种统计信息
    """
    
    def __init__(self, config: PreprocessConfig):
        """
        初始化统计分析器
        
        Args:
            config: 预处理配置对象
        """
        self.config = config
        self.stats: Dict[str, Any] = {}
    
    def compute_statistics(
        self, 
        data: Any,  # PyG Data对象
        df: pd.DataFrame,
        node_feature_names: list,
        edge_feature_names: list
    ) -> Dict[str, Any]:
        """
        计算综合统计信息
        
        Args:
            data: PyG Data对象
            df: 原始DataFrame
            node_feature_names: 节点特征名称列表
            edge_feature_names: 边特征名称列表
            
        Returns:
            统计信息字典
        """
        logging.info("计算图统计信息...")
        
        self.stats = {
            "basic": self._compute_basic_stats(data),
            "degree": self._compute_degree_stats(data),
            "node_features": self._compute_node_feature_stats(data, node_feature_names),
            "edge_features": self._compute_edge_feature_stats(data, edge_feature_names),
            "categorical_distribution": self._compute_categorical_distribution(df),
            "amount_distribution": self._compute_amount_distribution(df)
        }
        
        return self.stats
    
    def _compute_basic_stats(self, data: Any) -> Dict[str, Any]:
        """计算基本统计信息"""
        num_nodes = data.num_nodes
        num_edges_with_loops = data.edge_index.shape[1]
        num_edges_original = data.original_edge_index.shape[1] if hasattr(data, 'original_edge_index') else num_edges_with_loops
        
        # 计算图密度
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges_original / max_edges if max_edges > 0 else 0
        
        return {
            "num_nodes": int(num_nodes),
            "num_edges_original": int(num_edges_original),
            "num_edges_with_loops": int(num_edges_with_loops),
            "num_node_features": int(data.x.shape[1]),
            "graph_density": float(density),
            "avg_edges_per_node": float(num_edges_original / num_nodes) if num_nodes > 0 else 0
        }
    
    def _compute_degree_stats(self, data: Any) -> Dict[str, Any]:
        """计算度数统计"""
        try:
            from torch_geometric.utils import degree
            
            edge_index = data.original_edge_index if hasattr(data, 'original_edge_index') else data.edge_index
            
            # 入度和出度
            in_degree = degree(edge_index[1], num_nodes=data.num_nodes).cpu().numpy()
            out_degree = degree(edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
            total_degree = in_degree + out_degree
            
            def get_stats(arr):
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                    "zero_count": int(np.sum(arr == 0))
                }
            
            return {
                "in_degree": get_stats(in_degree),
                "out_degree": get_stats(out_degree),
                "total_degree": get_stats(total_degree)
            }
            
        except Exception as e:
            logging.error(f"度数统计计算失败: {e}")
            return {"error": str(e)}
    
    def _compute_node_feature_stats(self, data: Any, feature_names: list) -> Dict[str, Any]:
        """计算节点特征统计"""
        x = data.x.cpu().numpy()
        
        stats = {}
        for i, name in enumerate(feature_names):
            col = x[:, i]
            stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "zero_count": int(np.sum(col == 0)),
                "zero_percent": float(np.sum(col == 0) / len(col) * 100),
                "nan_count": int(np.sum(np.isnan(col))),
                "inf_count": int(np.sum(np.isinf(col)))
            }
        
        return stats
    
    def _compute_edge_feature_stats(self, data: Any, feature_names: list) -> Dict[str, Any]:
        """计算边特征统计"""
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            return {}
        
        edge_attr = data.edge_attr.cpu().numpy()
        
        stats = {}
        for i, name in enumerate(feature_names):
            if i < edge_attr.shape[1]:
                col = edge_attr[:, i]
                stats[name] = {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                    "zero_count": int(np.sum(col == 0)),
                    "zero_percent": float(np.sum(col == 0) / len(col) * 100),
                    "nan_count": int(np.sum(np.isnan(col))),
                    "inf_count": int(np.sum(np.isinf(col)))
                }
        
        return stats
    
    def _compute_categorical_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算类别特征分布"""
        col_idx = self.config.col_idx
        
        distributions = {}
        
        # 交易渠道 (第1列)
        distributions["payment_channel"] = self._get_value_counts(df, col_idx.payment_channel)
        
        # 支付状态码 (第4列)
        distributions["evt_tran_stat_cde"] = self._get_value_counts(df, col_idx.evt_tran_stat_cde)
        
        # 币种 (第5列)
        distributions["instructed_currency"] = self._get_value_counts(df, col_idx.instructed_currency)
        
        # 付款方式 (第13列)
        distributions["mop"] = self._get_value_counts(df, col_idx.mop)
        
        return distributions
    
    def _get_value_counts(self, df: pd.DataFrame, col_idx: int, top_k: int = 10) -> Dict:
        """获取列的值计数"""
        try:
            counts = df.iloc[:, col_idx].value_counts()
            total = len(df)
            
            top_values = {}
            for val, count in counts.head(top_k).items():
                top_values[str(val)] = {
                    "count": int(count),
                    "percent": float(count / total * 100)
                }
            
            return {
                "total_unique": int(counts.shape[0]),
                "top_values": top_values
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _compute_amount_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算金额分布"""
        col_idx = self.config.col_idx
        
        def get_amount_stats(col_i: int, name: str) -> Dict:
            try:
                amounts = pd.to_numeric(df.iloc[:, col_i], errors='coerce').dropna()
                if len(amounts) == 0:
                    return {"error": "no valid values"}
                
                return {
                    "name": name,
                    "mean": float(amounts.mean()),
                    "std": float(amounts.std()),
                    "min": float(amounts.min()),
                    "max": float(amounts.max()),
                    "median": float(amounts.median()),
                    "q25": float(amounts.quantile(0.25)),
                    "q75": float(amounts.quantile(0.75)),
                    "q95": float(amounts.quantile(0.95)),
                    "q99": float(amounts.quantile(0.99)),
                    "zero_count": int((amounts == 0).sum()),
                    "negative_count": int((amounts < 0).sum())
                }
            except Exception as e:
                return {"error": str(e)}
        
        return {
            "instructed_amount": get_amount_stats(col_idx.instructed_amount, "客户指定金额"),
            "payment_amount": get_amount_stats(col_idx.payment_amount, "银行使用金额"),
            "credit_amount": get_amount_stats(col_idx.credit_amount, "收款方接收金额")
        }
    
    def print_report(self):
        """打印统计报告"""
        print("\n" + "=" * 80)
        print("图数据统计报告 (Graph Statistics Report)")
        print("=" * 80)
        
        # 基本统计
        if "basic" in self.stats:
            basic = self.stats["basic"]
            print(f"\n📊 基本统计 (Basic Statistics):")
            print(f"  节点数 (Nodes): {basic['num_nodes']:,}")
            print(f"  边数-原始 (Edges-Original): {basic['num_edges_original']:,}")
            print(f"  边数-含自环 (Edges-With Loops): {basic['num_edges_with_loops']:,}")
            print(f"  节点特征维度 (Node Features): {basic['num_node_features']}")
            print(f"  图密度 (Density): {basic['graph_density']:.10f}")
            print(f"  平均每节点边数 (Avg Edges/Node): {basic['avg_edges_per_node']:.2f}")
        
        # 度数统计
        if "degree" in self.stats and "error" not in self.stats["degree"]:
            deg = self.stats["degree"]
            print(f"\n📈 度数统计 (Degree Statistics):")
            print(f"  入度 (In-degree):")
            print(f"    平均: {deg['in_degree']['mean']:.2f}, 最大: {deg['in_degree']['max']:.0f}")
            print(f"  出度 (Out-degree):")
            print(f"    平均: {deg['out_degree']['mean']:.2f}, 最大: {deg['out_degree']['max']:.0f}")
            print(f"  总度数 (Total):")
            print(f"    平均: {deg['total_degree']['mean']:.2f}, 最大: {deg['total_degree']['max']:.0f}")
        
        # 金额统计
        if "amount_distribution" in self.stats:
            print(f"\n💰 金额分布 (Amount Distribution):")
            for key, val in self.stats["amount_distribution"].items():
                if "error" not in val:
                    print(f"  {val.get('name', key)}:")
                    print(f"    平均: {val['mean']:,.2f}, 中位数: {val['median']:,.2f}, 最大: {val['max']:,.2f}")
        
        # 类别分布
        if "categorical_distribution" in self.stats:
            print(f"\n📋 类别分布 (Categorical Distribution):")
            for key, val in self.stats["categorical_distribution"].items():
                if "error" not in val:
                    print(f"  {key}: {val['total_unique']} 个唯一值")
        
        print("\n" + "=" * 80)
    
    def save_statistics(self, output_dir: Optional[str] = None):
        """保存统计信息到JSON文件"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        stats_path = os.path.join(output_dir, self.config.statistics_file)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logging.info(f"统计信息已保存到: {stats_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
