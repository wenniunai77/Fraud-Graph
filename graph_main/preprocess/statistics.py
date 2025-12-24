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
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.stats: Dict[str, Any] = {}
    
    def compute_statistics(
        self, 
        data: Any,
        df: pd.DataFrame,
        node_feature_names: list,
        edge_feature_names: list
    ) -> Dict[str, Any]:
        logging.info("Computing graph statistics...")
        
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
        num_nodes = data.num_nodes
        num_edges_with_loops = data.edge_index.shape[1]
        num_edges_original = data.original_edge_index.shape[1] if hasattr(data, 'original_edge_index') else num_edges_with_loops
        
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
        try:
            from torch_geometric.utils import degree
            
            edge_index = data.original_edge_index if hasattr(data, 'original_edge_index') else data.edge_index
            
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
            logging.error(f"Degree statistics computation failed: {e}")
            return {"error": str(e)}
    
    def _compute_node_feature_stats(self, data: Any, feature_names: list) -> Dict[str, Any]:
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
        col_idx = self.config.col_idx
        
        distributions = {}
        distributions["payment_channel"] = self._get_value_counts(df, col_idx.payment_channel)
        distributions["instructed_currency"] = self._get_value_counts(df, col_idx.instructed_currency)
        distributions["mop"] = self._get_value_counts(df, col_idx.mop)
        
        return distributions
    
    def _get_value_counts(self, df: pd.DataFrame, col_idx: int, top_k: int = 10) -> Dict:
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
            "instructed_amount": get_amount_stats(col_idx.instructed_amount, "instructed_amount"),
            "payment_amount": get_amount_stats(col_idx.payment_amount, "payment_amount"),
            "credit_amount": get_amount_stats(col_idx.credit_amount, "credit_amount")
        }
    
    def print_report(self):
        print("\n" + "=" * 80)
        print("Graph Statistics Report")
        print("=" * 80)
        
        if "basic" in self.stats:
            basic = self.stats["basic"]
            print(f"\nBasic Statistics:")
            print(f"  Nodes: {basic['num_nodes']:,}")
            print(f"  Edges (Original): {basic['num_edges_original']:,}")
            print(f"  Edges (With Loops): {basic['num_edges_with_loops']:,}")
            print(f"  Node Features: {basic['num_node_features']}")
            print(f"  Density: {basic['graph_density']:.10f}")
            print(f"  Avg Edges/Node: {basic['avg_edges_per_node']:.2f}")
        
        if "degree" in self.stats and "error" not in self.stats["degree"]:
            deg = self.stats["degree"]
            print(f"\nDegree Statistics:")
            print(f"  In-degree:")
            print(f"    Mean: {deg['in_degree']['mean']:.2f}, Max: {deg['in_degree']['max']:.0f}")
            print(f"  Out-degree:")
            print(f"    Mean: {deg['out_degree']['mean']:.2f}, Max: {deg['out_degree']['max']:.0f}")
            print(f"  Total:")
            print(f"    Mean: {deg['total_degree']['mean']:.2f}, Max: {deg['total_degree']['max']:.0f}")
        
        if "amount_distribution" in self.stats:
            print(f"\nAmount Distribution:")
            for key, val in self.stats["amount_distribution"].items():
                if "error" not in val:
                    print(f"  {val.get('name', key)}:")
                    print(f"    Mean: {val['mean']:,.2f}, Median: {val['median']:,.2f}, Max: {val['max']:,.2f}")
        
        if "categorical_distribution" in self.stats:
            print(f"\nCategorical Distribution:")
            for key, val in self.stats["categorical_distribution"].items():
                if "error" not in val:
                    print(f"  {key}: {val['total_unique']} unique values")
        
        print("\n" + "=" * 80)
    
    def save_statistics(self, output_dir: Optional[str] = None):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        stats_path = os.path.join(output_dir, self.config.statistics_file)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Statistics saved to: {stats_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.stats
