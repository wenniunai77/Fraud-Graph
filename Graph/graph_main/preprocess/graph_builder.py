import logging
import numpy as np
import pandas as pd
import torch
import pickle
import json
import os
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from config import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class GraphBuilder:
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.node_map: Dict[str, int] = {}
        self.reverse_node_map: Dict[int, str] = {}
        self.num_nodes: int = 0
        self.meta_info: Dict[str, Any] = {
            "graph_info": {},
            "node_mapping_info": {}
        }
    
    def create_node_mapping(self, df: pd.DataFrame) -> Tuple[Dict[str, int], int]:
        logging.info("Creating node mapping...")
        
        src_col = self.config.src_col
        dst_col = self.config.dst_col
        
        src_accounts = df.iloc[:, src_col].astype(str).unique()
        dst_accounts = df.iloc[:, dst_col].astype(str).unique()
        
        all_nodes = np.union1d(src_accounts, dst_accounts)
        self.num_nodes = len(all_nodes)
        
        self.node_map = {name: i for i, name in enumerate(all_nodes)}
        self.reverse_node_map = {i: name for name, i in self.node_map.items()}
        
        src_set = set(src_accounts)
        dst_set = set(dst_accounts)
        source_only = len(src_set - dst_set)
        dest_only = len(dst_set - src_set)
        bidirectional = len(src_set & dst_set)
        
        self.meta_info["node_mapping_info"] = {
            "total_nodes": self.num_nodes,
            "source_accounts": len(src_accounts),
            "dest_accounts": len(dst_accounts),
            "source_only_accounts": source_only,
            "dest_only_accounts": dest_only,
            "bidirectional_accounts": bidirectional
        }
        
        logging.info(f"Node mapping created:")
        logging.info(f"  - Total nodes: {self.num_nodes:,}")
        logging.info(f"  - Source-only accounts: {source_only:,}")
        logging.info(f"  - Dest-only accounts: {dest_only:,}")
        logging.info(f"  - Bidirectional accounts: {bidirectional:,}")
        
        return self.node_map, self.num_nodes
    
    def build_edge_index(self, df: pd.DataFrame) -> torch.Tensor:
        if not self.node_map:
            raise ValueError("Please call create_node_mapping() first")
        
        logging.info("Building edge index...")
        
        src_col = self.config.src_col
        dst_col = self.config.dst_col
        
        src_indices = df.iloc[:, src_col].astype(str).map(self.node_map).values
        dst_indices = df.iloc[:, dst_col].astype(str).map(self.node_map).values
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        self.meta_info["graph_info"]["num_edges"] = edge_index.shape[1]
        
        logging.info(f"Edge index built. Shape: {edge_index.shape}, Edges: {edge_index.shape[1]:,}")
        
        return edge_index
    
    def build_pyg_data(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        add_self_loop: bool = True
    ) -> Data:
        logging.info("Building PyG Data object...")
        
        original_edge_index = edge_index.clone()
        num_original_edges = edge_index.shape[1]
        
        if add_self_loop:
            edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            num_edges_with_loops = edge_index_with_loops.shape[1]
            logging.info(f"Added self loops: {num_original_edges:,} -> {num_edges_with_loops:,} edges")
        else:
            edge_index_with_loops = edge_index
            num_edges_with_loops = num_original_edges
        
        row, col = edge_index_with_loops
        deg = degree(row, self.num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        data = Data(
            x=node_features,
            edge_index=edge_index_with_loops,
            edge_weight=edge_weight,
            original_edge_index=original_edge_index,
            num_nodes=self.num_nodes
        )
        
        if edge_features is not None:
            data.edge_attr = edge_features
        
        self.meta_info["graph_info"].update({
            "num_nodes": self.num_nodes,
            "num_edges_original": num_original_edges,
            "num_edges_with_loops": num_edges_with_loops,
            "num_node_features": node_features.shape[1],
            "num_edge_features": edge_features.shape[1] if edge_features is not None else 0,
            "has_self_loops": add_self_loop,
            "has_edge_weight": True,
            "has_edge_attr": edge_features is not None
        })
        
        logging.info(f"PyG Data object built:")
        logging.info(f"  - Nodes: {data.num_nodes:,}")
        logging.info(f"  - Edges (with self loops): {data.edge_index.shape[1]:,}")
        logging.info(f"  - Node feature dim: {data.x.shape[1]}")
        if edge_features is not None:
            logging.info(f"  - Edge feature dim: {edge_features.shape[1]}")
        
        return data
    
    def save_graph_data(
        self,
        data: Data,
        node_feature_names: list,
        edge_feature_names: list,
        output_dir: Optional[str] = None
    ):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Saving graph data to {output_dir}...")
        
        graph_path = os.path.join(output_dir, self.config.graph_data_file)
        torch.save(data, graph_path)
        logging.info(f"  - Graph data: {graph_path}")
        
        mapping_path = os.path.join(output_dir, self.config.node_mapping_file)
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'node_map': self.node_map,
                'reverse_node_map': self.reverse_node_map
            }, f)
        logging.info(f"  - Node mapping: {mapping_path}")
        
        torch.save(data.x, os.path.join(output_dir, self.config.node_features_file))
        torch.save(data.original_edge_index, os.path.join(output_dir, self.config.edge_index_file))
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            torch.save(data.edge_attr, os.path.join(output_dir, self.config.edge_features_file))
        
        logging.info("Graph data saved")
    
    def get_meta_info(self) -> Dict[str, Any]:
        return self.meta_info
    
    def get_node_id(self, account: str) -> Optional[int]:
        return self.node_map.get(str(account))
    
    def get_account(self, node_id: int) -> Optional[str]:
        return self.reverse_node_map.get(node_id)


def load_graph_data(output_dir: str, config: PreprocessConfig) -> Tuple[Data, Dict]:
    logging.info(f"Loading graph data from {output_dir}...")
    
    graph_path = os.path.join(output_dir, config.graph_data_file)
    data = torch.load(graph_path, weights_only=False)
    
    mapping_path = os.path.join(output_dir, config.node_mapping_file)
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    logging.info(f"Graph data loaded:")
    logging.info(f"  - Nodes: {data.num_nodes:,}")
    logging.info(f"  - Edges: {data.edge_index.shape[1]:,}")
    
    return data, mapping
