"""
图构建模块
"""
import logging
import numpy as np
import pandas as pd
import torch
import pickle
import json
import os
from typing import Dict, Tuple, Optional, Any

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from configs import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class GraphBuilder:
    """图构建器"""
    
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
        """创建节点映射"""
        logging.info("创建节点映射...")
        
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
        
        logging.info(f"节点映射创建完成:")
        logging.info(f"  - 总节点数: {self.num_nodes:,}")
        logging.info(f"  - 仅作为源账户: {source_only:,}")
        logging.info(f"  - 仅作为目标账户: {dest_only:,}")
        logging.info(f"  - 双向账户: {bidirectional:,}")
        
        return self.node_map, self.num_nodes
    
    def build_edge_index(self, df: pd.DataFrame) -> torch.Tensor:
        """构建边索引"""
        if not self.node_map:
            raise ValueError("请先调用 create_node_mapping()")
        
        logging.info("构建边索引...")
        
        src_col = self.config.src_col
        dst_col = self.config.dst_col
        
        src_indices = df.iloc[:, src_col].astype(str).map(self.node_map).values
        dst_indices = df.iloc[:, dst_col].astype(str).map(self.node_map).values
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        self.meta_info["graph_info"]["num_edges"] = edge_index.shape[1]
        
        logging.info(f"边索引构建完成. Shape: {edge_index.shape}, 边数: {edge_index.shape[1]:,}")
        
        return edge_index
    
    def get_node_degrees(self, edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """获取节点度数（in/out）"""
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        
        out_degree = np.bincount(src_nodes, minlength=self.num_nodes)
        in_degree = np.bincount(dst_nodes, minlength=self.num_nodes)
        
        return in_degree, out_degree
    
    def build_pyg_data(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        add_self_loop: bool = True
    ) -> Data:
        """构建 PyG Data 对象"""
        logging.info("构建 PyG Data 对象...")
        
        original_edge_index = edge_index.clone()
        num_original_edges = edge_index.shape[1]
        
        if add_self_loop:
            edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            num_edges_with_loops = edge_index_with_loops.shape[1]
            logging.info(f"添加自环: {num_original_edges:,} -> {num_edges_with_loops:,} 边")
            
            # P0 修复: 自环时同步扩展 edge_attr
            if edge_features is not None:
                num_self_loops = num_edges_with_loops - num_original_edges
                edge_feature_dim = edge_features.shape[1]
                
                # 为自环创建零特征（语义：自环没有交易信息）
                self_loop_features = torch.zeros(num_self_loops, edge_feature_dim, dtype=edge_features.dtype)
                
                # 拼接：原始边特征 + 自环特征
                edge_features_with_loops = torch.cat([edge_features, self_loop_features], dim=0)
                
                logging.info(f"  边特征对齐: {edge_features.shape} -> {edge_features_with_loops.shape}")
            else:
                edge_features_with_loops = None
        else:
            edge_index_with_loops = edge_index
            num_edges_with_loops = num_original_edges
            edge_features_with_loops = edge_features
        
        # 计算边权重（归一化）
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
        
        if edge_features_with_loops is not None:
            data.edge_attr = edge_features_with_loops
            
            # 安全检查：确保 edge_index 与 edge_attr 维度对齐
            assert data.edge_index.shape[1] == data.edge_attr.shape[0], \
                f"边索引与边特征维度不一致: edge_index={data.edge_index.shape[1]} vs edge_attr={data.edge_attr.shape[0]}"
        
        self.meta_info["graph_info"].update({
            "num_nodes": self.num_nodes,
            "num_edges_original": num_original_edges,
            "num_edges_with_loops": num_edges_with_loops,
            "num_node_features": node_features.shape[1],
            "num_edge_features": edge_features_with_loops.shape[1] if edge_features_with_loops is not None else 0,
            "has_self_loops": add_self_loop,
            "has_edge_weight": True,
            "has_edge_attr": edge_features_with_loops is not None,
            "edge_attr_aligned": add_self_loop and edge_features is not None  # 新增：标记是否做了对齐
        })
        
        logging.info(f"PyG Data 对象构建完成:")
        logging.info(f"  - 节点数: {data.num_nodes:,}")
        logging.info(f"  - 边数（含自环）: {data.edge_index.shape[1]:,}")
        logging.info(f"  - 节点特征维度: {data.x.shape[1]}")
        if edge_features_with_loops is not None:
            logging.info(f"  - 边特征维度: {edge_features_with_loops.shape[1]}")
            logging.info(f"  - 边特征总数: {edge_features_with_loops.shape[0]}")
        
        return data
    
    def save_graph_data(
        self,
        data: Data,
        output_dir: str,
        save_components: bool = True
    ):
        """保存图数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整图数据
        graph_path = os.path.join(output_dir, "graph_data.pt")
        torch.save(data, graph_path)
        logging.info(f"图数据已保存: {graph_path}")
        
        if save_components:
            # 保存节点特征
            torch.save(data.x, os.path.join(output_dir, "node_features.pt"))
            # 保存边索引
            torch.save(data.edge_index, os.path.join(output_dir, "edge_index.pt"))
            # 保存原始边索引
            torch.save(data.original_edge_index, os.path.join(output_dir, "original_edge_index.pt"))
            # 保存边特征
            if data.edge_attr is not None:
                torch.save(data.edge_attr, os.path.join(output_dir, "edge_features.pt"))
        
        # 保存节点映射
        with open(os.path.join(output_dir, "node_mapping.pkl"), 'wb') as f:
            pickle.dump({
                "node_map": self.node_map,
                "reverse_node_map": self.reverse_node_map
            }, f)
        
        # 保存元信息
        with open(os.path.join(output_dir, "graph_meta.json"), 'w') as f:
            json.dump(self.meta_info, f, indent=2)
        
        logging.info(f"所有图数据组件已保存到: {output_dir}")
    
    def load_graph_data(self, data_dir: str) -> Data:
        """加载图数据"""
        graph_path = os.path.join(data_dir, "graph_data.pt")
        data = torch.load(graph_path)
        
        # 加载节点映射
        with open(os.path.join(data_dir, "node_mapping.pkl"), 'rb') as f:
            mapping = pickle.load(f)
            self.node_map = mapping["node_map"]
            self.reverse_node_map = mapping["reverse_node_map"]
            self.num_nodes = len(self.node_map)
        
        logging.info(f"图数据已加载: {graph_path}")
        return data
    
    def get_meta_info(self) -> Dict[str, Any]:
        """获取元信息"""
        return self.meta_info
