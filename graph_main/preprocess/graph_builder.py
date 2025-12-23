"""
图构建模块
负责将处理后的数据转换为PyG图数据结构
"""

import logging
import numpy as np
import pandas as pd
import torch
import pickle
import json
import os
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, degree
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logging.warning("torch_geometric 未安装，部分功能不可用")

from config import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class GraphBuilder:
    """
    图构建器
    负责创建节点映射、边索引和PyG图数据对象
    """
    
    def __init__(self, config: PreprocessConfig):
        """
        初始化图构建器
        
        Args:
            config: 预处理配置对象
        """
        self.config = config
        self.node_map: Dict[str, int] = {}
        self.reverse_node_map: Dict[int, str] = {}
        self.num_nodes: int = 0
        
        # 记录图构建元信息
        self.meta_info: Dict[str, Any] = {
            "graph_info": {},
            "node_mapping_info": {}
        }
    
    def create_node_mapping(self, df: pd.DataFrame) -> Tuple[Dict[str, int], int]:
        """
        创建节点映射
        
        根据源节点列(支付方账户)和目标节点列(收款方账户)创建唯一节点ID映射
        
        Args:
            df: 原始DataFrame
            
        Returns:
            (节点映射字典, 节点总数)
        """
        logging.info("创建节点映射...")
        
        src_col = self.config.src_col  # 第14列: 支付方账户
        dst_col = self.config.dst_col  # 第15列: 收款方账户
        
        # 获取所有唯一账户（转为字符串确保一致性）
        src_accounts = df.iloc[:, src_col].astype(str).unique()
        dst_accounts = df.iloc[:, dst_col].astype(str).unique()
        
        # 合并所有唯一节点
        all_nodes = np.union1d(src_accounts, dst_accounts)
        self.num_nodes = len(all_nodes)
        
        # 创建双向映射
        self.node_map = {name: i for i, name in enumerate(all_nodes)}
        self.reverse_node_map = {i: name for name, i in self.node_map.items()}
        
        # 统计账户类型
        src_set = set(src_accounts)
        dst_set = set(dst_accounts)
        source_only = len(src_set - dst_set)  # 仅作为发送方的账户
        dest_only = len(dst_set - src_set)    # 仅作为接收方的账户
        bidirectional = len(src_set & dst_set)  # 既发送又接收的账户
        
        # 记录映射信息
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
        logging.info(f"  - 仅发送方账户: {source_only:,}")
        logging.info(f"  - 仅接收方账户: {dest_only:,}")
        logging.info(f"  - 双向账户: {bidirectional:,}")
        
        return self.node_map, self.num_nodes
    
    def build_edge_index(self, df: pd.DataFrame) -> torch.Tensor:
        """
        构建边索引
        
        Args:
            df: 原始DataFrame
            
        Returns:
            边索引张量 [2, num_edges]
        """
        if not self.node_map:
            raise ValueError("请先调用 create_node_mapping() 创建节点映射")
        
        logging.info("构建边索引...")
        
        src_col = self.config.src_col  # 第14列: 支付方账户
        dst_col = self.config.dst_col  # 第15列: 收款方账户
        
        # 映射账户到节点索引
        src_indices = df.iloc[:, src_col].astype(str).map(self.node_map).values
        dst_indices = df.iloc[:, dst_col].astype(str).map(self.node_map).values
        
        # 创建边索引张量
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        # 记录边信息
        self.meta_info["graph_info"]["num_edges"] = edge_index.shape[1]
        
        logging.info(f"边索引构建完成。Shape: {edge_index.shape}, 边数: {edge_index.shape[1]:,}")
        
        return edge_index
    
    def build_pyg_data(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        add_self_loop: bool = True
    ) -> 'Data':
        """
        构建PyG Data对象
        
        Args:
            node_features: 节点特征张量 [num_nodes, num_node_features]
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征张量 [num_edges, num_edge_features] (可选)
            add_self_loop: 是否添加自环
            
        Returns:
            PyG Data对象
        """
        if not HAS_PYG:
            raise ImportError("torch_geometric 未安装，无法构建PyG Data对象")
        
        logging.info("构建PyG Data对象...")
        
        # 保存原始边索引（不含自环）
        original_edge_index = edge_index.clone()
        num_original_edges = edge_index.shape[1]
        
        # 添加自环
        if add_self_loop:
            edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            num_edges_with_loops = edge_index_with_loops.shape[1]
            logging.info(f"添加自环: {num_original_edges:,} -> {num_edges_with_loops:,} 边")
        else:
            edge_index_with_loops = edge_index
            num_edges_with_loops = num_original_edges
        
        # 计算边权重（度数归一化）
        row, col = edge_index_with_loops
        deg = degree(row, self.num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 创建Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index_with_loops,
            edge_weight=edge_weight,
            original_edge_index=original_edge_index,
            num_nodes=self.num_nodes
        )
        
        # 添加边特征（如果提供）
        if edge_features is not None:
            data.edge_attr = edge_features
        
        # 更新图信息
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
        
        logging.info(f"PyG Data对象构建完成:")
        logging.info(f"  - 节点数: {data.num_nodes:,}")
        logging.info(f"  - 边数(含自环): {data.edge_index.shape[1]:,}")
        logging.info(f"  - 节点特征维度: {data.x.shape[1]}")
        if edge_features is not None:
            logging.info(f"  - 边特征维度: {edge_features.shape[1]}")
        
        return data
    
    def save_graph_data(
        self,
        data: 'Data',
        node_feature_names: list,
        edge_feature_names: list,
        output_dir: Optional[str] = None
    ):
        """
        保存图数据到文件
        
        Args:
            data: PyG Data对象
            node_feature_names: 节点特征名称列表
            edge_feature_names: 边特征名称列表
            output_dir: 输出目录
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"保存图数据到 {output_dir}...")
        
        # 1. 保存完整图数据
        graph_path = os.path.join(output_dir, self.config.graph_data_file)
        torch.save(data, graph_path)
        logging.info(f"  - 图数据: {graph_path}")
        
        # 2. 保存节点映射
        mapping_path = os.path.join(output_dir, self.config.node_mapping_file)
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'node_map': self.node_map,
                'reverse_node_map': self.reverse_node_map
            }, f)
        logging.info(f"  - 节点映射: {mapping_path}")
        
        # 3. 单独保存各部分（便于调试和检查）
        torch.save(data.x, os.path.join(output_dir, self.config.node_features_file))
        torch.save(data.original_edge_index, os.path.join(output_dir, self.config.edge_index_file))
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            torch.save(data.edge_attr, os.path.join(output_dir, self.config.edge_features_file))
        
        logging.info("图数据保存完成")
    
    def get_meta_info(self) -> Dict[str, Any]:
        """获取图构建元信息"""
        return self.meta_info
    
    def get_node_id(self, account: str) -> Optional[int]:
        """根据账户获取节点ID"""
        return self.node_map.get(str(account))
    
    def get_account(self, node_id: int) -> Optional[str]:
        """根据节点ID获取账户"""
        return self.reverse_node_map.get(node_id)


def load_graph_data(output_dir: str, config: PreprocessConfig) -> Tuple['Data', Dict]:
    """
    加载保存的图数据
    
    Args:
        output_dir: 图数据保存目录
        config: 预处理配置
        
    Returns:
        (PyG Data对象, 节点映射字典)
    """
    logging.info(f"从 {output_dir} 加载图数据...")
    
    # 加载图数据
    graph_path = os.path.join(output_dir, config.graph_data_file)
    data = torch.load(graph_path)
    
    # 加载节点映射
    mapping_path = os.path.join(output_dir, config.node_mapping_file)
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    logging.info(f"图数据加载完成:")
    logging.info(f"  - 节点数: {data.num_nodes:,}")
    logging.info(f"  - 边数: {data.edge_index.shape[1]:,}")
    
    return data, mapping
