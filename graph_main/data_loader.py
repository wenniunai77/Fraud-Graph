"""
数据加载与图构建模块
负责从CSV加载数据并转换为图结构

注意：所有字段访问使用列索引而非列名
"""

import logging
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

try:
    import dgl
    USE_DGL = True
except ImportError:
    USE_DGL = False

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, degree
    USE_PYG = True
except ImportError:
    USE_PYG = False

from config import DataConfig, Config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class DataLoader:
    """
    数据加载器
    负责从CSV加载数据，处理特征，构建图结构
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.df = None
        self.node_map = None
        self.num_nodes = 0
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载CSV数据
        使用列索引访问数据，不依赖列名
        """
        path = file_path or self.config.data_path
        
        logging.info(f"Loading data from {path}...")
        
        if self.config.use_full_dataset:
            self.df = pd.read_csv(path, header=0)
            logging.info(f"Full dataset loaded. Shape: {self.df.shape}")
        else:
            self.df = pd.read_csv(path, header=0, nrows=self.config.sample_size)
            logging.info(f"Sampled data loaded ({self.config.sample_size} rows). Shape: {self.df.shape}")
        
        return self.df
    
    def get_column_by_index(self, col_idx: int) -> pd.Series:
        """通过列索引获取列数据"""
        return self.df.iloc[:, col_idx]
    
    def create_node_mapping(self) -> Tuple[Dict, int]:
        """
        创建节点映射
        根据源节点列和目标节点列创建节点ID映射
        """
        # 获取源节点和目标节点列
        src_col = self.config.src_col
        dst_col = self.config.dst_col
        
        # 获取所有唯一账户
        src_accounts = self.get_column_by_index(src_col).astype(str).unique()
        dst_accounts = self.get_column_by_index(dst_col).astype(str).unique()
        
        # 合并所有唯一节点
        all_nodes = np.union1d(src_accounts, dst_accounts)
        self.num_nodes = len(all_nodes)
        
        # 创建节点映射 {账户ID: 节点索引}
        self.node_map = {name: i for i, name in enumerate(all_nodes)}
        
        logging.info(f"Node mapping created. Total nodes: {self.num_nodes}")
        logging.info(f"  - Source accounts: {len(src_accounts)}")
        logging.info(f"  - Destination accounts: {len(dst_accounts)}")
        
        return self.node_map, self.num_nodes
    
    def build_edge_index(self) -> torch.Tensor:
        """
        构建边索引
        返回 [2, num_edges] 的边索引张量
        """
        src_col = self.config.src_col
        dst_col = self.config.dst_col
        
        # 映射节点ID到索引
        src_indices = self.get_column_by_index(src_col).astype(str).map(self.node_map).values
        dst_indices = self.get_column_by_index(dst_col).astype(str).map(self.node_map).values
        
        # 创建边索引
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        logging.info(f"Edge index built. Shape: {edge_index.shape}, Total edges: {edge_index.shape[1]}")
        
        return edge_index
    
    def process_numerical_features(self) -> torch.Tensor:
        """
        处理数值特征
        对数值列进行标准化
        """
        numerical_cols = self.config.numerical_cols
        
        # 提取数值特征
        num_features = []
        for col_idx in numerical_cols:
            col_data = self.get_column_by_index(col_idx)
            # 转换为数值，无效值填充为0
            col_data = pd.to_numeric(col_data, errors='coerce').fillna(0)
            num_features.append(col_data.values.reshape(-1, 1))
        
        if len(num_features) > 0:
            num_features = np.hstack(num_features)
            # 标准化
            num_features = self.scaler.fit_transform(num_features)
            return torch.tensor(num_features, dtype=torch.float)
        
        return torch.empty((len(self.df), 0), dtype=torch.float)
    
    def process_categorical_features(self) -> torch.Tensor:
        """
        处理类别特征
        使用LabelEncoder将类别特征编码为数值
        """
        categorical_cols = self.config.categorical_cols
        
        cat_features = []
        for col_idx in categorical_cols:
            col_data = self.get_column_by_index(col_idx).astype(str).fillna('UNKNOWN')
            
            # 创建LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(col_data)
            
            # 保存encoder以便后续使用
            self.label_encoders[col_idx] = le
            
            cat_features.append(encoded.reshape(-1, 1))
        
        if len(cat_features) > 0:
            cat_features = np.hstack(cat_features)
            return torch.tensor(cat_features, dtype=torch.float)
        
        return torch.empty((len(self.df), 0), dtype=torch.float)
    
    def process_time_features(self) -> torch.Tensor:
        """
        处理时间特征
        提取时间相关的数值特征（小时、星期几等）
        """
        time_cols = self.config.time_cols
        
        time_features = []
        for col_idx in time_cols:
            col_data = self.get_column_by_index(col_idx)
            
            try:
                # 尝试解析时间
                timestamps = pd.to_datetime(col_data, errors='coerce')
                
                # 提取时间特征
                hour = timestamps.dt.hour.fillna(0).values
                day_of_week = timestamps.dt.dayofweek.fillna(0).values
                day_of_month = timestamps.dt.day.fillna(0).values
                
                time_features.extend([
                    hour.reshape(-1, 1),
                    day_of_week.reshape(-1, 1),
                    day_of_month.reshape(-1, 1)
                ])
            except Exception as e:
                logging.warning(f"Failed to parse time column {col_idx}: {e}")
                continue
        
        if len(time_features) > 0:
            time_features = np.hstack(time_features)
            # 标准化时间特征
            scaler = StandardScaler()
            time_features = scaler.fit_transform(time_features)
            return torch.tensor(time_features, dtype=torch.float)
        
        return torch.empty((len(self.df), 0), dtype=torch.float)
    
    def build_edge_features(self) -> torch.Tensor:
        """
        构建边特征
        将所有处理后的特征拼接为边特征
        """
        logging.info("Building edge features...")
        
        # 处理各类特征
        num_features = self.process_numerical_features()
        cat_features = self.process_categorical_features()
        time_features = self.process_time_features()
        
        # 拼接所有特征
        all_features = []
        if num_features.shape[1] > 0:
            all_features.append(num_features)
            logging.info(f"  - Numerical features: {num_features.shape[1]}")
        if cat_features.shape[1] > 0:
            all_features.append(cat_features)
            logging.info(f"  - Categorical features: {cat_features.shape[1]}")
        if time_features.shape[1] > 0:
            all_features.append(time_features)
            logging.info(f"  - Time features: {time_features.shape[1]}")
        
        if len(all_features) > 0:
            edge_features = torch.cat(all_features, dim=1)
            logging.info(f"Edge features built. Shape: {edge_features.shape}")
            return edge_features
        
        logging.warning("No features extracted!")
        return torch.zeros((len(self.df), 1), dtype=torch.float)
    
    def aggregate_node_features(self, edge_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        聚合节点特征
        从边特征聚合到节点特征（双向聚合 + 统计特征）
        """
        logging.info("Aggregating node features...")
        
        num_edge_features = edge_features.shape[1]
        
        try:
            from torch_scatter import scatter_mean, scatter_add
            
            # 初始化节点特征矩阵
            # 特征维度: 源节点特征 + 目标节点特征 + 统计特征(出度、入度)
            x = torch.zeros((self.num_nodes, num_edge_features * 2 + 2), dtype=torch.float)
            
            # 1. 聚合作为源节点(发送方)的交易特征
            x_src = scatter_mean(edge_features, edge_index[0], dim=0, dim_size=self.num_nodes)
            
            # 2. 聚合作为目标节点(接收方)的交易特征
            x_dst = scatter_mean(edge_features, edge_index[1], dim=0, dim_size=self.num_nodes)
            
            # 3. 统计特征：交易次数
            src_counts = scatter_add(torch.ones(edge_index.shape[1]), edge_index[0], dim=0, dim_size=self.num_nodes)
            dst_counts = scatter_add(torch.ones(edge_index.shape[1]), edge_index[1], dim=0, dim_size=self.num_nodes)
            
            # 组合特征
            x[:, :num_edge_features] = x_src
            x[:, num_edge_features:2*num_edge_features] = x_dst
            x[:, 2*num_edge_features] = src_counts.float()
            x[:, 2*num_edge_features+1] = dst_counts.float()
            
            logging.info("Using torch_scatter for efficient aggregation")
            
        except ImportError:
            logging.warning("torch_scatter not available, using slower manual aggregation")
            
            # 手动聚合（较慢）
            x = torch.zeros((self.num_nodes, num_edge_features * 2 + 2), dtype=torch.float)
            
            # 统计每个节点的特征
            src_features = {}
            dst_features = {}
            src_counts = torch.zeros(self.num_nodes)
            dst_counts = torch.zeros(self.num_nodes)
            
            edge_index_np = edge_index.numpy()
            edge_features_np = edge_features.numpy()
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index_np[0, i], edge_index_np[1, i]
                feat = edge_features_np[i]
                
                if src not in src_features:
                    src_features[src] = []
                src_features[src].append(feat)
                src_counts[src] += 1
                
                if dst not in dst_features:
                    dst_features[dst] = []
                dst_features[dst].append(feat)
                dst_counts[dst] += 1
            
            # 聚合
            for node_id in range(self.num_nodes):
                if node_id in src_features:
                    x[node_id, :num_edge_features] = torch.tensor(np.mean(src_features[node_id], axis=0))
                if node_id in dst_features:
                    x[node_id, num_edge_features:2*num_edge_features] = torch.tensor(np.mean(dst_features[node_id], axis=0))
            
            x[:, 2*num_edge_features] = src_counts
            x[:, 2*num_edge_features+1] = dst_counts
        
        # 处理NaN值
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        logging.info(f"Node features aggregated. Shape: {x.shape}")
        
        return x
    
    def build_pyg_data(self, add_self_loop: bool = True) -> 'Data':
        """
        构建PyTorch Geometric数据对象
        """
        if not USE_PYG:
            raise ImportError("PyTorch Geometric is not installed!")
        
        # 创建节点映射
        self.create_node_mapping()
        
        # 构建边索引
        edge_index = self.build_edge_index()
        
        # 构建边特征并聚合为节点特征
        edge_features = self.build_edge_features()
        x = self.aggregate_node_features(edge_features, edge_index)
        
        # 保存原始边索引
        original_edge_index = edge_index.clone()
        
        # 添加自环
        if add_self_loop:
            edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            logging.info(f"Self-loops added. New edge count: {edge_index_with_loops.shape[1]}")
        else:
            edge_index_with_loops = edge_index
        
        # 计算边权重（归一化）
        row, col = edge_index_with_loops
        deg = degree(row, self.num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 构建Data对象
        data = Data(
            x=x,
            edge_index=edge_index_with_loops,
            edge_weight=edge_weight,
            original_edge_index=original_edge_index,
            edge_features=edge_features,
            num_nodes=self.num_nodes
        )
        
        logging.info("PyG Data object created successfully!")
        logging.info(f"  - Nodes: {data.num_nodes}")
        logging.info(f"  - Edges (with self-loops): {data.edge_index.shape[1]}")
        logging.info(f"  - Node features: {data.x.shape[1]}")
        
        return data
    
    def build_dgl_graph(self, add_self_loop: bool = True) -> 'dgl.DGLGraph':
        """
        构建DGL图对象
        """
        if not USE_DGL:
            raise ImportError("DGL is not installed!")
        
        # 创建节点映射
        self.create_node_mapping()
        
        # 构建边索引
        edge_index = self.build_edge_index()
        
        # 构建边特征并聚合为节点特征
        edge_features = self.build_edge_features()
        x = self.aggregate_node_features(edge_features, edge_index)
        
        # 创建DGL图
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=self.num_nodes)
        
        # 添加自环
        if add_self_loop:
            g = dgl.add_self_loop(g)
            logging.info(f"Self-loops added. New edge count: {g.num_edges()}")
        
        # 设置节点特征
        g.ndata['feat'] = x
        
        logging.info("DGL Graph object created successfully!")
        logging.info(f"  - Nodes: {g.num_nodes()}")
        logging.info(f"  - Edges: {g.num_edges()}")
        logging.info(f"  - Node features: {g.ndata['feat'].shape[1]}")
        
        return g


def load_fraud_graph_data(config: Config) -> Tuple:
    """
    加载欺诈检测图数据的便捷函数
    
    Args:
        config: 配置对象
    
    Returns:
        data: PyG Data对象或DGL Graph对象
        loader: DataLoader实例（包含原始数据和映射信息）
    """
    loader = DataLoader(config.data)
    loader.load_csv()
    
    if USE_DGL:
        graph = loader.build_dgl_graph()
        num_features = graph.ndata['feat'].shape[1]
        return graph, (num_features, loader)
    elif USE_PYG:
        data = loader.build_pyg_data()
        num_features = data.x.shape[1]
        return data, (num_features, loader)
    else:
        raise ImportError("Neither DGL nor PyTorch Geometric is installed!")


if __name__ == "__main__":
    # 测试代码
    from config import get_default_config
    
    config = get_default_config()
    config.data.data_path = "path/to/your/data.csv"
    
    loader = DataLoader(config.data)
    # loader.load_csv()
    # data = loader.build_pyg_data()
    # print(data)
