"""
特征工程模块
"""
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from configs import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.embeddings: Dict[int, torch.nn.Embedding] = {}
        self.category_mappings: Dict[int, Dict[str, int]] = {}
        self.numerical_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        
        self.meta_info: Dict[str, Any] = {
            "edge_features": {},
            "node_features": {},
            "tabular_features": {},
            "time_diff_info": {},
            "fillna_info": {},
            "encoding_info": {}
        }
    
    def build_edge_features(
        self, 
        df: pd.DataFrame,
        pretrained_embeddings: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        构建边特征（用于图模型）
        
        Args:
            df: 数据框
            pretrained_embeddings: 预训练的 embedding 字典（可选）
        """
        logging.info("构建边特征...")
        
        all_features = []
        feature_names = []
        
        # 1. 数值特征
        logging.info("处理数值特征...")
        num_features, num_names, num_fillna_info = self._process_numerical_features(df)
        if num_features is not None:
            all_features.append(num_features)
            feature_names.extend(num_names)
            self.meta_info["fillna_info"]["numerical"] = num_fillna_info
        
        # 2. 类别特征（嵌入）
        logging.info("处理类别特征...")
        cat_features, cat_names, cat_encoding_info = self._process_categorical_features(
            df, pretrained_embeddings
        )
        if cat_features is not None:
            all_features.append(cat_features)
            feature_names.extend(cat_names)
            self.meta_info["encoding_info"] = cat_encoding_info
        
        # 3. 时间特征
        logging.info("处理时间特征...")
        time_features, time_names, time_fillna_info = self._process_time_features(df)
        if time_features is not None:
            all_features.append(time_features)
            feature_names.extend(time_names)
            self.meta_info["fillna_info"]["time"] = time_fillna_info
        
        # 4. 时间差特征
        logging.info("处理时间差特征...")
        time_diff_features, time_diff_names, time_diff_info = self._process_time_diff_feature(df)
        if time_diff_features is not None:
            all_features.append(time_diff_features)
            feature_names.extend(time_diff_names)
            self.meta_info["time_diff_info"] = time_diff_info
        
        if len(all_features) > 0:
            edge_features = torch.cat(all_features, dim=1)
        else:
            logging.warning("没有提取到边特征！")
            edge_features = torch.zeros((len(df), 1), dtype=torch.float)
            feature_names = ["dummy"]
        
        self.meta_info["edge_features"] = {
            "shape": list(edge_features.shape),
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        logging.info(f"边特征构建完成. Shape: {edge_features.shape}")
        
        return edge_features, feature_names
    
    def build_tabular_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """构建表格特征（用于表格模型，不做嵌入，保留原始值）"""
        logging.info("构建表格特征...")
        
        all_features = []
        feature_names = []
        
        col_idx = self.config.col_idx
        
        # 1. 数值特征（金额）
        for col_i in self.config.numerical_cols:
            col_data = df.iloc[:, col_i]
            col_name = self._get_col_name(col_i, "num")
            
            numeric_data = pd.to_numeric(col_data, errors='coerce').fillna(0).values.reshape(-1, 1)
            all_features.append(numeric_data)
            feature_names.append(col_name)
        
        # 2. 类别特征（Label Encoding）
        for col_i in self.config.categorical_cols:
            col_data = df.iloc[:, col_i].astype(str).fillna('UNKNOWN')
            col_name = self._get_col_name(col_i, "cat")
            
            # 简单Label Encoding
            unique_vals = col_data.unique()
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            encoded = col_data.map(val_to_idx).values.reshape(-1, 1)
            
            all_features.append(encoded)
            feature_names.append(col_name + "_encoded")
        
        # 3. 时间特征
        for col_i in self.config.time_cols:
            col_data = df.iloc[:, col_i]
            prefix = "txn" if col_i == col_idx.txn_dt else "tds"
            
            try:
                timestamps = pd.to_datetime(col_data, errors='coerce')
                hour = timestamps.dt.hour.fillna(0).values.reshape(-1, 1)
                day_of_week = timestamps.dt.dayofweek.fillna(0).values.reshape(-1, 1)
                
                all_features.extend([hour, day_of_week])
                feature_names.extend([f"{prefix}_hour", f"{prefix}_dayofweek"])
            except Exception as e:
                logging.warning(f"时间特征提取失败 (col {col_i}): {e}")
        
        # 4. 时间差特征
        try:
            txn_dt = pd.to_datetime(df.iloc[:, col_idx.txn_dt], errors='coerce')
            tds_dt = pd.to_datetime(df.iloc[:, col_idx.tds_dt], errors='coerce')
            time_diff = (tds_dt - txn_dt).dt.total_seconds().fillna(0).values.reshape(-1, 1)
            all_features.append(time_diff)
            feature_names.append("time_diff_seconds")
        except Exception as e:
            logging.warning(f"时间差特征提取失败: {e}")
        
        if len(all_features) > 0:
            tabular_features = np.hstack(all_features).astype(np.float32)
        else:
            tabular_features = np.zeros((len(df), 1), dtype=np.float32)
            feature_names = ["dummy"]
        
        self.meta_info["tabular_features"] = {
            "shape": list(tabular_features.shape),
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        logging.info(f"表格特征构建完成. Shape: {tabular_features.shape}")
        
        return tabular_features, feature_names
    
    def build_node_features(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """构建节点特征（聚合边特征）"""
        logging.info("构建节点特征...")
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        num_edge_features = edge_features.shape[1]
        
        # 初始化聚合数组
        src_sum = torch.zeros(num_nodes, num_edge_features)
        src_count = torch.zeros(num_nodes, 1)
        dst_sum = torch.zeros(num_nodes, num_edge_features)
        dst_count = torch.zeros(num_nodes, 1)
        
        # 聚合边特征
        src_sum.index_add_(0, src_nodes, edge_features)
        src_count.index_add_(0, src_nodes, torch.ones(len(src_nodes), 1))
        dst_sum.index_add_(0, dst_nodes, edge_features)
        dst_count.index_add_(0, dst_nodes, torch.ones(len(dst_nodes), 1))
        
        # 计算平均
        src_avg = src_sum / (src_count + 1e-8)
        dst_avg = dst_sum / (dst_count + 1e-8)
        
        # 度特征
        src_degree = src_count.squeeze()
        dst_degree = dst_count.squeeze()
        total_degree = src_degree + dst_degree
        in_out_ratio = dst_degree / (src_degree + 1e-8)
        
        # 拼接节点特征
        node_features = torch.cat([
            src_avg,
            dst_avg,
            src_degree.unsqueeze(1),
            dst_degree.unsqueeze(1),
            total_degree.unsqueeze(1),
            in_out_ratio.unsqueeze(1)
        ], dim=1)
        
        # 构建特征名
        node_feature_names = []
        if feature_names:
            for name in feature_names:
                node_feature_names.append(f"src_avg_{name}")
            for name in feature_names:
                node_feature_names.append(f"dst_avg_{name}")
        else:
            for i in range(num_edge_features):
                node_feature_names.append(f"src_avg_feat_{i}")
            for i in range(num_edge_features):
                node_feature_names.append(f"dst_avg_feat_{i}")
        
        node_feature_names.extend([
            "src_tx_count", "dst_tx_count", "total_degree", "in_out_ratio"
        ])
        
        self.meta_info["node_features"] = {
            "shape": list(node_features.shape),
            "feature_names": node_feature_names,
            "num_features": len(node_feature_names)
        }
        
        logging.info(f"节点特征构建完成. Shape: {node_features.shape}")
        
        return node_features, node_feature_names
    
    def _process_numerical_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """处理数值特征"""
        numerical_cols = self.config.numerical_cols
        col_idx = self.config.col_idx
        
        if not numerical_cols:
            return None, [], {}
        
        col_name_map = {
            col_idx.instructed_amount: "instructed_amount",
            col_idx.payment_amount: "payment_amount",
            col_idx.credit_amount: "credit_amount"
        }
        
        features = []
        feature_names = []
        fillna_info = {}
        
        for col_i in numerical_cols:
            col_data = df.iloc[:, col_i]
            col_name = col_name_map.get(col_i, f"num_col_{col_i}")
            
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            
            null_count = numeric_data.isnull().sum()
            if null_count > 0:
                fillna_info[col_name] = {
                    "fillna_count": int(null_count),
                    "fillna_percent": float(null_count / len(df) * 100),
                    "fillna_value": 0
                }
            
            numeric_data = numeric_data.fillna(0)
            features.append(numeric_data.values.reshape(-1, 1))
            feature_names.append(col_name)
        
        features = np.hstack(features)
        features = self.numerical_scaler.fit_transform(features)
        
        return torch.tensor(features, dtype=torch.float), feature_names, fillna_info
    
    def _process_categorical_features(
        self, 
        df: pd.DataFrame,
        pretrained_embeddings: Optional[Dict] = None
    ) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """
        处理类别特征（嵌入）
        
        Args:
            df: 数据框
            pretrained_embeddings: 预训练的 embedding 字典（可选）
                格式: {"embeddings": {field_name: weight_tensor}, "category_mappings": {...}}
        
        Returns:
            cat_features: embedding 特征张量
            feature_names: 特征名列表
            encoding_info: 编码信息
        """
        categorical_cols = self.config.categorical_cols
        col_idx = self.config.col_idx
        
        if not categorical_cols:
            return None, [], {}
        
        col_name_map = {
            col_idx.payment_channel: "payment_channel",
            col_idx.debit_bic_code: "debit_bic_code",
            col_idx.bene_bic_code: "bene_bic_code",
            col_idx.instructed_currency: "instructed_currency",
            col_idx.payment_currency: "payment_currency",
            col_idx.credit_currency: "credit_currency",
            col_idx.mop: "mop"
        }
        
        embedding_dim = self.config.embedding_dim
        all_embeddings = []
        feature_names = []
        encoding_info = {}
        
        # 检查是否使用预训练 embedding
        use_pretrained = pretrained_embeddings is not None
        if use_pretrained:
            logging.info("使用预训练 embedding 权重")
            pretrained_emb_dict = pretrained_embeddings.get("embeddings", {})
            pretrained_mappings = pretrained_embeddings.get("category_mappings", {})
        else:
            logging.info("使用随机初始化 embedding（未启用预训练）")
        
        for col_i in categorical_cols:
            col_data = df.iloc[:, col_i].astype(str).fillna('UNKNOWN')
            col_name = col_name_map.get(col_i, f"cat_col_{col_i}")
            
            unique_values = col_data.unique()
            num_categories = len(unique_values)
            
            category_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            self.category_mappings[col_i] = category_to_idx
            
            indices = torch.tensor([category_to_idx[val] for val in col_data], dtype=torch.long)
            
            # 创建 embedding 层
            embedding_layer = torch.nn.Embedding(num_categories, embedding_dim)
            
            # 如果有预训练权重，尝试加载
            if use_pretrained and col_name in pretrained_emb_dict:
                pretrained_weight = pretrained_emb_dict[col_name]
                pretrained_mapping = pretrained_mappings.get(col_name, {})
                
                # 对齐预训练权重到当前数据的类别映射
                aligned_weight = self._align_pretrained_embedding(
                    pretrained_weight,
                    pretrained_mapping,
                    category_to_idx,
                    embedding_dim
                )
                embedding_layer.weight.data.copy_(aligned_weight)
                logging.info(f"  ✓ {col_name}: 已加载预训练权重 ({num_categories} 类别)")
            else:
                # 随机初始化
                torch.nn.init.xavier_uniform_(embedding_layer.weight)
                if use_pretrained:
                    logging.warning(f"  ⚠ {col_name}: 预训练权重不存在，使用随机初始化")
            
            self.embeddings[col_i] = embedding_layer
            
            # 提取 embedding 特征
            with torch.no_grad():
                embedded = embedding_layer(indices)
            
            all_embeddings.append(embedded)
            for i in range(embedding_dim):
                feature_names.append(f"{col_name}_emb_{i}")
            
            encoding_info[col_name] = {
                "num_categories": num_categories,
                "embedding_dim": embedding_dim,
                "pretrained": use_pretrained and col_name in pretrained_emb_dict
            }
        
        cat_features = torch.cat(all_embeddings, dim=1)
        return cat_features, feature_names, encoding_info
    
    def _align_pretrained_embedding(
        self,
        pretrained_weight: torch.Tensor,
        pretrained_mapping: Dict[str, int],
        current_mapping: Dict[str, int],
        embedding_dim: int
    ) -> torch.Tensor:
        """
        对齐预训练 embedding 到当前数据的类别映射
        
        对于当前数据中新出现的类别，使用随机初始化
        """
        num_current_categories = len(current_mapping)
        aligned_weight = torch.randn(num_current_categories, embedding_dim) * 0.01  # 新类别用小随机值
        
        # 复制预训练权重
        for category, current_idx in current_mapping.items():
            if category in pretrained_mapping:
                pretrained_idx = pretrained_mapping[category]
                if pretrained_idx < pretrained_weight.shape[0]:
                    aligned_weight[current_idx] = pretrained_weight[pretrained_idx]
        
        return aligned_weight
    
    def _process_time_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """处理时间特征"""
        col_idx = self.config.col_idx
        
        features = []
        feature_names = []
        fillna_info = {}
        
        for col_i, prefix in [(col_idx.txn_dt, "txn"), (col_idx.tds_dt, "tds")]:
            try:
                timestamps = pd.to_datetime(df.iloc[:, col_i], errors='coerce')
                
                null_count = timestamps.isnull().sum()
                if null_count > 0:
                    fillna_info[f"{prefix}_dt"] = {
                        "parse_failed_count": int(null_count),
                        "parse_failed_percent": float(null_count / len(df) * 100)
                    }
                
                hour = timestamps.dt.hour.fillna(0).values
                day_of_week = timestamps.dt.dayofweek.fillna(0).values
                day_of_month = timestamps.dt.day.fillna(1).values
                
                features.append(np.column_stack([hour, day_of_week, day_of_month]))
                feature_names.extend([f"{prefix}_hour", f"{prefix}_day_of_week", f"{prefix}_day_of_month"])
            except Exception as e:
                logging.warning(f"时间特征提取失败 ({prefix}): {e}")
        
        if len(features) > 0:
            features = np.hstack(features)
            features = self.time_scaler.fit_transform(features)
            return torch.tensor(features, dtype=torch.float), feature_names, fillna_info
        
        return None, [], fillna_info
    
    def _process_time_diff_feature(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """处理时间差特征"""
        col_idx = self.config.col_idx
        
        time_diff_info = {"computed": False, "stats": {}}
        
        try:
            txn_dt = pd.to_datetime(df.iloc[:, col_idx.txn_dt], errors='coerce')
            tds_dt = pd.to_datetime(df.iloc[:, col_idx.tds_dt], errors='coerce')
            
            time_diff = (tds_dt - txn_dt).dt.total_seconds()
            
            valid_mask = time_diff.notna()
            valid_count = valid_mask.sum()
            
            if valid_count > 0:
                time_diff = time_diff.fillna(0)
                
                # 标准化
                mean_val = time_diff[valid_mask].mean()
                std_val = time_diff[valid_mask].std()
                if std_val > 0:
                    time_diff_scaled = (time_diff - mean_val) / std_val
                else:
                    time_diff_scaled = time_diff - mean_val
                
                time_diff_info = {
                    "computed": True,
                    "valid_count": int(valid_count),
                    "valid_percent": float(valid_count / len(df) * 100),
                    "stats": {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "min": float(time_diff[valid_mask].min()),
                        "max": float(time_diff[valid_mask].max())
                    }
                }
                
                return torch.tensor(time_diff_scaled.values.reshape(-1, 1), dtype=torch.float), ["time_diff_seconds"], time_diff_info
        except Exception as e:
            logging.warning(f"时间差特征计算失败: {e}")
        
        return None, [], time_diff_info
    
    def _get_col_name(self, col_i: int, prefix: str) -> str:
        """获取列名"""
        col_idx = self.config.col_idx
        
        name_map = {
            col_idx.instructed_amount: "instructed_amount",
            col_idx.payment_amount: "payment_amount",
            col_idx.credit_amount: "credit_amount",
            col_idx.payment_channel: "payment_channel",
            col_idx.debit_bic_code: "debit_bic_code",
            col_idx.bene_bic_code: "bene_bic_code",
            col_idx.instructed_currency: "instructed_currency",
            col_idx.payment_currency: "payment_currency",
            col_idx.credit_currency: "credit_currency",
            col_idx.mop: "mop"
        }
        
        return name_map.get(col_i, f"{prefix}_col_{col_i}")
    
    def get_meta_info(self) -> Dict[str, Any]:
        """获取元信息"""
        return self.meta_info
