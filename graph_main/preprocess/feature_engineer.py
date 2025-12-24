import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from collections import Counter

from config import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class FeatureEngineer:
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.embeddings: Dict[int, torch.nn.Embedding] = {}
        self.category_mappings: Dict[int, Dict[str, int]] = {}
        self.numerical_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        
        self.meta_info: Dict[str, Any] = {
            "edge_features": {},
            "node_features": {},
            "time_diff_info": {},
            "fillna_info": {},
            "encoding_info": {}
        }
    
    def build_edge_features(self, df: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        logging.info("Building edge features...")
        
        all_features = []
        feature_names = []
        
        logging.info("Processing numerical features...")
        num_features, num_names, num_fillna_info = self._process_numerical_features(df)
        if num_features is not None:
            all_features.append(num_features)
            feature_names.extend(num_names)
            self.meta_info["fillna_info"]["numerical"] = num_fillna_info
        
        logging.info("Processing categorical features...")
        cat_features, cat_names, cat_encoding_info = self._process_categorical_features(df)
        if cat_features is not None:
            all_features.append(cat_features)
            feature_names.extend(cat_names)
            self.meta_info["encoding_info"] = cat_encoding_info
        
        logging.info("Processing time features...")
        time_features, time_names, time_fillna_info = self._process_time_features(df)
        if time_features is not None:
            all_features.append(time_features)
            feature_names.extend(time_names)
            self.meta_info["fillna_info"]["time"] = time_fillna_info
        
        logging.info("Processing time difference feature (time_diff = tds_dt - txn_dt)...")
        time_diff_features, time_diff_names, time_diff_info = self._process_time_diff_feature(df)
        if time_diff_features is not None:
            all_features.append(time_diff_features)
            feature_names.extend(time_diff_names)
            self.meta_info["time_diff_info"] = time_diff_info
        
        if len(all_features) > 0:
            edge_features = torch.cat(all_features, dim=1)
        else:
            logging.warning("No edge features extracted!")
            edge_features = torch.zeros((len(df), 1), dtype=torch.float)
            feature_names = ["dummy"]
        
        self.meta_info["edge_features"] = {
            "shape": list(edge_features.shape),
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        logging.info(f"Edge features built. Shape: {edge_features.shape}")
        logging.info(f"Feature list: {feature_names}")
        
        return edge_features, feature_names
    
    def _process_numerical_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
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
                logging.warning(f"Column {col_name} (col {col_i}): {null_count} values filled with 0 "
                              f"({null_count/len(df)*100:.2f}%)")
            
            numeric_data = numeric_data.fillna(0)
            features.append(numeric_data.values.reshape(-1, 1))
            feature_names.append(col_name)
        
        features = np.hstack(features)
        features = self.numerical_scaler.fit_transform(features)
        
        return torch.tensor(features, dtype=torch.float), feature_names, fillna_info
    
    def _process_categorical_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
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
        
        for col_i in categorical_cols:
            col_data = df.iloc[:, col_i].astype(str).fillna('UNKNOWN')
            col_name = col_name_map.get(col_i, f"cat_col_{col_i}")
            
            unique_values = col_data.unique()
            num_categories = len(unique_values)
            
            category_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            self.category_mappings[col_i] = category_to_idx
            
            indices = torch.tensor([category_to_idx[val] for val in col_data], dtype=torch.long)
            
            embedding_layer = torch.nn.Embedding(num_categories, embedding_dim)
            torch.nn.init.xavier_uniform_(embedding_layer.weight)
            self.embeddings[col_i] = embedding_layer
            
            with torch.no_grad():
                embedded = embedding_layer(indices)
            
            all_embeddings.append(embedded)
            for i in range(embedding_dim):
                feature_names.append(f"{col_name}_emb_{i}")
            
            encoding_info[col_name] = {
                "num_categories": num_categories,
                "embedding_dim": embedding_dim,
                "sample_categories": list(unique_values)[:10],
                "has_unknown": 'UNKNOWN' in unique_values
            }
        
        cat_features = torch.cat(all_embeddings, dim=1)
        return cat_features, feature_names, encoding_info
    
    def _process_time_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        col_idx = self.config.col_idx
        
        features = []
        feature_names = []
        fillna_info = {}
        
        txn_features, txn_names, txn_fillna = self._extract_time_components(
            df.iloc[:, col_idx.txn_dt], 
            prefix="txn"
        )
        if txn_features is not None:
            features.append(txn_features)
            feature_names.extend(txn_names)
            if txn_fillna:
                fillna_info["txn_dt"] = txn_fillna
        
        tds_features, tds_names, tds_fillna = self._extract_time_components(
            df.iloc[:, col_idx.tds_dt],
            prefix="tds",
            double_parse=True
        )
        if tds_features is not None:
            features.append(tds_features)
            feature_names.extend(tds_names)
            if tds_fillna:
                fillna_info["tds_dt"] = tds_fillna
        
        if len(features) > 0:
            features = np.hstack(features)
            features = self.time_scaler.fit_transform(features)
            return torch.tensor(features, dtype=torch.float), feature_names, fillna_info
        
        return None, [], fillna_info
    
    def _extract_time_components(self, time_series: pd.Series, prefix: str, double_parse: bool = False) -> Tuple[Optional[np.ndarray], List[str], Dict]:
        try:
            timestamps = pd.to_datetime(time_series, errors='coerce')
            
            if double_parse:
                failed_mask = timestamps.isnull()
                if failed_mask.any():
                    logging.info(f"{prefix}_dt: First parse failed for {failed_mask.sum()} values, trying second parse...")
                    timestamps_retry = pd.to_datetime(time_series[failed_mask], errors='coerce')
                    timestamps.loc[failed_mask] = timestamps_retry
            
            null_count = timestamps.isnull().sum()
            fillna_info = {}
            if null_count > 0:
                fillna_info = {
                    "parse_failed_count": int(null_count),
                    "parse_failed_percent": float(null_count / len(time_series) * 100)
                }
                logging.warning(f"{prefix}_dt: {null_count} time values failed to parse")
            
            hour = timestamps.dt.hour.fillna(0).values
            day_of_week = timestamps.dt.dayofweek.fillna(0).values
            day_of_month = timestamps.dt.day.fillna(1).values
            
            features = np.column_stack([hour, day_of_week, day_of_month])
            feature_names = [
                f"{prefix}_hour",
                f"{prefix}_day_of_week",
                f"{prefix}_day_of_month"
            ]
            
            return features, feature_names, fillna_info
            
        except Exception as e:
            logging.error(f"Time feature extraction failed ({prefix}): {e}")
            return None, [], {"error": str(e)}
    
    def _process_time_diff_feature(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        col_idx = self.config.col_idx
        
        time_diff_info = {
            "computed": False,
            "stats": {},
            "fillna_info": {}
        }
        
        try:
            txn_dt = pd.to_datetime(df.iloc[:, col_idx.txn_dt], errors='coerce')
            
            tds_dt = pd.to_datetime(df.iloc[:, col_idx.tds_dt], errors='coerce')
            failed_mask = tds_dt.isnull()
            if failed_mask.any():
                tds_dt_retry = pd.to_datetime(df.iloc[:, col_idx.tds_dt][failed_mask], errors='coerce')
                tds_dt.loc[failed_mask] = tds_dt_retry
            
            time_diff = (tds_dt - txn_dt).dt.total_seconds()
            
            valid_diff = time_diff.dropna()
            time_diff_info["stats"] = {
                "valid_count": int(len(valid_diff)),
                "invalid_count": int(time_diff.isnull().sum()),
                "mean_seconds": float(valid_diff.mean()) if len(valid_diff) > 0 else None,
                "median_seconds": float(valid_diff.median()) if len(valid_diff) > 0 else None,
                "min_seconds": float(valid_diff.min()) if len(valid_diff) > 0 else None,
                "max_seconds": float(valid_diff.max()) if len(valid_diff) > 0 else None,
                "std_seconds": float(valid_diff.std()) if len(valid_diff) > 0 else None,
                "negative_count": int((valid_diff < 0).sum()) if len(valid_diff) > 0 else 0
            }
            
            null_count = time_diff.isnull().sum()
            if null_count > 0:
                time_diff_info["fillna_info"] = {
                    "fillna_count": int(null_count),
                    "fillna_percent": float(null_count / len(df) * 100),
                    "fillna_value": "median"
                }
                median_val = valid_diff.median() if len(valid_diff) > 0 else 0
                time_diff = time_diff.fillna(median_val)
                logging.warning(f"time_diff: {null_count} values filled with median {median_val:.2f} seconds")
            
            time_diff_array = time_diff.values.reshape(-1, 1)
            time_diff_scaled = StandardScaler().fit_transform(time_diff_array)
            
            time_diff_info["computed"] = True
            
            logging.info(f"Time diff feature built successfully:")
            logging.info(f"  - Valid values: {time_diff_info['stats']['valid_count']}")
            logging.info(f"  - Mean delay: {time_diff_info['stats']['mean_seconds']:.2f} seconds")
            logging.info(f"  - Median delay: {time_diff_info['stats']['median_seconds']:.2f} seconds")
            if time_diff_info['stats']['negative_count'] > 0:
                logging.warning(f"  - Negative count: {time_diff_info['stats']['negative_count']} (tds earlier than txn)")
            
            return (
                torch.tensor(time_diff_scaled, dtype=torch.float),
                ["time_diff_seconds"],
                time_diff_info
            )
            
        except Exception as e:
            logging.error(f"Time diff feature computation failed: {e}")
            time_diff_info["error"] = str(e)
            return None, [], time_diff_info
    
    def build_node_features(
        self, 
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        df: pd.DataFrame
    ) -> Tuple[torch.Tensor, List[str]]:
        logging.info("Building node features...")
        
        num_edge_features = edge_features.shape[1]
        
        try:
            from torch_scatter import scatter_mean, scatter_add
            use_scatter = True
            logging.info("Using torch_scatter for efficient aggregation")
        except ImportError:
            use_scatter = False
            logging.warning("torch_scatter not available, using manual aggregation (slower)")
        
        if use_scatter:
            node_features, feature_names = self._aggregate_with_scatter(
                edge_features, edge_index, num_nodes
            )
        else:
            node_features, feature_names = self._aggregate_manual(
                edge_features, edge_index, num_nodes
            )
        
        node_features = torch.where(
            torch.isnan(node_features) | torch.isinf(node_features),
            torch.zeros_like(node_features),
            node_features
        )
        
        self.meta_info["node_features"] = {
            "shape": list(node_features.shape),
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        logging.info(f"Node features built. Shape: {node_features.shape}")
        
        return node_features, feature_names
    
    def _aggregate_with_scatter(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, List[str]]:
        from torch_scatter import scatter_mean, scatter_add
        
        num_edge_features = edge_features.shape[1]
        
        src_feat = scatter_mean(edge_features, edge_index[0], dim=0, dim_size=num_nodes)
        dst_feat = scatter_mean(edge_features, edge_index[1], dim=0, dim_size=num_nodes)
        
        src_counts = scatter_add(
            torch.ones(edge_index.shape[1]), 
            edge_index[0], dim=0, dim_size=num_nodes
        )
        dst_counts = scatter_add(
            torch.ones(edge_index.shape[1]), 
            edge_index[1], dim=0, dim_size=num_nodes
        )
        
        total_degree = src_counts + dst_counts
        in_out_ratio = dst_counts / (src_counts + 1e-8)
        
        node_features = torch.cat([
            src_feat,
            dst_feat,
            src_counts.unsqueeze(1),
            dst_counts.unsqueeze(1),
            total_degree.unsqueeze(1),
            in_out_ratio.unsqueeze(1)
        ], dim=1)
        
        feature_names = []
        for i in range(num_edge_features):
            feature_names.append(f"src_avg_feat_{i}")
        for i in range(num_edge_features):
            feature_names.append(f"dst_avg_feat_{i}")
        feature_names.extend([
            "src_tx_count",
            "dst_tx_count",
            "total_degree",
            "in_out_ratio"
        ])
        
        return node_features, feature_names
    
    def _aggregate_manual(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, List[str]]:
        num_edge_features = edge_features.shape[1]
        
        src_feat_sum = torch.zeros((num_nodes, num_edge_features))
        dst_feat_sum = torch.zeros((num_nodes, num_edge_features))
        src_counts = torch.zeros(num_nodes)
        dst_counts = torch.zeros(num_nodes)
        
        edge_index_np = edge_index.numpy()
        edge_features_np = edge_features.numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            feat = edge_features_np[i]
            
            src_feat_sum[src] += torch.tensor(feat)
            dst_feat_sum[dst] += torch.tensor(feat)
            src_counts[src] += 1
            dst_counts[dst] += 1
        
        src_feat = src_feat_sum / (src_counts.unsqueeze(1) + 1e-8)
        dst_feat = dst_feat_sum / (dst_counts.unsqueeze(1) + 1e-8)
        
        total_degree = src_counts + dst_counts
        in_out_ratio = dst_counts / (src_counts + 1e-8)
        
        node_features = torch.cat([
            src_feat,
            dst_feat,
            src_counts.unsqueeze(1),
            dst_counts.unsqueeze(1),
            total_degree.unsqueeze(1),
            in_out_ratio.unsqueeze(1)
        ], dim=1)
        
        feature_names = []
        for i in range(num_edge_features):
            feature_names.append(f"src_avg_feat_{i}")
        for i in range(num_edge_features):
            feature_names.append(f"dst_avg_feat_{i}")
        feature_names.extend([
            "src_tx_count",
            "dst_tx_count", 
            "total_degree",
            "in_out_ratio"
        ])
        
        return node_features, feature_names
    
    def get_meta_info(self) -> Dict[str, Any]:
        return self.meta_info
