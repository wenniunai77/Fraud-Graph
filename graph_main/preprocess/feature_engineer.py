"""
特征工程模块
负责处理节点特征和边特征，包括时间差特征

特征说明见 config.py 中的 NODE_FEATURE_DESCRIPTION 和 EDGE_FEATURE_DESCRIPTION
"""

import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

from config import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class FeatureEngineer:
    """
    特征工程器
    负责边特征和节点特征的构建
    """
    
    def __init__(self, config: PreprocessConfig):
        """
        初始化特征工程器
        
        Args:
            config: 预处理配置对象
        """
        self.config = config
        self.label_encoders: Dict[int, LabelEncoder] = {}
        self.numerical_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        
        # 记录特征工程元信息（用于检查）
        self.meta_info: Dict[str, Any] = {
            "edge_features": {},
            "node_features": {},
            "time_diff_info": {},
            "fillna_info": {},  # 记录哪些值被fillna处理
            "encoding_info": {}
        }
    
    def build_edge_features(self, df: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        """
        构建边特征
        
        边级特征包括:
        【数值特征】(标准化后)
        - instructed_amount: 客户指定金额 (第6列)
        - payment_amount: 银行使用金额 (第8列)
        - credit_amount: 收款方接收金额 (第10列)
        
        【类别特征】(LabelEncoder编码后)
        - payment_channel_encoded: 交易渠道编码 (第1列)
        - debit_bic_code_encoded: 支付方BIC码编码 (第2列)
        - bene_bic_code_encoded: 收款方BIC码编码 (第3列)
        - evt_tran_stat_cde_encoded: 支付状态码编码 (第4列)
        - instructed_currency_encoded: 客户指定币种编码 (第5列)
        - payment_currency_encoded: 银行使用币种编码 (第7列)
        - credit_currency_encoded: 收款方接收币种编码 (第9列)
        - mop_encoded: 付款方式编码 (第13列)
        
        【时间特征】
        - txn_hour: 交易发生小时 (0-23)
        - txn_day_of_week: 交易发生星期几 (0-6, 0=周一)
        - txn_day_of_month: 交易发生日期 (1-31)
        - tds_hour: 入库小时
        - tds_day_of_week: 入库星期几
        - tds_day_of_month: 入库日期
        
        【时间差特征】
        - time_diff_seconds: tds_dt - txn_dt 的时间差（秒）
          含义：从事件发生到入库的延迟时间
        
        Args:
            df: 原始DataFrame
            
        Returns:
            (边特征张量, 特征名称列表)
        """
        logging.info("开始构建边特征...")
        
        all_features = []
        feature_names = []
        
        # ============================================================
        # 1. 数值特征处理
        # ============================================================
        logging.info("处理数值特征...")
        num_features, num_names, num_fillna_info = self._process_numerical_features(df)
        if num_features is not None:
            all_features.append(num_features)
            feature_names.extend(num_names)
            self.meta_info["fillna_info"]["numerical"] = num_fillna_info
        
        # ============================================================
        # 2. 类别特征处理
        # ============================================================
        logging.info("处理类别特征...")
        cat_features, cat_names, cat_encoding_info = self._process_categorical_features(df)
        if cat_features is not None:
            all_features.append(cat_features)
            feature_names.extend(cat_names)
            self.meta_info["encoding_info"] = cat_encoding_info
        
        # ============================================================
        # 3. 时间特征处理
        # ============================================================
        logging.info("处理时间特征...")
        time_features, time_names, time_fillna_info = self._process_time_features(df)
        if time_features is not None:
            all_features.append(time_features)
            feature_names.extend(time_names)
            self.meta_info["fillna_info"]["time"] = time_fillna_info
        
        # ============================================================
        # 4. 时间差特征处理 (tds_dt - txn_dt)
        # ============================================================
        logging.info("处理时间差特征 (time_diff = tds_dt - txn_dt)...")
        time_diff_features, time_diff_names, time_diff_info = self._process_time_diff_feature(df)
        if time_diff_features is not None:
            all_features.append(time_diff_features)
            feature_names.extend(time_diff_names)
            self.meta_info["time_diff_info"] = time_diff_info
        
        # ============================================================
        # 合并所有特征
        # ============================================================
        if len(all_features) > 0:
            edge_features = torch.cat(all_features, dim=1)
        else:
            logging.warning("未提取到任何边特征!")
            edge_features = torch.zeros((len(df), 1), dtype=torch.float)
            feature_names = ["dummy"]
        
        # 记录边特征信息
        self.meta_info["edge_features"] = {
            "shape": list(edge_features.shape),
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        logging.info(f"边特征构建完成。Shape: {edge_features.shape}")
        logging.info(f"特征列表: {feature_names}")
        
        return edge_features, feature_names
    
    def _process_numerical_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """
        处理数值特征
        
        数值特征包括:
        - 第6列 instructed_amount: 客户指定金额
        - 第8列 payment_amount: 银行使用金额  
        - 第10列 credit_amount: 收款方接收金额
        
        Returns:
            (特征张量, 特征名称列表, fillna信息)
        """
        numerical_cols = self.config.numerical_cols
        col_idx = self.config.col_idx
        
        if not numerical_cols:
            return None, [], {}
        
        # 特征名称映射
        col_name_map = {
            col_idx.instructed_amount: "instructed_amount",   # 第6列: 客户指定金额
            col_idx.payment_amount: "payment_amount",         # 第8列: 银行使用金额
            col_idx.credit_amount: "credit_amount"            # 第10列: 收款方接收金额
        }
        
        features = []
        feature_names = []
        fillna_info = {}
        
        for col_i in numerical_cols:
            col_data = df.iloc[:, col_i]
            col_name = col_name_map.get(col_i, f"num_col_{col_i}")
            
            # 转换为数值类型
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            
            # 记录被fillna的数量
            null_count = numeric_data.isnull().sum()
            if null_count > 0:
                fillna_info[col_name] = {
                    "fillna_count": int(null_count),
                    "fillna_percent": float(null_count / len(df) * 100),
                    "fillna_value": 0
                }
                logging.warning(f"列 {col_name} (第{col_i}列): {null_count} 个值被fillna为0 "
                              f"({null_count/len(df)*100:.2f}%)")
            
            # 填充缺失值为0
            numeric_data = numeric_data.fillna(0)
            features.append(numeric_data.values.reshape(-1, 1))
            feature_names.append(col_name)
        
        # 合并并标准化
        features = np.hstack(features)
        features = self.numerical_scaler.fit_transform(features)
        
        return torch.tensor(features, dtype=torch.float), feature_names, fillna_info
    
    def _process_categorical_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """
        处理类别特征
        
        类别特征包括:
        - 第1列 payment_channel: 交易渠道 (NET/BIB/HCN/BFT/PIB)
        - 第2列 debit_bic_code: 支付方BIC码
        - 第3列 bene_bic_code: 收款方BIC码
        - 第4列 evt_tran_stat_cde: 支付状态码
        - 第5列 instructed_currency: 客户指定币种
        - 第7列 payment_currency: 银行使用币种
        - 第9列 credit_currency: 收款方接收币种
        - 第13列 mop: 付款方式
        
        Returns:
            (特征张量, 特征名称列表, encoding信息)
        """
        categorical_cols = self.config.categorical_cols
        col_idx = self.config.col_idx
        
        if not categorical_cols:
            return None, [], {}
        
        # 特征名称映射
        col_name_map = {
            col_idx.payment_channel: "payment_channel",        # 第1列: 交易渠道
            col_idx.debit_bic_code: "debit_bic_code",         # 第2列: 支付方BIC码
            col_idx.bene_bic_code: "bene_bic_code",           # 第3列: 收款方BIC码
            col_idx.evt_tran_stat_cde: "evt_tran_stat_cde",   # 第4列: 支付状态码
            col_idx.instructed_currency: "instructed_currency", # 第5列: 客户指定币种
            col_idx.payment_currency: "payment_currency",      # 第7列: 银行使用币种
            col_idx.credit_currency: "credit_currency",        # 第9列: 收款方接收币种
            col_idx.mop: "mop"                                  # 第13列: 付款方式
        }
        
        features = []
        feature_names = []
        encoding_info = {}
        
        for col_i in categorical_cols:
            col_data = df.iloc[:, col_i].astype(str).fillna('UNKNOWN')
            col_name = col_name_map.get(col_i, f"cat_col_{col_i}")
            
            # 使用LabelEncoder编码
            le = LabelEncoder()
            encoded = le.fit_transform(col_data)
            
            # 保存encoder
            self.label_encoders[col_i] = le
            
            features.append(encoded.reshape(-1, 1))
            feature_names.append(f"{col_name}_encoded")
            
            # 记录编码信息
            encoding_info[col_name] = {
                "num_classes": len(le.classes_),
                "classes": list(le.classes_)[:10],  # 只记录前10个类别
                "has_unknown": 'UNKNOWN' in le.classes_
            }
        
        features = np.hstack(features)
        return torch.tensor(features, dtype=torch.float), feature_names, encoding_info
    
    def _process_time_features(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """
        处理时间特征
        
        从时间列提取:
        - 第11列 txn_dt: 事件发生时间
          - txn_hour: 小时 (0-23)
          - txn_day_of_week: 星期几 (0-6, 0=周一)
          - txn_day_of_month: 日期 (1-31)
        - 第12列 tds_dt: 入库时间
          - tds_hour: 小时
          - tds_day_of_week: 星期几
          - tds_day_of_month: 日期
        
        Returns:
            (特征张量, 特征名称列表, fillna信息)
        """
        col_idx = self.config.col_idx
        
        features = []
        feature_names = []
        fillna_info = {}
        
        # 处理 txn_dt (第11列: 事件发生时间)
        txn_features, txn_names, txn_fillna = self._extract_time_components(
            df.iloc[:, col_idx.txn_dt], 
            prefix="txn"
        )
        if txn_features is not None:
            features.append(txn_features)
            feature_names.extend(txn_names)
            if txn_fillna:
                fillna_info["txn_dt"] = txn_fillna
        
        # 处理 tds_dt (第12列: 入库时间戳)
        tds_features, tds_names, tds_fillna = self._extract_time_components(
            df.iloc[:, col_idx.tds_dt],
            prefix="tds"
        )
        if tds_features is not None:
            features.append(tds_features)
            feature_names.extend(tds_names)
            if tds_fillna:
                fillna_info["tds_dt"] = tds_fillna
        
        if len(features) > 0:
            features = np.hstack(features)
            # 标准化时间特征
            features = self.time_scaler.fit_transform(features)
            return torch.tensor(features, dtype=torch.float), feature_names, fillna_info
        
        return None, [], fillna_info
    
    def _extract_time_components(self, time_series: pd.Series, prefix: str) -> Tuple[Optional[np.ndarray], List[str], Dict]:
        """
        从时间序列提取时间组件
        
        Args:
            time_series: 时间序列
            prefix: 特征名称前缀
            
        Returns:
            (特征数组, 特征名称列表, fillna信息)
        """
        try:
            timestamps = pd.to_datetime(time_series, errors='coerce')
            
            # 记录解析失败的数量
            null_count = timestamps.isnull().sum()
            fillna_info = {}
            if null_count > 0:
                fillna_info = {
                    "parse_failed_count": int(null_count),
                    "parse_failed_percent": float(null_count / len(time_series) * 100)
                }
                logging.warning(f"{prefix}_dt: {null_count} 个时间值解析失败")
            
            # 提取时间特征
            hour = timestamps.dt.hour.fillna(0).values                    # 小时 (0-23)
            day_of_week = timestamps.dt.dayofweek.fillna(0).values       # 星期几 (0-6)
            day_of_month = timestamps.dt.day.fillna(1).values            # 日期 (1-31)
            
            features = np.column_stack([hour, day_of_week, day_of_month])
            feature_names = [
                f"{prefix}_hour",         # 小时
                f"{prefix}_day_of_week",  # 星期几
                f"{prefix}_day_of_month"  # 日期
            ]
            
            return features, feature_names, fillna_info
            
        except Exception as e:
            logging.error(f"时间特征提取失败 ({prefix}): {e}")
            return None, [], {"error": str(e)}
    
    def _process_time_diff_feature(self, df: pd.DataFrame) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """
        处理时间差特征
        
        time_diff = tds_dt - txn_dt
        
        含义：从事件发生(txn_dt)到入库(tds_dt)的延迟时间
        用途：异常交易可能有异常的处理延迟
        
        Returns:
            (特征张量, 特征名称列表, 时间差信息)
        """
        col_idx = self.config.col_idx
        
        time_diff_info = {
            "computed": False,
            "stats": {},
            "fillna_info": {}
        }
        
        try:
            # 解析两个时间列
            # 第11列: txn_dt - 事件发生时间
            txn_dt = pd.to_datetime(df.iloc[:, col_idx.txn_dt], errors='coerce')
            # 第12列: tds_dt - 入库时间戳  
            tds_dt = pd.to_datetime(df.iloc[:, col_idx.tds_dt], errors='coerce')
            
            # 计算时间差（秒）
            # time_diff = tds_dt - txn_dt
            # 正值表示入库时间晚于事件发生时间（正常情况）
            time_diff = (tds_dt - txn_dt).dt.total_seconds()
            
            # 记录统计信息
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
            
            # 处理无效值
            null_count = time_diff.isnull().sum()
            if null_count > 0:
                time_diff_info["fillna_info"] = {
                    "fillna_count": int(null_count),
                    "fillna_percent": float(null_count / len(df) * 100),
                    "fillna_value": "median"  # 使用中位数填充
                }
                # 使用中位数填充缺失值
                median_val = valid_diff.median() if len(valid_diff) > 0 else 0
                time_diff = time_diff.fillna(median_val)
                logging.warning(f"time_diff: {null_count} 个值被fillna为中位数 {median_val:.2f}秒")
            
            # 标准化
            time_diff_array = time_diff.values.reshape(-1, 1)
            time_diff_scaled = StandardScaler().fit_transform(time_diff_array)
            
            time_diff_info["computed"] = True
            
            logging.info(f"时间差特征构建成功:")
            logging.info(f"  - 有效值: {time_diff_info['stats']['valid_count']}")
            logging.info(f"  - 平均延迟: {time_diff_info['stats']['mean_seconds']:.2f}秒")
            logging.info(f"  - 中位延迟: {time_diff_info['stats']['median_seconds']:.2f}秒")
            if time_diff_info['stats']['negative_count'] > 0:
                logging.warning(f"  - 负值数量: {time_diff_info['stats']['negative_count']} (入库时间早于事件时间)")
            
            return (
                torch.tensor(time_diff_scaled, dtype=torch.float),
                ["time_diff_seconds"],  # 特征名: 时间差（秒）
                time_diff_info
            )
            
        except Exception as e:
            logging.error(f"时间差特征计算失败: {e}")
            time_diff_info["error"] = str(e)
            return None, [], time_diff_info
    
    def build_node_features(
        self, 
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        df: pd.DataFrame
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        构建节点特征（从边特征聚合）
        
        节点级特征包括:
        
        【作为发送方(源节点)时的聚合特征】
        - src_avg_*: 作为发送方时，各边特征的平均值
        - src_tx_count: 作为发送方的交易次数（出度）
        
        【作为接收方(目标节点)时的聚合特征】
        - dst_avg_*: 作为接收方时，各边特征的平均值
        - dst_tx_count: 作为接收方的交易次数（入度）
        
        【统计特征】
        - total_degree: 总交易次数（出度+入度）
        - in_out_ratio: 入度/出度比值
        
        Args:
            edge_features: 边特征张量 [num_edges, num_edge_features]
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点数量
            df: 原始数据（用于提取额外统计特征）
            
        Returns:
            (节点特征张量, 特征名称列表)
        """
        logging.info("开始构建节点特征...")
        
        num_edge_features = edge_features.shape[1]
        
        try:
            from torch_scatter import scatter_mean, scatter_add
            use_scatter = True
            logging.info("使用 torch_scatter 进行高效聚合")
        except ImportError:
            use_scatter = False
            logging.warning("torch_scatter 不可用，使用手动聚合（较慢）")
        
        if use_scatter:
            node_features, feature_names = self._aggregate_with_scatter(
                edge_features, edge_index, num_nodes
            )
        else:
            node_features, feature_names = self._aggregate_manual(
                edge_features, edge_index, num_nodes
            )
        
        # 处理NaN和Inf
        node_features = torch.where(
            torch.isnan(node_features) | torch.isinf(node_features),
            torch.zeros_like(node_features),
            node_features
        )
        
        # 记录节点特征信息
        self.meta_info["node_features"] = {
            "shape": list(node_features.shape),
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        logging.info(f"节点特征构建完成。Shape: {node_features.shape}")
        
        return node_features, feature_names
    
    def _aggregate_with_scatter(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """使用torch_scatter进行高效聚合"""
        from torch_scatter import scatter_mean, scatter_add
        
        num_edge_features = edge_features.shape[1]
        
        # 1. 聚合作为源节点(发送方)的交易特征均值
        # src_avg_*: 作为发送方时，各边特征的平均值
        src_feat = scatter_mean(edge_features, edge_index[0], dim=0, dim_size=num_nodes)
        
        # 2. 聚合作为目标节点(接收方)的交易特征均值
        # dst_avg_*: 作为接收方时，各边特征的平均值
        dst_feat = scatter_mean(edge_features, edge_index[1], dim=0, dim_size=num_nodes)
        
        # 3. 统计特征：交易次数
        # src_tx_count: 作为发送方的交易次数（出度）
        src_counts = scatter_add(
            torch.ones(edge_index.shape[1]), 
            edge_index[0], dim=0, dim_size=num_nodes
        )
        # dst_tx_count: 作为接收方的交易次数（入度）
        dst_counts = scatter_add(
            torch.ones(edge_index.shape[1]), 
            edge_index[1], dim=0, dim_size=num_nodes
        )
        
        # 4. 总度数和入出度比值
        # total_degree: 总交易次数
        total_degree = src_counts + dst_counts
        # in_out_ratio: 入度/出度比值（避免除零）
        in_out_ratio = dst_counts / (src_counts + 1e-8)
        
        # 组合所有特征
        node_features = torch.cat([
            src_feat,                          # 源节点特征均值
            dst_feat,                          # 目标节点特征均值
            src_counts.unsqueeze(1),           # 出度
            dst_counts.unsqueeze(1),           # 入度
            total_degree.unsqueeze(1),         # 总度数
            in_out_ratio.unsqueeze(1)          # 入出度比值
        ], dim=1)
        
        # 构建特征名称
        feature_names = []
        for i in range(num_edge_features):
            feature_names.append(f"src_avg_feat_{i}")
        for i in range(num_edge_features):
            feature_names.append(f"dst_avg_feat_{i}")
        feature_names.extend([
            "src_tx_count",    # 出度：作为发送方的交易次数
            "dst_tx_count",    # 入度：作为接收方的交易次数
            "total_degree",    # 总度数
            "in_out_ratio"     # 入出度比值
        ])
        
        return node_features, feature_names
    
    def _aggregate_manual(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """手动聚合（当torch_scatter不可用时）"""
        num_edge_features = edge_features.shape[1]
        
        # 初始化
        src_feat_sum = torch.zeros((num_nodes, num_edge_features))
        dst_feat_sum = torch.zeros((num_nodes, num_edge_features))
        src_counts = torch.zeros(num_nodes)
        dst_counts = torch.zeros(num_nodes)
        
        edge_index_np = edge_index.numpy()
        edge_features_np = edge_features.numpy()
        
        # 遍历所有边
        for i in range(edge_index.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            feat = edge_features_np[i]
            
            src_feat_sum[src] += torch.tensor(feat)
            dst_feat_sum[dst] += torch.tensor(feat)
            src_counts[src] += 1
            dst_counts[dst] += 1
        
        # 计算平均值
        src_feat = src_feat_sum / (src_counts.unsqueeze(1) + 1e-8)
        dst_feat = dst_feat_sum / (dst_counts.unsqueeze(1) + 1e-8)
        
        # 总度数和比值
        total_degree = src_counts + dst_counts
        in_out_ratio = dst_counts / (src_counts + 1e-8)
        
        # 组合特征
        node_features = torch.cat([
            src_feat,
            dst_feat,
            src_counts.unsqueeze(1),
            dst_counts.unsqueeze(1),
            total_degree.unsqueeze(1),
            in_out_ratio.unsqueeze(1)
        ], dim=1)
        
        # 构建特征名称
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
        """获取特征工程元信息"""
        return self.meta_info
