"""
数据加载模块
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import sys
sys.path.append('..')
from config import PreprocessConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.raw_shape: Optional[Tuple[int, int]] = None
        self.meta_info: Dict[str, Any] = {
            "load_info": {},
            "missing_info": {},
            "dtype_info": {}
        }
    
    def load_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """加载CSV数据"""
        path = file_path or self.config.data_path
        
        if not path:
            raise ValueError("数据路径未指定！请设置 config.data_path 或传入 file_path 参数")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        logging.info(f"正在加载数据: {path}")
        
        np.random.seed(self.config.random_seed)
        
        if self.config.use_full_dataset:
            logging.info("加载全量数据集...")
            self.df = pd.read_csv(path, header=0)
            logging.info(f"全量数据加载完成. Shape: {self.df.shape}")
        else:
            logging.info(f"加载采样数据集 ({self.config.sample_size} 行)...")
            self.df = pd.read_csv(path, header=0, nrows=self.config.sample_size)
            logging.info(f"采样数据加载完成. Shape: {self.df.shape}")
        
        self.raw_shape = self.df.shape
        
        # 过滤支付方和收款方账户相同的数据
        self._filter_same_account_transactions()
        
        self.meta_info["load_info"] = {
            "file_path": str(path),
            "raw_rows": self.raw_shape[0],
            "raw_cols": self.raw_shape[1],
            "filtered_rows": self.df.shape[0],
            "use_full_dataset": self.config.use_full_dataset,
            "sample_size": self.config.sample_size if not self.config.use_full_dataset else None,
            "column_names": list(self.df.columns)
        }
        
        self._check_missing_values()
        self._check_dtypes()
        
        return self.df
    
    def get_column_by_index(self, col_idx: int) -> pd.Series:
        """按索引获取列"""
        if self.df is None:
            raise ValueError("数据未加载！请先调用 load_csv()")
        
        if col_idx < 0 or col_idx >= self.df.shape[1]:
            raise IndexError(f"列索引 {col_idx} 超出范围 [0, {self.df.shape[1]-1}]")
        
        return self.df.iloc[:, col_idx]
    
    def get_columns_by_indices(self, col_indices: list) -> pd.DataFrame:
        """按索引列表获取多列"""
        if self.df is None:
            raise ValueError("数据未加载！请先调用 load_csv()")
        
        return self.df.iloc[:, col_indices]
    
    def _filter_same_account_transactions(self):
        """过滤支付方账户和收款方账户相同的交易"""
        if self.df is None:
            return
        
        src_col = self.config.src_col
        dst_col = self.config.dst_col
        
        before_count = len(self.df)
        
        # 找出支付方和收款方账户相同的记录
        same_account_mask = self.df.iloc[:, src_col] == self.df.iloc[:, dst_col]
        removed_count = same_account_mask.sum()
        
        # 过滤掉这些记录
        self.df = self.df[~same_account_mask].reset_index(drop=True)
        
        after_count = len(self.df)
        
        if removed_count > 0:
            logging.info(f"过滤相同账户交易: 移除 {removed_count:,} 条记录 "
                        f"({removed_count/before_count*100:.2f}%), "
                        f"剩余 {after_count:,} 条记录")
        else:
            logging.info("未发现相同账户交易")
    
    def _check_missing_values(self):
        """检查缺失值"""
        if self.df is None:
            return
        
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df) * 100).round(2)
        
        missing_info = {}
        for i, (count, percent) in enumerate(zip(missing_counts, missing_percent)):
            if count > 0:
                col_name = self.df.columns[i] if i < len(self.df.columns) else f"col_{i}"
                missing_info[f"col_{i}_{col_name}"] = {
                    "count": int(count),
                    "percent": float(percent)
                }
        
        self.meta_info["missing_info"] = missing_info
        
        if missing_info:
            logging.warning(f"发现 {len(missing_info)} 列有缺失值")
            for col, info in list(missing_info.items())[:5]:
                logging.warning(f"  {col}: {info['count']} ({info['percent']}%)")
        else:
            logging.info("无缺失值")
    
    def _check_dtypes(self):
        """检查数据类型"""
        if self.df is None:
            return
        
        dtype_info = {}
        for i, dtype in enumerate(self.df.dtypes):
            col_name = self.df.columns[i] if i < len(self.df.columns) else f"col_{i}"
            dtype_info[f"col_{i}_{col_name}"] = str(dtype)
        
        self.meta_info["dtype_info"] = dtype_info
        logging.info("数据类型检查完成")
    
    def get_meta_info(self) -> Dict[str, Any]:
        """获取元信息"""
        return self.meta_info
    
    def get_account_columns(self) -> Tuple[pd.Series, pd.Series]:
        """获取账户列（源账户和目标账户）"""
        src_col = self.get_column_by_index(self.config.src_col)
        dst_col = self.get_column_by_index(self.config.dst_col)
        return src_col, dst_col
    
    def get_time_columns(self) -> Tuple[pd.Series, pd.Series]:
        """获取时间列"""
        col_idx = self.config.col_idx
        txn_dt = self.get_column_by_index(col_idx.txn_dt)
        tds_dt = self.get_column_by_index(col_idx.tds_dt)
        return txn_dt, tds_dt
    
    def get_numerical_columns(self) -> pd.DataFrame:
        """获取数值列"""
        return self.get_columns_by_indices(self.config.numerical_cols)
    
    def get_categorical_columns(self) -> pd.DataFrame:
        """获取类别列"""
        return self.get_columns_by_indices(self.config.categorical_cols)
    
    def get_transaction_ids(self) -> pd.Series:
        """获取交易ID列"""
        return self.get_column_by_index(self.config.col_idx.uetr)
