"""
数据加载模块
负责从CSV加载原始数据

注意：所有字段访问使用列索引而非列名
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from config import PreprocessConfig, ColumnIndex

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class DataLoader:
    """
    数据加载器
    负责从CSV加载原始数据并进行基础处理
    """
    
    def __init__(self, config: PreprocessConfig):
        """
        初始化数据加载器
        
        Args:
            config: 预处理配置对象
        """
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.raw_shape: Optional[Tuple[int, int]] = None
        
        # 记录数据加载元信息（用于检查）
        self.meta_info: Dict[str, Any] = {
            "load_info": {},
            "missing_info": {},
            "dtype_info": {}
        }
    
    def load_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载CSV数据
        使用列索引访问数据，不依赖列名
        
        Args:
            file_path: CSV文件路径，如果为None则使用配置中的路径
            
        Returns:
            加载的DataFrame
        """
        path = file_path or self.config.data_path
        
        if not path:
            raise ValueError("数据路径未指定！请设置 config.data_path 或传入 file_path 参数")
        
        logging.info(f"开始加载数据: {path}")
        
        # 设置随机种子（用于采样）
        np.random.seed(self.config.random_seed)
        
        if self.config.use_full_dataset:
            logging.info("加载完整数据集...")
            self.df = pd.read_csv(path, header=0)
            logging.info(f"完整数据集加载完成。Shape: {self.df.shape}")
        else:
            logging.info(f"加载采样数据集 ({self.config.sample_size} 行)...")
            self.df = pd.read_csv(path, header=0, nrows=self.config.sample_size)
            logging.info(f"采样数据加载完成。Shape: {self.df.shape}")
        
        self.raw_shape = self.df.shape
        
        # 记录加载信息
        self.meta_info["load_info"] = {
            "file_path": path,
            "raw_rows": self.raw_shape[0],
            "raw_cols": self.raw_shape[1],
            "use_full_dataset": self.config.use_full_dataset,
            "sample_size": self.config.sample_size if not self.config.use_full_dataset else None,
            "column_names": list(self.df.columns)  # 记录原始列名（仅供参考）
        }
        
        # 检查缺失值
        self._check_missing_values()
        
        # 检查数据类型
        self._check_dtypes()
        
        return self.df
    
    def get_column_by_index(self, col_idx: int) -> pd.Series:
        """
        通过列索引获取列数据
        
        Args:
            col_idx: 列索引
            
        Returns:
            对应列的Series
        """
        if self.df is None:
            raise ValueError("数据尚未加载！请先调用 load_csv()")
        
        if col_idx < 0 or col_idx >= self.df.shape[1]:
            raise IndexError(f"列索引 {col_idx} 超出范围 [0, {self.df.shape[1]-1}]")
        
        return self.df.iloc[:, col_idx]
    
    def get_columns_by_indices(self, col_indices: list) -> pd.DataFrame:
        """
        通过列索引列表获取多列数据
        
        Args:
            col_indices: 列索引列表
            
        Returns:
            对应列的DataFrame
        """
        if self.df is None:
            raise ValueError("数据尚未加载！请先调用 load_csv()")
        
        return self.df.iloc[:, col_indices]
    
    def _check_missing_values(self):
        """检查缺失值情况"""
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
            logging.warning(f"发现 {len(missing_info)} 列存在缺失值")
            for col, info in missing_info.items():
                logging.warning(f"  {col}: {info['count']} ({info['percent']}%)")
        else:
            logging.info("未发现缺失值")
    
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
        """获取数据加载元信息"""
        return self.meta_info
    
    def print_data_overview(self):
        """打印数据概览"""
        if self.df is None:
            print("数据尚未加载！")
            return
        
        print("=" * 80)
        print("数据概览 (Data Overview)")
        print("=" * 80)
        
        print(f"\n📊 基本信息:")
        print(f"  - 行数: {self.df.shape[0]:,}")
        print(f"  - 列数: {self.df.shape[1]}")
        
        print(f"\n📋 各列信息 (按索引):")
        for i in range(self.df.shape[1]):
            col_name = self.df.columns[i]
            dtype = self.df.iloc[:, i].dtype
            null_count = self.df.iloc[:, i].isnull().sum()
            unique_count = self.df.iloc[:, i].nunique()
            
            print(f"  第{i}列 [{col_name}]: dtype={dtype}, "
                  f"null={null_count}, unique={unique_count}")
        
        print("\n" + "=" * 80)
    
    def get_account_columns(self) -> Tuple[pd.Series, pd.Series]:
        """
        获取账户列（用于构建图）
        
        Returns:
            (支付方账户列, 收款方账户列)
        """
        src_col = self.get_column_by_index(self.config.src_col)
        dst_col = self.get_column_by_index(self.config.dst_col)
        return src_col, dst_col
    
    def get_time_columns(self) -> Tuple[pd.Series, pd.Series]:
        """
        获取时间列
        
        Returns:
            (txn_dt列, tds_dt列)
        """
        col_idx = self.config.col_idx
        txn_dt = self.get_column_by_index(col_idx.txn_dt)
        tds_dt = self.get_column_by_index(col_idx.tds_dt)
        return txn_dt, tds_dt
    
    def get_numerical_columns(self) -> pd.DataFrame:
        """获取数值特征列"""
        return self.get_columns_by_indices(self.config.numerical_cols)
    
    def get_categorical_columns(self) -> pd.DataFrame:
        """获取类别特征列"""
        return self.get_columns_by_indices(self.config.categorical_cols)
