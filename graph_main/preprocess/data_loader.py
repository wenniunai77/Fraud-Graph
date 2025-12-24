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
        path = file_path or self.config.data_path
        
        if not path:
            raise ValueError("Data path not specified! Please set config.data_path or pass file_path parameter")
        
        logging.info(f"Loading data: {path}")
        
        np.random.seed(self.config.random_seed)
        
        if self.config.use_full_dataset:
            logging.info("Loading full dataset...")
            self.df = pd.read_csv(path, header=0)
            logging.info(f"Full dataset loaded. Shape: {self.df.shape}")
        else:
            logging.info(f"Loading sampled dataset ({self.config.sample_size} rows)...")
            self.df = pd.read_csv(path, header=0, nrows=self.config.sample_size)
            logging.info(f"Sampled data loaded. Shape: {self.df.shape}")
        
        self.raw_shape = self.df.shape
        
        self.meta_info["load_info"] = {
            "file_path": path,
            "raw_rows": self.raw_shape[0],
            "raw_cols": self.raw_shape[1],
            "use_full_dataset": self.config.use_full_dataset,
            "sample_size": self.config.sample_size if not self.config.use_full_dataset else None,
            "column_names": list(self.df.columns)
        }
        
        self._check_missing_values()
        self._check_dtypes()
        
        return self.df
    
    def get_column_by_index(self, col_idx: int) -> pd.Series:
        if self.df is None:
            raise ValueError("Data not loaded! Please call load_csv() first")
        
        if col_idx < 0 or col_idx >= self.df.shape[1]:
            raise IndexError(f"Column index {col_idx} out of range [0, {self.df.shape[1]-1}]")
        
        return self.df.iloc[:, col_idx]
    
    def get_columns_by_indices(self, col_indices: list) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded! Please call load_csv() first")
        
        return self.df.iloc[:, col_indices]
    
    def _check_missing_values(self):
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
            logging.warning(f"Found {len(missing_info)} columns with missing values")
            for col, info in missing_info.items():
                logging.warning(f"  {col}: {info['count']} ({info['percent']}%)")
        else:
            logging.info("No missing values found")
    
    def _check_dtypes(self):
        if self.df is None:
            return
        
        dtype_info = {}
        for i, dtype in enumerate(self.df.dtypes):
            col_name = self.df.columns[i] if i < len(self.df.columns) else f"col_{i}"
            dtype_info[f"col_{i}_{col_name}"] = str(dtype)
        
        self.meta_info["dtype_info"] = dtype_info
        logging.info("Data type check completed")
    
    def get_meta_info(self) -> Dict[str, Any]:
        return self.meta_info
    
    def print_data_overview(self):
        if self.df is None:
            print("Data not loaded!")
            return
        
        print("=" * 80)
        print("Data Overview")
        print("=" * 80)
        
        print(f"\nBasic Info:")
        print(f"  - Rows: {self.df.shape[0]:,}")
        print(f"  - Columns: {self.df.shape[1]}")
        
        print(f"\nColumn Info (by index):")
        for i in range(self.df.shape[1]):
            col_name = self.df.columns[i]
            dtype = self.df.iloc[:, i].dtype
            null_count = self.df.iloc[:, i].isnull().sum()
            unique_count = self.df.iloc[:, i].nunique()
            
            print(f"  Col {i} [{col_name}]: dtype={dtype}, "
                  f"null={null_count}, unique={unique_count}")
        
        print("\n" + "=" * 80)
    
    def get_account_columns(self) -> Tuple[pd.Series, pd.Series]:
        src_col = self.get_column_by_index(self.config.src_col)
        dst_col = self.get_column_by_index(self.config.dst_col)
        return src_col, dst_col
    
    def get_time_columns(self) -> Tuple[pd.Series, pd.Series]:
        col_idx = self.config.col_idx
        txn_dt = self.get_column_by_index(col_idx.txn_dt)
        tds_dt = self.get_column_by_index(col_idx.tds_dt)
        return txn_dt, tds_dt
    
    def get_numerical_columns(self) -> pd.DataFrame:
        return self.get_columns_by_indices(self.config.numerical_cols)
    
    def get_categorical_columns(self) -> pd.DataFrame:
        return self.get_columns_by_indices(self.config.categorical_cols)
