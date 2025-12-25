import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ColumnIndex:
    uetr: int = 0
    payment_channel: int = 1
    debit_bic_code: int = 2
    bene_bic_code: int = 3
    evt_tran_stat_cde: int = 4
    instructed_currency: int = 5
    payment_currency: int = 7
    credit_currency: int = 9
    mop: int = 13
    instructed_amount: int = 6
    payment_amount: int = 8
    credit_amount: int = 10
    txn_dt: int = 11
    tds_dt: int = 12
    debit_account_masked: int = 14
    bene_account_masked: int = 15


@dataclass
class PreprocessConfig:
    data_path: str = ""
    output_dir: str = "../processed_data"
    col_idx: ColumnIndex = field(default_factory=ColumnIndex)
    src_col: int = 14
    dst_col: int = 15
    numerical_cols: List[int] = field(default_factory=lambda: [6, 8, 10])
    categorical_cols: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 9, 13])
    time_cols: List[int] = field(default_factory=lambda: [11, 12])
    embedding_dim: int = 8
    use_full_dataset: bool = True
    sample_size: int = 500000
    random_seed: int = 42
    graph_data_file: str = "graph_data.pt"
    node_features_file: str = "node_features.pt"
    edge_features_file: str = "edge_features.pt"
    edge_index_file: str = "edge_index.pt"
    node_mapping_file: str = "node_mapping.pkl"
    statistics_file: str = "statistics.json"
    preprocess_meta_file: str = "preprocess_meta.json"
    
    def get_output_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)
    
    def ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
