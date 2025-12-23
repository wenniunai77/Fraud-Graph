"""
预处理配置文件
定义数据列索引、路径配置等

字段对应关系（按列索引）- 基于实际数据集：
第0列: uetr - 交易唯一标识 (UUID格式)
第1列: payment_channel - 交易渠道 (NET/BIB/HCN/BFT/PIB等)
第2列: debit_bic_code - 支付方BIC码 (银行标识)
第3列: bene_bic_code - 收款方BIC码 (银行标识)
第4列: evt_tran_stat_cde - 支付状态码 (ACCC等)
第5列: instructed_currency - 客户指定币种 (USD等)
第6列: instructed_amount - 客户指定金额 (数值型)
第7列: payment_currency - 银行使用币种 (USD等)
第8列: payment_amount - 银行使用金额 (数值型)
第9列: credit_currency - 收款方接收币种 (USD等)
第10列: credit_amount - 收款方接收金额 (数值型)
第11列: txn_dt - 事件发生时间 (时间戳格式: 2024-12-25 16:44:03 +00:00)
第12列: tds_dt - 入库时间戳 (时间戳格式: 2024-12-25 17:44:11.921000+00:00)
第13列: mop - 付款方式 (SWIFT/CHAPGB/CHATUS/BOOK/CHIPS等)
第14列: debit_account_masked - 支付方账户(masked)
第15列: bene_account_masked - 收款方账户(masked)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ColumnIndex:
    """
    列索引定义
    所有字段访问使用索引而非列名，确保数据访问的稳定性
    """
    # ========== 标识类字段 ==========
    uetr: int = 0                    # 第0列: 交易唯一标识 (UUID)
    
    # ========== 类别类字段 ==========
    payment_channel: int = 1         # 第1列: 交易渠道 (NET/BIB/HCN/BFT/PIB)
    debit_bic_code: int = 2          # 第2列: 支付方BIC码 (银行标识)
    bene_bic_code: int = 3           # 第3列: 收款方BIC码 (银行标识)
    evt_tran_stat_cde: int = 4       # 第4列: 支付状态码 (ACCC等)
    instructed_currency: int = 5     # 第5列: 客户指定币种 (USD等)
    payment_currency: int = 7        # 第7列: 银行使用币种 (USD等)
    credit_currency: int = 9         # 第9列: 收款方接收币种 (USD等)
    mop: int = 13                    # 第13列: 付款方式 (SWIFT/CHAPGB等)
    
    # ========== 数值类字段 ==========
    instructed_amount: int = 6       # 第6列: 客户指定金额
    payment_amount: int = 8          # 第8列: 银行使用金额
    credit_amount: int = 10          # 第10列: 收款方接收金额
    
    # ========== 时间类字段 ==========
    txn_dt: int = 11                 # 第11列: 事件发生时间
    tds_dt: int = 12                 # 第12列: 入库时间戳
    
    # ========== 账户类字段（用于构建图）==========
    debit_account_masked: int = 14   # 第14列: 支付方账户 (源节点)
    bene_account_masked: int = 15    # 第15列: 收款方账户 (目标节点)


@dataclass
class PreprocessConfig:
    """预处理配置"""
    
    # ========== 路径配置 ==========
    data_path: str = ""                          # 原始CSV数据路径
    output_dir: str = "./preprocessed_data"      # 预处理输出目录
    
    # ========== 列索引配置 ==========
    col_idx: ColumnIndex = field(default_factory=ColumnIndex)
    
    # ========== 图构建配置 ==========
    src_col: int = 14                 # 源节点列（支付方账户）
    dst_col: int = 15                 # 目标节点列（收款方账户）
    
    # ========== 特征列分类 ==========
    # 数值特征列索引（金额相关）
    numerical_cols: List[int] = field(default_factory=lambda: [6, 8, 10])
    
    # 类别特征列索引
    categorical_cols: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 9, 13])
    
    # 时间特征列索引
    time_cols: List[int] = field(default_factory=lambda: [11, 12])
    
    # ========== 采样配置 ==========
    use_full_dataset: bool = False
    sample_size: int = 500000
    random_seed: int = 42
    
    # ========== 输出文件名 ==========
    graph_data_file: str = "graph_data.pt"           # PyG图数据文件
    node_features_file: str = "node_features.pt"     # 节点特征文件
    edge_features_file: str = "edge_features.pt"     # 边特征文件
    edge_index_file: str = "edge_index.pt"           # 边索引文件
    node_mapping_file: str = "node_mapping.pkl"      # 节点映射文件
    statistics_file: str = "statistics.json"         # 统计信息文件
    preprocess_meta_file: str = "preprocess_meta.json"  # 预处理元信息（用于检查）
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件的完整路径"""
        return os.path.join(self.output_dir, filename)
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# 特征含义说明（用于代码注释和文档）
# ============================================================================

NODE_FEATURE_DESCRIPTION = """
节点级特征说明（聚合自边特征）:

【作为发送方(源节点)时的聚合特征】
- src_avg_instructed_amount: 作为发送方时，平均客户指定金额
- src_avg_payment_amount: 作为发送方时，平均银行使用金额
- src_avg_credit_amount: 作为发送方时，平均收款方接收金额
- src_payment_channel_mode: 作为发送方时，最常用的交易渠道(编码后)
- src_debit_bic_mode: 作为发送方时，最常用的支付方BIC码(编码后)
- src_bene_bic_mode: 作为发送方时，最常用的收款方BIC码(编码后)
- src_currency_mode: 作为发送方时，最常用的币种(编码后)
- src_mop_mode: 作为发送方时，最常用的付款方式(编码后)
- src_avg_hour: 作为发送方时，平均交易小时
- src_avg_day_of_week: 作为发送方时，平均交易星期几
- src_tx_count: 作为发送方的交易次数（出度）

【作为接收方(目标节点)时的聚合特征】
- dst_avg_instructed_amount: 作为接收方时，平均客户指定金额
- dst_avg_payment_amount: 作为接收方时，平均银行使用金额
- dst_avg_credit_amount: 作为接收方时，平均收款方接收金额
- dst_payment_channel_mode: 作为接收方时，最常用的交易渠道(编码后)
- dst_debit_bic_mode: 作为接收方时，最常用的支付方BIC码(编码后)
- dst_bene_bic_mode: 作为接收方时，最常用的收款方BIC码(编码后)
- dst_currency_mode: 作为接收方时，最常用的币种(编码后)
- dst_mop_mode: 作为接收方时，最常用的付款方式(编码后)
- dst_avg_hour: 作为接收方时，平均交易小时
- dst_avg_day_of_week: 作为接收方时，平均交易星期几
- dst_tx_count: 作为接收方的交易次数（入度）

【统计特征】
- total_degree: 总交易次数（出度+入度）
- in_out_ratio: 入度/出度比值（体现资金流向特征）
"""

EDGE_FEATURE_DESCRIPTION = """
边级特征说明:

【数值特征】
- instructed_amount: 客户指定金额（第6列）
- payment_amount: 银行使用金额（第8列）
- credit_amount: 收款方接收金额（第10列）

【类别特征（编码后）】
- payment_channel_encoded: 交易渠道编码（第1列）
- debit_bic_code_encoded: 支付方BIC码编码（第2列）
- bene_bic_code_encoded: 收款方BIC码编码（第3列）
- evt_tran_stat_cde_encoded: 支付状态码编码（第4列）
- instructed_currency_encoded: 客户指定币种编码（第5列）
- payment_currency_encoded: 银行使用币种编码（第7列）
- credit_currency_encoded: 收款方接收币种编码（第9列）
- mop_encoded: 付款方式编码（第13列）

【时间特征】
- txn_hour: 交易发生小时（0-23）
- txn_day_of_week: 交易发生星期几（0-6）
- txn_day_of_month: 交易发生日期（1-31）
- tds_hour: 入库小时
- tds_day_of_week: 入库星期几
- tds_day_of_month: 入库日期

【时间差特征 - 新增】
- time_diff_seconds: tds_dt - txn_dt 的时间差（秒）
  含义：从事件发生到入库的延迟时间
  用途：异常交易可能有异常的处理延迟
"""
