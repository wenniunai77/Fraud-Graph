"""
基础配置 - 数据列索引
"""
from dataclasses import dataclass


@dataclass
class ColumnIndex:
    """数据列索引配置"""
    uetr: int = 0
    payment_channel: int = 1
    debit_bic_code: int = 2
    bene_bic_code: int = 3
    evt_tran_stat_cde: int = 4
    instructed_currency: int = 5
    instructed_amount: int = 6
    payment_currency: int = 7
    payment_amount: int = 8
    credit_currency: int = 9
    credit_amount: int = 10
    txn_dt: int = 11
    tds_dt: int = 12
    mop: int = 13
    debit_account_masked: int = 14
    bene_account_masked: int = 15
