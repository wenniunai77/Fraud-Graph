
| 字段                   | 说明                                               | 样例                                                                                                                     |
| -------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| uetr                 | 一笔支付的唯一标识                                        | ffffdad6-a050-40c4-a8d4-80d70184ec4f                                                                                   |
| payment_channel      | 交易渠道                                             | NET  486880 - large customers<br>BIB  268393 - medium and small customers<br>HCN  259245<br>BFT  160337<br>PIB  125945 |
| debit_bic_code       | 支付方的BIC码(银行标识)                                   | HBUKGB4B                                                                                                               |
| bene_bic_code        | 收款方的BIC码(银行标识)                                   | TACBHK                                                                                                                 |
| evt_tran_stat_cde    | 事件级:支付状态码                                        | ACCC  2608869                                                                                                          |
| instructed_currency  | 客户指定的交易币种                                        | USD                                                                                                                    |
| instructed_amount    | 客户指定的交易金额                                        | 20000.0                                                                                                                |
| payment_currency     | 银行使用的交易币种                                        | USD                                                                                                                    |
| payment_amount       | 银行使用的交易金额                                        | 20000.0                                                                                                                |
| credit_currency      | 事件级:收款方接收的币种(confirmedcurrency)                  | USD                                                                                                                    |
| credit_amount        | 事件级:收款方接收的金额(confirmed amount)                   | 20000.0                                                                                                                |
| txn_dt               | 事件级:事件发生的时间戳                                     | 2024-12-25 16:44:03 +00:00                                                                                             |
| tds_dt               | 事件级:入库的时间戳                                       | 2024-12-25 17:44:11.921000+00:00                                                                                       |
| mop                  | 事件级:付款方式                                         | SWIFT  612192<br>CHAPGB  298732<br>CHATUS  136071<br>BOOK  72898<br>CHIPS  65856<br>SWT  43230                         |
| debit_account_masked | masked之后的支付方account number，以序列号代替原account number | 1                                                                                                                      |
| bene_account_masked  | masked之后的收款方account number,以序列号代替原account number | 724353                                                                                                                 |
