# GraphMAE 欺诈检测项目

基于 GraphMAE (Graph Masked Autoencoder) 的支付交易欺诈检测系统。

## 项目结构

项目分为两大部分：

```
graph_main/
├── preprocess/                    # 【Part 1】数据预处理模块
│   ├── config.py                  # 预处理配置
│   ├── data_loader.py             # 数据加载
│   ├── feature_engineer.py        # 特征工程（包含time_diff特征）
│   ├── graph_builder.py           # 图构建
│   ├── statistics.py              # 统计分析
│   ├── run_preprocess.py          # 预处理主脚本
│   └── preprocess_check.ipynb     # 预处理检查notebook
│
├── config.py                      # 【Part 2】主程序配置
├── models/                        # 模型定义
│   ├── graphmae.py               # GraphMAE模型
│   ├── encoder.py                # 编码器（GAT/GCN）
│   └── loss_func.py              # 损失函数（SCE）
├── trainer.py                     # 训练器
├── anomaly_detector.py           # 异常检测器
├── visualization.py              # 可视化
└── run_main.py                   # 主程序脚本
```

## 运行流程

### 必须按顺序运行：

**Step 1: 数据预处理**
```bash
cd graph_main/preprocess
python run_preprocess.py \
    --data_path /path/to/your/data.csv \
    --output_dir ./preprocessed_data \
    --sample_size 500000
```

**Step 2: 检查预处理结果（可选但推荐）**

在 Jupyter 中打开 `preprocess/preprocess_check.ipynb`，运行所有单元格检查：
- fillna 情况
- time_diff 特征是否成功构造
- 数据质量检查

**Step 3: 运行主程序**
```bash
cd graph_main
python run_main.py \
    --preprocessed_dir ./preprocess/preprocessed_data \
    --output_dir ./output \
    --epochs 500 \
    --device 0
```

## 特征说明

### 边特征 (Edge Features)

| 特征名 | 来源列 | 含义 |
|--------|--------|------|
| instructed_amount | 第6列 | 客户指定金额 |
| payment_amount | 第8列 | 银行使用金额 |
| credit_amount | 第10列 | 收款方接收金额 |
| payment_channel_encoded | 第1列 | 交易渠道编码 |
| debit_bic_code_encoded | 第2列 | 支付方BIC码编码 |
| bene_bic_code_encoded | 第3列 | 收款方BIC码编码 |
| evt_tran_stat_cde_encoded | 第4列 | 支付状态码编码 |
| instructed_currency_encoded | 第5列 | 客户指定币种编码 |
| payment_currency_encoded | 第7列 | 银行使用币种编码 |
| credit_currency_encoded | 第9列 | 收款方接收币种编码 |
| mop_encoded | 第13列 | 付款方式编码 |
| txn_hour | 第11列 | 交易发生小时 |
| txn_day_of_week | 第11列 | 交易发生星期几 |
| txn_day_of_month | 第11列 | 交易发生日期 |
| tds_hour | 第12列 | 入库小时 |
| tds_day_of_week | 第12列 | 入库星期几 |
| tds_day_of_month | 第12列 | 入库日期 |
| **time_diff_seconds** | 第12-11列 | **时间差 (tds_dt - txn_dt)** |

### 节点特征 (Node Features)

| 特征名 | 含义 |
|--------|------|
| src_avg_feat_* | 作为发送方时，各边特征的平均值 |
| dst_avg_feat_* | 作为接收方时，各边特征的平均值 |
| src_tx_count | 出度：作为发送方的交易次数 |
| dst_tx_count | 入度：作为接收方的交易次数 |
| total_degree | 总度数 |
| in_out_ratio | 入出度比值 |

## 输出文件

### 预处理输出 (`preprocess/preprocessed_data/`)
- `graph_data.pt`: PyG图数据对象
- `node_features.pt`: 节点特征
- `edge_features.pt`: 边特征
- `edge_index.pt`: 边索引
- `node_mapping.pkl`: 节点映射
- `statistics.json`: 统计信息
- `preprocess_meta.json`: 预处理元信息（用于检查）

### 主程序输出 (`output/`)
- `graphmae_model.pt`: 训练好的模型
- `anomaly_results.json`: 异常检测结果
- `comprehensive_report.png`: 综合可视化报告
- `embeddings_tsne.png`: t-SNE可视化

## 依赖安装

```bash
pip install torch torch_geometric torch_scatter
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 关键参数说明

### 预处理参数
- `--sample_size`: 采样大小，0表示全量数据
- `--src_col`: 源节点列索引（默认14，支付方账户）
- `--dst_col`: 目标节点列索引（默认15，收款方账户）

### 模型参数
- `--encoder_type`: 编码器类型 (gat/gcn)
- `--hidden_channels`: 隐藏层维度
- `--num_layers`: GNN层数
- `--mask_rate`: 掩码比例
- `--epochs`: 训练轮数
- `--patience`: 早停耐心值

## 检查点

预处理完成后，务必运行 `preprocess_check.ipynb` 检查：

1. ✅ fillna 情况：哪些列被填充了缺失值
2. ✅ time_diff 构造：时间差特征是否成功计算
3. ✅ 数据质量：是否有NaN、Inf等异常值
4. ✅ 特征分布：特征分布是否合理
