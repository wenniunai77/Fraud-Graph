# Fusion Model: 图模型 + 表格模型融合欺诈检测

基于 **Fusion.md** 方案实现的无监督/自监督欺诈检测融合框架。

## 核心思路

在无标签场景下，单一模型难以可靠评估；本项目融合两类方法：

1. **图模型（GraphMAE）**：利用账户交易图的结构信息，通过自监督重构检测结构异常
2. **表格模型（IsolationForest/LOF/AutoEncoder）**：利用交易级特征，检测特征空间的异常点

**融合策略**：根据节点活跃度自适应调整权重——活跃账户更信任图模型，冷启动账户更信任表格模型。

## 项目结构

```
Fusion Model/
├── config.py                    # 配置文件
├── README.md                    # 项目说明
├── run_fusion.py               # 主运行脚本
│
├── preprocess/                  # 数据预处理
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载
│   ├── feature_engineer.py     # 特征工程
│   └── graph_builder.py        # 图构建
│
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── tabular.py              # 表格无监督模型
│   ├── graph_model.py          # 图模型（GraphMAE封装）
│   └── graphmae/               # GraphMAE核心实现
│       ├── __init__.py
│       ├── encoder.py
│       ├── graphmae.py
│       ├── loss_func.py
│       └── utils.py
│
├── fusion/                      # 融合策略
│   ├── __init__.py
│   └── fusion.py               # 融合方法实现
│
├── evaluation/                  # 无标签评估
│   ├── __init__.py
│   └── unsupervised_eval.py    # 稳定性/弱规则/分数分布
│
├── output/                      # 输出目录
│   ├── processed_data/         # 预处理数据
│   ├── scores/                 # 异常分数
│   └── reports/                # 评估报告
│
└── checkpoints/                 # 模型检查点
```

## 运行流程

### 1. 安装依赖

```bash
pip install torch torch_geometric torch_scatter
pip install pandas numpy scikit-learn matplotlib seaborn
pip install networkx scipy
```

### 2. 准备数据

将你的交易数据CSV放到 `../graph_main/raw_data/` 目录，或修改 `config.py` 中的 `data_path`。

数据格式要求（列顺序）：
- Col 0: uetr（交易ID）
- Col 1: payment_channel（支付渠道）
- Col 2-3: debit_bic_code, bene_bic_code（BIC代码）
- Col 5-10: 币种和金额字段
- Col 11-12: txn_dt, tds_dt（时间字段）
- Col 13: mop（支付方式）
- Col 14-15: debit_account_masked, bene_account_masked（账户）

### 3. 运行融合检测

```bash
cd "Graph/Fusion Model"
python run_fusion.py --data_path ../graph_main/raw_data/xxx.csv --epochs 300
```

### 4. 主要参数

```bash
python run_fusion.py \
    --data_path ../graph_main/raw_data/xxx.csv \
    --output_dir ./output \
    --epochs 300 \
    --fusion_method gated \
    --tabular_model ensemble \
    --device 0
```

## 融合策略说明

### 1. 门控融合（Gated Fusion）—— 默认推荐

根据节点活跃度（degree）自适应调整权重：

```
if min(deg(src), deg(dst)) < threshold:
    final_score = α_low * graph_score + (1 - α_low) * tab_score
else:
    final_score = α_high * graph_score + (1 - α_high) * tab_score
```

- 高活跃度账户：更信任图模型（α_high = 0.7）
- 冷启动账户：更信任表格模型（α_low = 0.3）

### 2. 加权融合（Weighted Fusion）

固定权重线性组合：
```
final_score = w * graph_score + (1 - w) * tab_score
```

### 3. 排名融合（Rank Fusion）

将两个模型的分数转为排名后平均：
```
final_rank = (rank_graph + rank_tab) / 2
```

### 4. 一致性融合（Consistent Fusion）

两个模型都认为高风险才报警（高精度）：
```
is_anomaly = (graph_score > threshold_graph) AND (tab_score > threshold_tab)
```

## 无标签评估指标

由于没有真实标签，本项目提供以下评估维度：

### 1. 稳定性评估
- 多随机种子训练，计算 Top-K 重叠率（Jaccard@K）
- 排名相关性（Spearman）

### 2. 弱规则命中率
- 定义简单规则（极端大额、极端时延等）
- 评估 Top-K 中规则命中比例

### 3. 分数分布分析
- 分位数统计（90/95/99）
- 分数分布可视化

### 4. 融合增益分析
- 对比单一模型 vs 融合模型的稳定性和规则命中

## 输出文件

运行完成后，`output/` 目录包含：

- `processed_data/`: 预处理后的图数据和特征
- `scores/transaction_scores.csv`: 交易级异常分数（含 tab_score, graph_score, fusion_score）
- `scores/account_scores.csv`: 账户级风险分数
- `reports/evaluation_report.json`: 无标签评估报告
- `reports/top_k_transactions.csv`: Top-K 可疑交易清单
- `reports/visualization.png`: 可视化报告

## 与 graph_main 的关系

本项目复用 `graph_main` 的：
- 数据预处理逻辑（ColumnIndex、特征工程）
- GraphMAE 模型实现
- 异常分数计算逻辑

新增：
- 表格无监督模型（IsolationForest/LOF/AutoEncoder）
- 融合策略（门控/加权/排名/一致性）
- 无标签评估框架（稳定性/弱规则/分数分布）

## 引用

- GraphMAE: Self-Supervised Masked Graph Autoencoders
- Fusion.md 中的方案设计
