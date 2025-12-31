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
├── README.md                    # 项目说明
├── run_preprocess.py           # 预处理脚本（数据加载+特征工程+图构建）
├── run_training.py             # 训练脚本（模型训练+融合+评估）
├── utils.py                     # 工具函数
│
├── configs/                     # 配置模块
│   ├── __init__.py
│   ├── base_config.py          # 基础配置（列索引）
│   ├── preprocess_config.py    # 预处理配置
│   └── training_config.py      # 训练配置
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
│   └── graph_model.py          # 图模型（GraphMAE封装）
│
├── fusion/                      # 融合策略
│   ├── __init__.py
│   └── fusion.py               # 融合方法实现
│
├── evaluation/                  # 无标签评估
│   ├── __init__.py
│   └── unsupervised_eval.py    # 稳定性/弱规则/分数分布
│
├── visualization/               # 可视化模块（4大主题）
│   ├── __init__.py             # 模块导出
│   ├── utils.py                # 样式设置和工具函数
│   ├── model_performance.py    # 模型性能可视化
│   ├── fusion_analysis.py      # 融合分析可视化
│   ├── feature_contribution.py # 特征贡献可视化
│   ├── anomaly_distribution.py # 异常分布可视化
│   └── dashboard.py            # 综合报告仪表板
│
├── processed_data/              # 预处理输出（图数据、特征等）
├── output/                      # 训练输出（分数、模型等）
└── checkpoints/                 # 模型检查点
```

## 运行流程（推荐：分步执行）

### 步骤 1: 预处理

预处理阶段负责数据加载、特征工程和图构建，输出预处理完成的数据。

```bash
cd "Graph/Fusion Model"
python run_preprocess.py --data ../graph_main/raw_data/xxx.csv --output ./processed_data
```

**预处理参数说明：**
```bash
python run_preprocess.py \
    --data ../graph_main/raw_data/xxx.csv \  # 输入数据路径
    --output ./processed_data \               # 输出目录
    --sample-size 500000 \                    # 采样大小（可选）
    --seed 42 \                               # 随机种子
    --no-self-loops                           # 不添加自环（可选）
```

**预处理输出文件：**
```
processed_data/
├── graph_data.pt               # 完整图数据（PyG Data对象）
├── tabular_features.npy        # 表格特征矩阵
├── raw_data.pkl                # 原始数据副本
├── node_mapping.pkl            # 节点ID映射
├── feature_engineer.pkl        # 特征工程器（用于推理）
├── preprocess_meta.json        # 元信息
├── node_features.pt            # 节点特征
├── edge_index.pt               # 边索引
├── original_edge_index.pt      # 原始边索引（不含自环）
└── edge_features.pt            # 边特征
```

### 步骤 2: 模型训练

训练阶段读取预处理结果，训练图模型和表格模型，进行融合和评估。

```bash
python run_training.py --processed-data ./processed_data --output ./output
```

**训练参数说明：**
```bash
python run_training.py \
    --processed-data ./processed_data \  # 预处理数据目录
    --output ./output \                  # 输出目录
    --strategy gated \                   # 融合策略
    --epochs 100 \                       # 训练轮数
    --top-k 1000 \                       # Top-K异常数
    --device 0 \                         # GPU设备号（-1表示CPU）
    --no-visualize                       # 禁用可视化（可选）
```

**训练输出文件：**
```
output/
├── fusion_scores.csv           # 融合分数
├── top_1000_anomalies.csv      # Top-K异常交易
├── tabular_model.pkl           # 表格模型
├── graph_model.pt              # 图模型
├── training_config.json        # 训练配置
└── visualizations/             # 可视化结果（4大主题）
    ├── training_curves.png         # 训练曲线
    ├── model_comparison.png        # 模型对比
    ├── score_statistics.png        # 分数统计
    ├── fusion_overview.png         # 融合概览
    ├── fusion_weights_distribution.png # 融合权重分布
    ├── model_agreement.png         # 模型一致性
    ├── feature_importance.png      # 特征重要性
    ├── model_contribution.png      # 模型贡献
    ├── score_distributions.png     # 分数分布
    ├── anomaly_scatter.png         # 异常散点图
    ├── topk_analysis.png           # Top-K分析
    └── comprehensive_report.png    # 综合报告
```

## 可视化主题说明

可视化模块按4大主题组织，便于理解不同维度的分析结果：

### 1. 模型性能（model_performance）
- **训练曲线**：展示训练过程中损失的变化
- **模型对比**：对比不同模型的分数分布和表现
- **分数统计**：统计各模型分数的分位数和分布特征

### 2. 融合分析（fusion_analysis）
- **融合概览**：展示融合前后分数的关系
- **融合权重分布**：分析门控融合中权重的分布情况
- **模型一致性**：分析两模型对异常判断的一致性程度

### 3. 特征贡献（feature_contribution）
- **特征重要性**：展示各特征对异常分数的贡献
- **模型贡献**：分析图模型vs表格模型在不同场景下的贡献比例

### 4. 异常分布（anomaly_distribution）
- **分数分布**：对比图/表格/融合三种分数的分布
- **异常散点图**：二维展示图分数vs表格分数的关系
- **Top-K分析**：分析不同K值下模型的重叠率和一致性

## 一键运行（向后兼容）

如果希望一次性完成预处理和训练，可以使用原有的 `run_fusion.py`：

```bash
python run_fusion.py --data ../graph_main/raw_data/xxx.csv --output ./output
```

## 安装依赖

```bash
pip install torch torch_geometric torch_scatter
pip install pandas numpy scikit-learn matplotlib seaborn
pip install networkx scipy
```

## 数据格式要求

数据格式要求（列顺序）：
- Col 0: uetr（交易ID）
- Col 1: payment_channel（支付渠道）
- Col 2-3: debit_bic_code, bene_bic_code（BIC代码）
- Col 5-10: 币种和金额字段
- Col 11-12: txn_dt, tds_dt（时间字段）
- Col 13: mop（支付方式）
- Col 14-15: debit_account_masked, bene_account_masked（账户）

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
