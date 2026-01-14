# Fusion Model: 图+表格融合欺诈检测

> 无监督/自监督欺诈检测框架，融合图结构信息与交易特征

## 📋 目录

- [核心思想](#核心思想)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [功能特性](#功能特性)
- [运行流程](#运行流程)
- [配置说明](#配置说明)
- [输出结果](#输出结果)
- [常见问题](#常见问题)

---

## 核心思想

在无标签场景下，融合两种互补的异常检测方法：

| 模型类型 | 利用信息 | 检测能力 | 适用场景 |
|---------|---------|---------|---------|
| **图模型** (GraphMAE) | 交易图结构 | 结构异常、社区异常 | 活跃账户、关联交易 |
| **表格模型** (Ensemble) | 交易特征 | 特征异常、孤立点 | 冷启动账户、单笔异常 |

**融合策略**：根据节点活跃度（度数）自适应调整权重
- 高活跃账户 → 更信任图模型
- 低活跃账户 → 更信任表格模型

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch torch_geometric
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. 数据预处理

```bash
cd "Graph/Fusion Model"
python run_preprocess.py --data ../graph_main/raw_data/xxx.csv --output ./processed_data
```

**首次运行会自动训练 Embedding**（Masked Attribute Modeling），需要额外 1-10 分钟

### 3. 模型训练

#### 单 Seed 模式（默认）

```bash
python run_training.py --processed-data ./processed_data --output ./output
```

#### 多 Seed 模式（稳定性验证）

修改 `configs/training_config.py`：

```python
@dataclass
class TrainingMainConfig:
    # ...
    enable_multi_seed: bool = True  # 启用多 seed 模式
    seeds: List[int] = [42, 123, 456, 789, 2024]  # 配置多个种子
```

然后运行训练：

```bash
python run_training.py --processed-data ./processed_data --output ./output_multi_seed
```

**多 Seed 模式会自动生成：**
- 每个 seed 的独立结果 → `output/seed_42/`, `output/seed_123/`, ...
- Jaccard 相似度矩阵 → `output/jaccard_similarity.csv`
- 稳定异常节点列表 → `output/stable_anomalies.csv`（所有 seed 交集）
- 汇总报告 → `output/multi_seed_summary.json`

### 4. 可视化分析

```bash
python run_visualization.py --results ./output --output ./output/visualizations
```

---

## 项目结构

```
Fusion Model/
├── README.md                       # 本文档
├── BUGFIX_P3.md                   # 修复记录 (OOV映射/除零/对齐检查)
├── REMOVAL_WEAK_RULES.md          # 功能移除记录 (弱规则)
├── MULTI_SEED_GUIDE.md            # 🔥 多 Seed 模式完整指南
├── MULTI_SEED_EXAMPLES.md         # 🔥 多 Seed 配置示例
│
├── run_preprocess.py              # 预处理脚本
├── run_training.py                # 训练脚本
├── run_visualization.py           # 可视化脚本
├── utils.py                       # 工具函数
│
├── configs/                       # 配置模块
│   ├── base_config.py            # 基础配置 (列索引)
│   ├── preprocess_config.py      # 预处理配置
│   ├── embedding_config.py       # Embedding 预训练配置
│   └── training_config.py        # 训练/融合/评估配置
│
├── preprocess/                    # 数据预处理
│   ├── data_loader.py            # 数据加载
│   ├── feature_engineer.py       # 特征工程 (含预训练embedding)
│   ├── graph_builder.py          # 图构建
│   └── train_embeddings.py       # Embedding 预训练 (MAM)
│
├── models/                        # 模型定义
│   ├── graph_model.py            # GraphMAE (GAT encoder + MLP decoder)
│   └── tabular.py                # 表格模型 (IF/LOF/AE ensemble)
│
├── fusion/                        # 融合策略
│   └── fusion.py                 # 门控/加权/排名/一致性融合
│
├── evaluation/                    # 无标签评估
│   └── unsupervised_eval.py      # 稳定性评估 + 分数分布分析
│
├── visualization/                 # 可视化模块
│   ├── model_performance.py      # 模型性能可视化
│   ├── fusion_analysis.py        # 融合分析可视化
│   ├── feature_contribution.py   # 特征贡献可视化
│   ├── anomaly_distribution.py   # 异常分布可视化
│   └── dashboard.py              # 综合报告
│
├── processed_data/                # 预处理输出
│   ├── graph_data.pt             # 图数据 (PyG Data)
│   ├── tabular_features.npy      # 表格特征
│   ├── pretrained_embeddings.pt  # 预训练 embedding 权重
│   └── *.pkl                     # 元信息/映射
│
└── output/                        # 训练输出
    ├── fusion_scores.csv         # 融合分数
    ├── top_1000_anomalies.csv    # Top-K 异常
    ├── training_results.pkl      # 完整结果 (可视化用)
    └── visualizations/           # 可视化图表
```

---

## 功能特性

### ✨ 核心功能

1. **类别特征预训练** (Masked Attribute Modeling)
   - 类别特征不再随机初始化，通过无监督任务学习语义
   - 相似类别值获得相似 embedding
   - 支持自适应维度：`dim = num_categories^0.25`

2. **图模型** (GraphMAE)
   - Encoder: GAT (Graph Attention Network)
   - Decoder: MLP (快速) 或 GAT (精确)
   - Loss: SCE (Scaled Cosine Error)
   - 支持 DropEdge 数据增强

3. **表格模型** (Ensemble)
   - Isolation Forest: 检测全局孤立点
   - Local Outlier Factor: 检测局部密度异常
   - AutoEncoder: 检测重构误差异常
   - 集成投票提高鲁棒性

4. **自适应融合**
   - **门控融合** (推荐): 根据节点度数动态调权
   - **加权融合**: 固定权重线性组合
   - **排名融合**: 基于排名的融合
   - **一致性融合**: 两模型都判异常才报警

5. **无标签评估**
   - 稳定性评估: 多随机种子 Jaccard@K 重叠率
   - 分数分布分析: 统计量/百分位数/尾部权重
   - 可视化分析: 12+ 图表全面展示

### 🔧 最近修复 (P3)

1. **OOV 映射修复**: 未见类别不再错误映射到索引 0，使用 UNKNOWN 或均值 embedding
2. **除零/NaN 保护**: skewness/kurtosis/tail_weight 的兜底处理
3. **对齐检查强化**: 图侧/表侧分数长度不一致时报错，不再悄悄截断

详见 `BUGFIX_P3.md`

---

## 运行流程

### 步骤 1: 预处理

```bash
python run_preprocess.py \
    --data <输入CSV路径> \
    --output ./processed_data \
    --seed 42 \
    --no-self-loops  # 可选：不添加自环
```

**输出文件**:
- `graph_data.pt` - 完整图数据
- `tabular_features.npy` - 表格特征矩阵
- `pretrained_embeddings.pt` - 预训练 embedding
- `*.pkl` - 映射和元信息

**时间**: 首次运行需额外 1-10 分钟训练 embedding

---

### 步骤 2: 训练

```bash
python run_training.py \
    --processed-data ./processed_data \
    --output ./output \
    --strategy gated \      # 融合策略: gated/weighted/rank/consistent
    --epochs 100 \
    --top-k 1000 \
    --device 0              # GPU 设备号 (-1=CPU)
```

**输出文件**:
- `fusion_scores.csv` - **所有边**的融合分数
  - `graph_score`: 图模型分数
  - `tabular_score`: 表格模型分数
  - `fused_score`: 融合后分数
  - `fusion_weight`: 融合权重 α (门控融合时有值)
  
- `top_1000_anomalies.csv` - **Top-K 异常交易**（包含完整交易信息）
  - 原始数据所有字段（含 `debit_account_masked`, `bene_account_masked` 等）
  - `rank`: 异常排名 (1~K)
  - `fused_score`: 融合分数
  - `graph_score`: 图模型分数
  - `tabular_score`: 表格模型分数
  - `fusion_weight`: 融合权重 α (门控融合时)
  - `node_degree_proxy`: 活跃度代理 = min(src_out_degree, dst_in_degree)
    - **用于解释为什么更信图/表**: 度数越高 → α 越大 → 更信任图模型
  
- `training_results.pkl` - 完整结果 (可视化用)
- `training_report.txt` - 训练报告
- `graph_model.pt` / `tabular_model.pkl` - 模型权重 (可选)

**时间**: 10-30 分钟 (取决于数据量和 GPU)

---

### 步骤 3: 可视化

```bash
python run_visualization.py \
    --results ./output \
    --output ./output/visualizations \
    --top-k 1000
```

**生成图表** (12 张):
1. **模型性能**: 训练曲线、模型对比、分数统计
2. **融合分析**: 融合概览、权重分布、模型一致性
3. **特征贡献**: 特征重要性、模型贡献
4. **异常分布**: 分数分布、异常散点图、Top-K 分析
5. **综合报告**: 4x3 拼接大图

---

## 配置说明

### 多 Seed 稳定性验证

修改 `configs/training_config.py` 启用多 seed 模式：

```python
@dataclass
class TrainingMainConfig:
    # ...
    enable_multi_seed: bool = True  # 启用多 seed 模式
    seeds: List[int] = [42, 123, 456, 789, 2024]  # 配置多个种子
```

**多 Seed 输出结构**：
```
output_multi_seed/
├── seed_42/                      # 每个 seed 独立结果
│   ├── fusion_scores.csv
│   ├── top_1000_anomalies.csv
│   └── training_results.pkl
├── seed_123/
├── seed_456/
├── ...
│
├── jaccard_similarity.csv        # 🔥 跨 Seed Jaccard 相似度矩阵
├── stable_anomalies.csv          # 🔥 所有 seed 都识别的异常
└── multi_seed_summary.json       # 🔥 稳定性汇总报告
```

#### 稳定性指标解读

**1. Jaccard 相似度** (`jaccard_similarity.csv`)

| seed_i | seed_j | jaccard_fused | jaccard_graph | jaccard_tabular |
|--------|--------|---------------|---------------|-----------------|
| 42     | 123    | 0.856         | 0.823         | 0.901           |
| 42     | 456    | 0.871         | 0.845         | 0.889           |

- **0.7~0.9**: 模型稳定可靠 ✅
- **0.5~0.7**: 中等稳定，可使用
- **<0.5**: 对初始化敏感，需调参 ⚠️

**2. 稳定节点列表** (`stable_anomalies.csv`)

列出所有 seed 都识别的高置信度异常（Top-K 交集），**优先人工审核**

**3. 汇总报告** (`multi_seed_summary.json`)

```json
{
  "stable_nodes": {
    "all_seeds": 678,
    "ratio": 0.678
  }
}
```

- **Stable Ratio > 0.6**: 高置信度，可直接使用
- **0.4~0.6**: 建议人工复核
- **< 0.4**: 需要更多数据或调整模型

---

## 常见问题

