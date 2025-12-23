# GraphMAE Fraud Detection

基于GraphMAE（Graph Masked Autoencoder）的支付交易欺诈检测系统。

## 项目概述

本项目实现了一个无监督的图神经网络异常检测系统，专门针对支付交易数据进行欺诈检测。核心方法基于GraphMAE，通过掩码自编码学习图的表示，利用重建误差作为异常分数来识别潜在的欺诈交易。

### 主要特点

- **无监督学习**：不需要欺诈标签，适用于真实场景
- **图结构建模**：将交易数据建模为图，捕获账户间的交易关系
- **灵活的特征工程**：支持数值、类别和时间特征的自动处理
- **按列索引访问**：所有数据字段通过列索引访问，避免列名依赖

## 数据字段说明

本项目针对以下数据格式设计（按列索引）：

| 列索引 | 字段名 | 说明 |
|--------|--------|------|
| 0 | uetr | 交易唯一标识 |
| 1 | payment_channel | 交易渠道 |
| 2 | debit_bic_code | 支付方BIC码 |
| 3 | bene_bic_code | 收款方BIC码 |
| 4 | evt_tran_stat_cde | 支付状态码 |
| 5 | instructed_currency | 客户指定币种 |
| 6 | instructed_amount | 客户指定金额 |
| 7 | payment_currency | 银行使用币种 |
| 8 | payment_amount | 银行使用金额 |
| 9 | credit_currency | 收款方接收币种 |
| 10 | credit_amount | 收款方接收金额 |
| 11 | txn_dt | 事件发生时间 |
| 12 | tds_dt | 入库时间戳 |
| 13 | mop | 付款方式 |
| 14 | debit_account_masked | 支付方账户(masked) |
| 15 | bene_account_masked | 收款方账户(masked) |

## 项目结构

```
graph_main/
├── config.py              # 配置文件，定义所有参数
├── data_loader.py         # 数据加载和图构建模块
├── statistics.py          # 描述性统计分析模块
├── trainer.py             # 训练器模块
├── anomaly_detector.py    # 异常检测模块
├── visualization.py       # 可视化模块
├── main.py                # 主程序入口
├── requirements.txt       # 依赖包列表
├── README.md              # 项目说明文档
└── models/                # 模型模块
    ├── __init__.py
    ├── utils.py           # 工具函数
    ├── loss_func.py       # 损失函数
    ├── encoder.py         # 编码器（GAT/GCN）
    └── graphmae.py        # GraphMAE核心模型
```

## 安装

```bash
# 创建虚拟环境（推荐）
conda create -n graphmae python=3.9
conda activate graphmae

# 安装依赖
pip install -r requirements.txt

# 或者使用conda安装PyTorch和PyG
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
```

## 使用方法

### 命令行运行

```bash
# 基本用法
python main.py --data_path /path/to/your/data.csv --output_dir ./output

# 完整参数示例
python main.py \
    --data_path /path/to/data.csv \
    --output_dir ./output \
    --sample_size 500000 \
    --src_col 14 \
    --dst_col 15 \
    --encoder_type gat \
    --decoder_type gat \
    --hidden_channels 256 \
    --out_channels 128 \
    --num_layers 2 \
    --epochs 500 \
    --lr 0.001 \
    --mask_rate 0.5 \
    --device 0 \
    --save_model \
    --visualize
```

### Python API

```python
from config import Config, DataConfig, ModelConfig, TrainConfig, AnomalyConfig
from data_loader import DataLoader
from models import GraphMAE, set_random_seed
from trainer import Trainer
from anomaly_detector import AnomalyDetector

# 1. 配置
config = Config()
config.data.data_path = "path/to/data.csv"
config.data.src_col = 14  # 支付方账户列
config.data.dst_col = 15  # 收款方账户列

# 2. 加载数据
loader = DataLoader(config.data)
loader.load_csv()
data = loader.build_pyg_data()

# 3. 构建模型
model = GraphMAE(
    in_channels=data.x.shape[1],
    hidden_channels=256,
    out_channels=128,
    encoder_type='gat',
    decoder_type='gat',
    mask_rate=0.5
)

# 4. 训练
trainer = Trainer(model, config.train, device)
history = trainer.train(data)

# 5. 异常检测
detector = AnomalyDetector(model, config.anomaly)
node_scores = detector.compute_reconstruction_error(data)
edge_scores = detector.compute_edge_anomaly_scores(data)

# 6. 获取Top异常
top_indices, top_scores = detector.get_top_anomalies(k=100)
```

## 核心算法

### GraphMAE工作原理

1. **特征掩码 (Feature Masking)**
   - 随机选择一定比例（如50%）的节点
   - 将这些节点的特征替换为可学习的掩码令牌

2. **编码 (Encoding)**
   - 使用GNN（如GAT或GCN）对掩码后的图进行编码
   - 获取所有节点的嵌入表示

3. **解码 (Decoding)**
   - 使用解码器尝试重建被掩码节点的原始特征
   - 训练目标是最小化重建误差

4. **异常检测**
   - 正常节点：特征可以从邻居信息中被准确重建
   - 异常节点：行为模式与正常节点不同，难以被正确重建
   - 重建误差高的节点/边更可能是异常

### 损失函数

使用Scaled Cosine Error (SCE)损失：

$$
\mathcal{L}_{SCE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} (1 - \cos(x_i, \hat{x}_i))^\alpha
$$

其中：
- $\mathcal{M}$ 是被掩码节点的集合
- $x_i$ 是原始特征
- $\hat{x}_i$ 是重建特征
- $\alpha$ 是缩放参数（默认为2）

## 配置说明

### 数据配置 (DataConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| data_path | "" | CSV数据文件路径 |
| src_col | 14 | 源节点列索引 |
| dst_col | 15 | 目标节点列索引 |
| numerical_cols | [6, 8, 10] | 数值特征列索引 |
| categorical_cols | [1, 2, 3, 4, 5, 7, 9, 13] | 类别特征列索引 |
| time_cols | [11, 12] | 时间特征列索引 |
| sample_size | 500000 | 采样大小 |

### 模型配置 (ModelConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| encoder_type | "gat" | 编码器类型 |
| decoder_type | "gat" | 解码器类型 |
| hidden_channels | 256 | 隐藏层维度 |
| out_channels | 128 | 输出嵌入维度 |
| num_layers | 2 | GNN层数 |
| num_heads | 4 | GAT注意力头数 |
| dropout | 0.2 | Dropout率 |

### 训练配置 (TrainConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 500 | 训练轮数 |
| lr | 0.001 | 学习率 |
| weight_decay | 1e-5 | 权重衰减 |
| mask_rate | 0.5 | 掩码比例 |
| replace_rate | 0.1 | 随机替换比例 |
| patience | 20 | 早停耐心值 |

## 输出结果

运行后会生成以下文件：

```
output/run_YYYYMMDD_HHMMSS/
├── statistics_report.json     # 数据统计报告
├── anomaly_results.json       # 异常检测结果
├── graphmae_model.pt          # 训练好的模型（如果启用保存）
├── comprehensive_report.png   # 综合可视化报告
└── embeddings_tsne.png        # 节点嵌入t-SNE可视化
```

## 参考文献

- Hou, Z., et al. "GraphMAE: Self-supervised masked graph autoencoders." KDD 2022.
- Hamilton, W., et al. "Inductive representation learning on large graphs." NeurIPS 2017.
- Veličković, P., et al. "Graph attention networks." ICLR 2018.

## License

MIT License
