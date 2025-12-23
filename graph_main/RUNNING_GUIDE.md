# GraphMAE 项目运行指南

本文档详细说明如何运行整个项目，以及在每个步骤可以查看哪些中间状态。

---

## 📚 目录

1. [快速开始](#快速开始)
2. [分步骤运行](#分步骤运行)
3. [中间状态查看](#中间状态查看)
4. [输出文件说明](#输出文件说明)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 方式一：一键运行（推荐新手）

```bash
# 使用简化脚本
python graph_main/run.py \
    --data_path /path/to/data.csv \
    --output_dir ./output \
    --sample_size 500000 \
    --epochs 300 \
    --device 0
```

**说明**：
- `--data_path`: CSV数据文件路径（必填）
- `--output_dir`: 输出目录，默认`./output`
- `--sample_size`: 采样大小，0表示全量数据
- `--epochs`: 训练轮数
- `--device`: GPU设备ID，-1表示CPU

### 方式二：完整参数运行

```bash
# 使用主程序，支持更多参数
python graph_main/main.py \
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
    --num_heads 4 \
    --epochs 500 \
    --lr 0.001 \
    --mask_rate 0.5 \
    --patience 20 \
    --device 0 \
    --save_model \
    --visualize
```

### 方式三：分步骤运行（推荐学习和调试）

```bash
# 使用交互式分步骤脚本
python graph_main/step_by_step_demo.py \
    --data_path /path/to/data.csv \
    --sample_size 100000 \
    --epochs 50 \
    --device 0
```

**特点**：
- 每个步骤结束后等待确认
- 显示详细的中间结果
- 可以逐步检查每个阶段的输出

---

## 📊 分步骤运行

### 步骤1: 数据加载

**运行命令**：
```python
from graph_main.data_loader import DataLoader
from graph_main.config import Config

config = Config()
config.data.data_path = "path/to/data.csv"
config.data.sample_size = 500000

loader = DataLoader(config.data)
loader.load_csv()
```

**可查看的中间状态**：
```python
# 查看加载的数据
print(loader.df.shape)              # 数据形状
print(loader.df.head())             # 前几行数据
print(loader.df.info())             # 数据类型信息
print(loader.df.isnull().sum())     # 缺失值统计

# 查看特定列（按索引）
print(loader.df.iloc[:, 14].head()) # 支付方账户列
print(loader.df.iloc[:, 15].head()) # 收款方账户列
print(loader.df.iloc[:, 6].head())  # 金额列
```

**输出文件**：无

---

### 步骤2: 图构建

**运行命令**：
```python
# 2.1 创建节点映射
node_map, num_nodes = loader.create_node_mapping()

# 2.2 构建边索引
edge_index = loader.build_edge_index()

# 2.3 构建完整图
data = loader.build_pyg_data()
```

**可查看的中间状态**：
```python
# 节点映射
print(f"节点数: {num_nodes}")
print(f"部分映射: {list(node_map.items())[:5]}")

# 边信息
print(f"边数: {edge_index.shape[1]}")
print(f"前5条边: {edge_index[:, :5]}")

# 图数据对象
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.edge_index.shape[1]}")
print(f"节点特征维度: {data.x.shape[1]}")
print(f"节点特征形状: {data.x.shape}")

# 查看节点特征统计
import torch
print(f"特征均值: {data.x.mean(dim=0)}")
print(f"特征标准差: {data.x.std(dim=0)}")
print(f"特征最大值: {data.x.max(dim=0).values}")
print(f"特征最小值: {data.x.min(dim=0).values}")
```

**输出文件**：无

---

### 步骤3: 描述性统计

**运行命令**：
```python
from graph_main.statistics import GraphStatistics

stats = GraphStatistics(data, loader)
report = stats.print_full_report()
```

**可查看的中间状态**：
```python
# 基本统计
basic = report['basic']
print(f"节点数: {basic['num_nodes']}")
print(f"边数: {basic['num_edges_with_loops']}")

# 度数统计
degree_stats = report['degree']
print(f"平均入度: {degree_stats['in_degree']['mean']}")
print(f"平均出度: {degree_stats['out_degree']['mean']}")
print(f"最大度数: {degree_stats['total_degree']['max']}")

# 图密度
density = report['density']
print(f"图密度: {density:.10f}")

# 账户类型
accounts = report['accounts']
print(f"仅发送账户: {accounts['source_only_accounts']}")
print(f"仅接收账户: {accounts['dest_only_accounts']}")
print(f"双向账户: {accounts['bidirectional_accounts']}")
```

**输出文件**：
- `./output/intermediate/statistics_step3.json`：完整统计报告

**查看方式**：
```bash
# Windows
notepad ./output/intermediate/statistics_step3.json

# Linux/Mac
cat ./output/intermediate/statistics_step3.json | python -m json.tool
```

---

### 步骤4: 构建模型

**运行命令**：
```python
from graph_main.models import GraphMAE
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GraphMAE(
    in_channels=data.x.shape[1],
    hidden_channels=256,
    out_channels=128,
    encoder_type='gat',
    decoder_type='gat',
    num_layers=2,
    num_heads=4,
    dropout=0.2,
    mask_rate=0.5,
    replace_rate=0.1,
    loss_fn='sce',
    alpha_l=2.0,
    use_dgl=False
).to(device)
```

**可查看的中间状态**：
```python
# 模型结构
print(model)

# 参数统计
from graph_main.models import count_parameters
total_params = count_parameters(model)
print(f"总参数量: {total_params:,}")

# 各层参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} ({param.numel()} params)")

# 模型配置
print(f"编码器类型: {model._encoder_type}")
print(f"解码器类型: {model._decoder_type}")
print(f"掩码比例: {model.mask_rate}")
```

**输出文件**：无

---

### 步骤5: 训练模型

**运行命令**：
```python
from graph_main.trainer import Trainer

trainer = Trainer(model, config.train, device)
history = trainer.train(data, verbose=True)
```

**可查看的中间状态**：
```python
# 训练过程（每个epoch）
# 会自动显示：
# - 当前epoch
# - 当前损失
# - 学习率

# 训练完成后
print(f"训练轮数: {history['epochs_trained']}")
print(f"最佳损失: {history['best_loss']:.6f}")
print(f"最终损失: {history['train_losses'][-1]:.6f}")

# 查看训练曲线
import matplotlib.pyplot as plt
plt.plot(history['train_losses'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 保存模型
torch.save(model.state_dict(), './output/model_checkpoint.pt')
```

**输出文件**：
- `./output/intermediate/training_history_step5.json`：训练历史
- `./checkpoints/graphmae_checkpoint.pt`：模型检查点（如果启用）

**查看训练历史**：
```python
import json
with open('./output/intermediate/training_history_step5.json', 'r') as f:
    history = json.load(f)
    
print(f"损失变化: {history['train_losses'][:10]}")  # 前10个epoch
```

---

### 步骤6: 异常检测

**运行命令**：
```python
from graph_main.anomaly_detector import AnomalyDetector, UnsupervisedEvaluator

detector = AnomalyDetector(model, config.anomaly)

# 6.1 计算节点异常分数
node_scores = detector.compute_reconstruction_error(data)

# 6.2 计算边异常分数
edge_scores = detector.compute_edge_anomaly_scores(data)

# 6.3 获取节点嵌入
node_embeddings = detector.get_node_embeddings(data)

# 6.4 评估报告
evaluator = UnsupervisedEvaluator(detector)
evaluator.print_report('edge')
```

**可查看的中间状态**：
```python
import numpy as np

# 节点异常分数
print(f"节点分数形状: {node_scores.shape}")
print(f"平均分数: {np.mean(node_scores):.6f}")
print(f"标准差: {np.std(node_scores):.6f}")
print(f"最小值: {np.min(node_scores):.6f}")
print(f"最大值: {np.max(node_scores):.6f}")
print(f"95th分位数: {np.percentile(node_scores, 95):.6f}")

# 边异常分数
print(f"边分数形状: {edge_scores.shape}")
print(f"平均分数: {np.mean(edge_scores):.6f}")
print(f"95th分位数: {np.percentile(edge_scores, 95):.6f}")

# Top异常
top_indices, top_scores = detector.get_top_anomalies(k=100, level='edge')
print(f"\nTop 10 最异常的交易:")
for i, (idx, score) in enumerate(zip(top_indices[:10], top_scores[:10])):
    print(f"  {i+1}. 交易 {idx}: 分数 {score:.6f}")

# 异常分布
threshold_95 = np.percentile(edge_scores, 95)
anomalies = (edge_scores > threshold_95).sum()
print(f"\n异常交易数（95th分位数）: {anomalies} ({anomalies/len(edge_scores)*100:.2f}%)")
```

**输出文件**：
- `./output/intermediate/anomaly_scores_step6.json`：异常分数和统计

**查看异常结果**：
```python
import json
with open('./output/intermediate/anomaly_scores_step6.json', 'r') as f:
    results = json.load(f)
    
# 查看Top异常
print("Top 10 异常:")
for i, (idx, score) in enumerate(zip(results['top_100_indices'][:10], 
                                      results['top_100_scores'][:10])):
    print(f"{i+1}. 交易 {idx}: {score:.6f}")

# 查看统计信息
print("\n统计信息:")
print(results['statistics'])
```

---

### 步骤7: 可视化

**运行命令**：
```python
from graph_main.visualization import Visualizer
from torch_geometric.utils import degree

visualizer = Visualizer('./output')

# 计算节点度数
node_degrees = degree(data.original_edge_index[0], num_nodes=data.num_nodes).cpu().numpy()

# 7.1 训练损失曲线
visualizer.plot_training_loss(
    history['train_losses'],
    save_path='./output/training_loss.png'
)

# 7.2 异常分数分布
visualizer.plot_score_distribution(
    edge_scores,
    save_path='./output/score_distribution.png'
)

# 7.3 度数vs分数
visualizer.plot_node_degree_vs_score(
    node_degrees,
    node_scores,
    save_path='./output/degree_vs_score.png'
)

# 7.4 综合报告
visualizer.plot_comprehensive_report(
    train_losses=history['train_losses'],
    node_scores=node_scores,
    edge_scores=edge_scores,
    node_degrees=node_degrees,
    save_path='./output/comprehensive_report.png'
)

# 7.5 t-SNE嵌入可视化
visualizer.plot_embeddings_tsne(
    embeddings=node_embeddings,
    scores=node_scores,
    sample_size=3000,
    save_path='./output/embeddings_tsne.png'
)
```

**输出文件**：
- `./output/training_loss.png`：训练损失曲线
- `./output/score_distribution.png`：异常分数分布
- `./output/degree_vs_score.png`：度数vs分数
- `./output/comprehensive_report.png`：综合报告
- `./output/embeddings_tsne.png`：t-SNE可视化

**查看图片**：
- Windows: 直接双击打开PNG文件
- Linux: `eog *.png` 或 `feh *.png`
- Mac: `open *.png`

---

## 📁 输出文件说明

### 文件结构

```
output/
├── run_YYYYMMDD_HHMMSS/          # 每次运行的时间戳目录
│   ├── statistics_report.json     # 数据统计报告
│   ├── anomaly_results.json       # 异常检测结果
│   ├── graphmae_model.pt          # 训练好的模型
│   ├── comprehensive_report.png   # 综合可视化
│   └── embeddings_tsne.png        # t-SNE可视化
│
└── intermediate/                  # 分步骤运行的中间结果
    ├── statistics_step3.json
    ├── training_history_step5.json
    ├── anomaly_scores_step6.json
    ├── training_loss.png
    ├── score_distribution.png
    ├── degree_vs_score.png
    └── top_anomalies.png
```

### 文件内容说明

#### 1. `statistics_report.json`
```json
{
  "basic": {
    "num_nodes": 123456,
    "num_edges": 500000,
    "num_node_features": 18
  },
  "degree": {
    "in_degree": {"mean": 4.05, "max": 1234},
    "out_degree": {"mean": 4.05, "max": 987}
  },
  "density": 0.00003245,
  "accounts": {
    "source_only_accounts": 45678,
    "dest_only_accounts": 56789,
    "bidirectional_accounts": 21000
  }
}
```

#### 2. `anomaly_results.json`
```json
{
  "node_scores": [0.234, 0.456, ...],
  "edge_scores": [0.123, 0.345, ...],
  "top_anomaly_indices": [12345, 67890, ...],
  "top_anomaly_scores": [0.987, 0.876, ...],
  "training_history": {
    "losses": [1.234, 0.987, ...],
    "best_loss": 0.456,
    "epochs": 250
  }
}
```

---

## 🔍 中间状态查看技巧

### 1. 使用Jupyter Notebook

创建 `analysis.ipynb`：
```python
# Cell 1: 导入和配置
import sys
sys.path.insert(0, './graph_main')
from data_loader import DataLoader
from config import Config

config = Config()
config.data.data_path = "path/to/data.csv"

# Cell 2: 加载数据
loader = DataLoader(config.data)
loader.load_csv()
loader.df.head()

# Cell 3: 查看统计
loader.df.describe()

# 后续步骤...
```

### 2. 使用Python交互式

```bash
python -i graph_main/run.py --data_path data.csv
```

运行完成后会进入交互式模式，可以查看变量。

### 3. 添加断点调试

在代码中添加：
```python
import pdb; pdb.set_trace()
```

### 4. 保存中间结果

```python
# 在任何步骤后保存
import pickle
with open('intermediate_data.pkl', 'wb') as f:
    pickle.dump({'data': data, 'model': model}, f)

# 后续加载
with open('intermediate_data.pkl', 'rb') as f:
    saved = pickle.load(f)
```

---

## ❓ 常见问题

### Q1: 内存不足怎么办？
**A**: 减小采样大小
```bash
python run.py --data_path data.csv --sample_size 100000
```

### Q2: 训练太慢怎么办？
**A**: 减少训练轮数或使用GPU
```bash
python run.py --data_path data.csv --epochs 100 --device 0
```

### Q3: 如何查看具体的异常交易详情？
**A**: 使用异常索引查询原始数据
```python
import json
with open('./output/intermediate/anomaly_scores_step6.json', 'r') as f:
    results = json.load(f)

top_indices = results['top_100_indices']

# 查看这些交易的原始数据
anomalous_transactions = loader.df.iloc[top_indices]
anomalous_transactions.to_csv('./anomalous_transactions.csv')
```

### Q4: 如何调整异常阈值？
**A**: 修改配置或手动设置
```python
# 方法1: 修改配置
config.anomaly.threshold_percentile = 99.0  # 更严格

# 方法2: 手动分类
threshold = np.percentile(edge_scores, 99)
anomalies = edge_scores > threshold
```

### Q5: 如何重新加载训练好的模型？
**A**:
```python
model.load_state_dict(torch.load('./output/run_xxx/graphmae_model.pt'))
model.eval()
```

---

## 📝 完整运行示例

```bash
# 1. 安装依赖
pip install -r graph_main/requirements.txt

# 2. 快速测试（小数据）
python graph_main/step_by_step_demo.py \
    --data_path data.csv \
    --sample_size 50000 \
    --epochs 30 \
    --device 0

# 3. 正式运行（完整流程）
python graph_main/main.py \
    --data_path data.csv \
    --sample_size 500000 \
    --epochs 500 \
    --device 0 \
    --save_model \
    --visualize

# 4. 查看结果
ls -lh output/run_*/
```

---

## 📊 推荐的检查流程

1. **数据检查**（步骤1-2）
   - 确认数据加载正确
   - 检查节点和边的数量
   - 验证特征提取

2. **统计分析**（步骤3）
   - 查看图的基本属性
   - 了解度数分布
   - 识别潜在问题

3. **训练监控**（步骤5）
   - 观察损失下降
   - 检查是否过拟合
   - 确认早停触发

4. **结果分析**（步骤6-7）
   - 查看异常分数分布
   - 分析Top异常交易
   - 验证结果合理性

---

希望这份指南能帮助你顺利运行项目！如有问题，请查看日志输出或提issue。
