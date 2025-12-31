# Processed Data

此目录用于存储预处理结果，由 `run_preprocess.py` 生成。

## 包含文件

- `graph_data.pt` - 完整图数据（PyG Data对象）
- `tabular_features.npy` - 表格特征矩阵
- `raw_data.pkl` - 原始数据副本
- `node_mapping.pkl` - 节点ID映射
- `feature_engineer.pkl` - 特征工程器
- `preprocess_meta.json` - 元信息
- `node_features.pt` - 节点特征
- `edge_index.pt` - 边索引
- `original_edge_index.pt` - 原始边索引
- `edge_features.pt` - 边特征

## 使用方法

```bash
# 生成预处理数据
python run_preprocess.py --data ../graph_main/raw_data/xxx.csv --output ./processed_data

# 使用预处理数据进行训练
python run_training.py --processed-data ./processed_data --output ./output
```
