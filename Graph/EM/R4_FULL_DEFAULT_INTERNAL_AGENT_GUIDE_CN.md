# R4-Full 默认实现：内部 Agent 使用说明

> Python 实现：[`r4_full_default.py`](r4_full_default.py)  
> 用途：在内部图数据上训练完整 R4-Full，导出节点 embedding 和异常分数。

## 1. 方法

R4-Full 的计算路径为：

```text
raw attributes
  ├─ RP256 context ──> 第一层 neighbour/message
  └─ ERF root ───────> 第一层 root/self
                           ↓
               role-separated GraphSAGE
                           ↓
                 64-d node embedding
                           ↓
                  shared anomaly logit
                    ↙             ↘
          seen-normal BCE    R-vs-U origin loss
```

完整方法对应冻结实验中的 `method=bce_nmcr`，不包含 NSReg edge labeller、伪异常、伪标签或多分支分数融合。

## 2. 输入

Python 程序读取一个 NPZ 文件，必须包含：

| Key | Shape | 含义 |
|---|---:|---|
| `x` | `[N, d]` | 原始节点属性，float |
| `edge_index` | `[2, M]` | PyG COO 图边，int64 |
| `normal_idx` | `[n_R]` | 已确认正常的训练节点 |
| `seen_idx` | `[n_S]` | 已确认异常的训练节点 |
| `deployment_idx` | `[n_U]` | outcome 不可见的部署节点 |

`normal_idx`、`seen_idx` 和 `deployment_idx` 互不重叠。R4 是 transductive 方法：训练时会使用 deployment 节点的属性、连接和集合归属，但不使用它们的正常/异常 outcome。

没有现成 split 时，默认使用正常节点的 5% 作为 `normal_idx`、50 个已知异常作为 `seen_idx`，其余待评分节点作为 `deployment_idx`。split 由数据准备阶段写入 NPZ，训练程序不再重新划分。

## 3. 完整组件

### RP256 neighbour view

原始属性依次经过 signed-log1p、逐行 L2、256 维 Sparse Random Projection 和再次逐行 L2。投影种子固定为 `260715`。该视图只进入第一层邻居聚合端，训练期间冻结。

### ERF root view

ERF 使用 `normal_idx` 对应的原始属性做 float64 economy SVD，以 float32 数值秩保留正常参考行空间，再用固定 signed-permutation sketch 把正交残差折叠回参考坐标。最终以正常参考节点的全局 RMS 缩放。该视图只进入第一层 root/self 端，训练期间冻结。

### GraphSAGE 与 embedding

编码器为两层 mean GraphSAGE。第一层分别接收 RP neighbour 和 ERF root，输出 64 维隐藏表示；第二层恢复普通共享 GraphSAGE。随后使用 `64 → 64 → 64` projector 和 `64 → 32 → 1` classifier。

最终 embedding 是 projector 输出、classifier 之前的 64 维向量。

### 训练目标

模型只产生一个共享异常 logit。训练目标为：

$$
\mathcal L
=\mathcal L_{\mathrm{seen}}
+2\mathcal L_{\mathrm{origin}}.
$$

Seen loss 在全部已知异常和每轮至多 512 个已知正常节点上计算 BCE。

Origin 分支先在完整正常参考集上计算：

$$
a=\operatorname{logsumexp}(z_R)-\log|R|,
\qquad u_i=z_i-a.
$$

然后使用可学习标量 $\eta$ 和 $\alpha=\sigma(\eta)$：

$$
o_i=\log[(1-\alpha)+\alpha e^{u_i}],
$$

并计算分组平衡的来源损失：

$$
\mathcal L_{\mathrm{origin}}
=\frac12\operatorname{BCELogit}(o_R,0)
+\frac12\operatorname{BCELogit}(o_U,1).
$$

## 4. 固定默认参数

| 参数 | 默认值 |
|---|---:|
| Seed | 42 |
| RP dimension | 256 |
| RP seed | 260715 |
| ERF sketch id | 0 |
| ERF residual multiplier | 1.0 |
| GraphSAGE layers | 2 |
| Hidden dimension | 64 |
| Embedding dimension | 64 |
| Classifier hidden dimension | 32 |
| Dropout | 0.5 |
| Fanouts | `[25, 10]` |
| Full-graph root batch | 50000 |
| Seen BCE normal batch | 512 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Epochs | 201 |
| Origin weight | 2.0 |
| $\eta$ initialization | -2.0 |
| Early stopping | 不使用 |

训练结束后只使用 final model，不使用 scheduler、validation early stopping、temperature calibration 或 ensemble。

## 5. 运行

在 `renqi` 环境中执行：

```bash
conda activate renqi
python r4_full_default.py \
  --input /path/to/internal_graph.npz \
  --dataset-name internal_graph \
  --output-dir /path/to/r4_output \
  --device cuda:0
```

## 6. 输出

输出目录包含：

| 文件 | 内容 |
|---|---|
| `r4_embedding.npz` | 节点 embedding、logit、score 和 node id |
| `r4_model.pt` | R4-Full 模型参数 |

`r4_embedding.npz` 中：

| Key | Shape | 含义 |
|---|---:|---|
| `embedding` | `[N, 64]` | 内部 Graph Embedding |
| `logit` | `[N]` | 共享异常 logit |
| `score` | `[N]` | `sigmoid(logit)`，越大越异常 |
| `node_id` | `[N]` | 与输入节点顺序一致 |

## 7. 代码入口

所有实现均在 [`r4_full_default.py`](r4_full_default.py)：

- `build_rp_context`：RP256；
- `build_erf_root`：ERF；
- `R4Full`：角色分离 GraphSAGE、projector 和 classifier；
- `forward_all`：全节点 sampled forward；
- `r4_loss`：完整 R4 objective；
- `train_r4`：固定 201 epochs 训练；
- `save_outputs`：embedding、score 和模型导出。

冻结实现参考：

- [`build_erf_root_view.py`](idea-stage/pilots/build_erf_root_view.py)
- [`train_runner.py`](analysis/r4_iclr_evidence_20260725/code_snapshot_direct_origin/runners/train_runner.py)
- [`base_gnns.py`](analysis/r4_iclr_evidence_20260725/code_snapshot_direct_origin/models/base_gnns.py)
