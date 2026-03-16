

## 1. 摘要

当前已完成：实现了一个“双视图 generalized-EM(GLEM)”训练闭环（表视图 + 图视图），包含 warmup → 交替伪标签 EM 轮次（M→E→M），并引入“验证集 AUPRC 选优 + 早停”避免后期训练崩坏。接下来将尝试不同互学方式。


## 2. 已完成工作

### 2.1 Tabular / M-view
- 数据与特征：交易级样本，过滤 `TRANSFER/CASH_OUT` 后按 `step` 顺序切分 train/val/test，并构造一组交易级数值与指示特征。
- warmup：用 `IsolationForest` 产出表侧初始风险分数 `M_warmup`（0-1 归一化），作为后续自举与互学的基础信号。
- 学生：表学生支持 `LightGBM` 或 fallback `HistGradientBoostingClassifier`，用伪标签在 train 内拟合得到 `M0/Mr` 的风险分数。

### 2.2 Graph / E-view

- 图构建：以交易为节点，按 `nameOrig/nameDest` 的最近 `k` 条历史交易连边，并补成双向边形成交易图。
- 图输入：图模型主输入是对表特征做 `StandardScaler`后的 dense 特征。
- 图模型：图侧 warmup 也是 `IsolationForest` 得到 `E_warmup`；图学生支持 `gcn` 或 `dominant`，训练得到 `E0/Er` 的风险分数。

### 2.3 EM 怎么构建的（warmup / 自举 / 互相伪标签 / 选优早停）

- warmup 与自举：先得到 `M_warmup/E_warmup`，再分别用自身 warmup 在 train 内分位数截断生成伪标签训练出 `M0/E0`。
- 互学轮次：按 `M → E → M` 交替用对方分数生成伪标签训练学生；支持 `peer_only`、`blended_rank`、`agreement_filtered` 三种互学打标方式。
- 选优与早停：在有 `isFraud` 时按 `val` 的 `AUPRC`为两支分别选最佳 stage，并在连续无提升达到 `patience` 时早停；输出为 `summary.json` 与 `dashboard.png`。

---

## 3. 当前实验结果分析

### 3.1 结果快照对应的事实

- 本次被选中的 stage 是表侧 `M_warmup` 与图侧 `E0`，而不是更后面的学生轮次。
- 训练仅完成 1 轮 EM 即早停，说明“继续互学”在该快照下没有稳定增益。
- train/val/test 欺诈率随时间显著上升，属于强时间漂移场景。


## 4. 现在的结果可以得出哪些结论

- 有效的部分只发生在早期；当前对称的 EM 没有带来稳定提升，继续迭代反而可能退化。
- 互学必须有保护：要么非对称互学，要么加门控/一致性约束，避免弱分支主导强分支。

## 5. 接下来要做什么

### 5.1 EM 相关

- 尝试不同互学方式：对比 `peer_only`/`blended_rank`/`agreement_filtered`，并加入非对称互学或门控互学，优先保护强分支不被反向污染。
- 降噪与稳健化：下调伪标签比例并引入置信度/时间窗口约束，让互学信号更“少而准”。
- 保留选优早停：继续以 `val AUPRC` 选最佳 stage 并早停，把“可用的早期增益”稳定保留下来。

### 5.2 其他部分

- 强化诊断输出：在结果里固定输出 split 欺诈率与伪标签统计（数量/占比/置信度），降低“看不见的漂移与噪声”。
- 建立低成本基线：在验证集上做图/表分数融合搜索，并明确当前环境下表模型 warm-start 能力的真实情况。
