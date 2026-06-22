# NoCo: Normality-aware Codebook Learning for Open-Set Graph Anomaly Detection

---

## 1. Problem Formulation

### 1.1 Graph Anomaly Detection (GAD)

Given an attributed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$ where $\mathcal{V}$ is the node set, $\mathcal{E}$ is the edge set, and $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times d}$ denotes node features, the goal of graph anomaly detection is to assign an anomaly score $s(v) \in \mathbb{R}$ to each node $v \in \mathcal{V}$, such that anomalous nodes receive higher scores than normal ones.

In practice, the training set typically contains a small set of labeled normal nodes $\mathcal{V}_N$ and a set of labeled anomalies $\mathcal{V}_A$, with the vast majority of nodes remaining unlabeled. Critically, $\mathcal{V}_A$ does **not** exhaust the space of possible anomaly types.

### 1.2 The Open-Set Challenge

The fundamental challenge in GAD is the **open-set** nature of anomalies: the anomaly types observed during training are an incomplete subset of all possible anomaly types that may appear at test time. A model trained to discriminate known anomaly types from normal ones may fail to generalize to **novel anomaly types** — previously unseen patterns that deviate from normality in unforeseen ways.

This motivates a shift in modeling philosophy: rather than modeling what anomalies look like, we model what **normality** looks like, and flag deviations from it.

---

## 2. Core Idea

> **Learn a compact codebook that encodes normality. Score nodes by the description length — the number of additional bits required to encode a node beyond what the normality codebook can explain.**

The key insight is that **normality is compressible** while **anomalies are not**. Normal nodes share common structural and attribute patterns that can be captured by a small set of prototypical codes. Anomalies, by their diverse and unpredictable nature, cannot be efficiently represented by the same codebook.

This connects directly to the **Minimum Description Length (MDL)** principle: given a learned normality codebook $\mathcal{C}$, the anomaly score of a node is the description length surplus:

$$s(v) = L(v \mid \mathcal{C}) - L_{\text{baseline}}$$

where $L(v \mid \mathcal{C})$ is the minimal coding cost of node $v$ using codebook $\mathcal{C}$, and $L_{\text{baseline}}$ is the expected coding cost for a normal node of the same structural role.

---

## 3. Multi-View Encoding

A single view of a node is insufficient for robust anomaly detection. A node may appear normal in attribute space while exhibiting anomalous relational patterns, or vice versa. NoCo therefore models two complementary evidence channels:

| View | Definition | Examples |
|------|-----------|----------|
| **Node Evidence** ($\mathbf{x}_v$) | The intrinsic features of node $v$ | User profile attributes, transaction amounts, review content embeddings |
| **Relational Evidence** ($\mathbf{r}_v$) | The behavioral pattern of $v$ within the graph topology | Neighborhood composition, message-passing patterns with neighbors, local structural role |

The critical signal for anomaly detection often lies in **cross-view inconsistency**: a node whose attributes suggest normality but whose relational context is anomalous (or vice versa) is a strong anomaly candidate. NoCo explicitly models this disagreement.

---

## 4. Method

### 4.1 Normality Codebook

NoCo learns a discrete codebook $\mathcal{C} = \{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_K\}$ where $K$ is small (e.g., 32 or 64). Each code vector $\mathbf{e}_k \in \mathbb{R}^{d_c}$ represents a **prototypical normal pattern** — not a cluster center of normal nodes, but a reusable building block for encoding normal behavior.

The codebook is learned from:
- **Labeled normal nodes** $\mathcal{V}_N$: direct supervision on what normality looks like.
- **High-confidence unlabeled nodes**: selected via a confidence threshold during training, under the assumption that the majority of unlabeled nodes are normal.

### 4.2 Encoding Process

For each node $v$, NoCo performs three operations:

**① Code Assignment.** Select the codebook entry that best explains the node:

$$c_v = \arg\min_{k \in \{1,\dots,K\}} \|\mathbf{x}_v - \text{Dec}_x(\mathbf{e}_k)\|^2$$

where $\text{Dec}_x$ is an attribute decoder. The assignment cost is $L(z_v = c_v \mid \mathbf{c}_v)$, where $\mathbf{c}_v$ denotes the structural context (degree, role) of $v$.

**② Attribute Reconstruction.** Decode the node's own features from the assigned code:

$$\hat{\mathbf{x}}_v = \text{Dec}_x(\mathbf{e}_{c_v})$$

The attribute description length is $L(\mathbf{x}_v \mid z_v = c_v, \mathbf{c}_v)$, proportional to the reconstruction error $\|\mathbf{x}_v - \hat{\mathbf{x}}_v\|^2$.

**③ Conditional Relational Reconstruction.** Predict the node's relational evidence conditioned on both the assigned code and its attributes:

$$\hat{\mathbf{r}}_v = \text{Dec}_r(\mathbf{e}_{c_v}, \mathbf{x}_v)$$

The relational description length is $L(\mathbf{r}_v \mid z_v = c_v, \mathbf{x}_v, \mathbf{c}_v)$, proportional to $\|\mathbf{r}_v - \hat{\mathbf{r}}_v\|^2$.

For a normal node, all three costs are low and mutually consistent. For an anomalous node, at least one cost is elevated, and cross-view agreement breaks down.

### 4.3 Anti-Code Training with Known Anomalies

Known anomalies are **not** used to learn "what anomalies look like." Instead, they serve as **negative constraints** on the codebook: they define what the codebook must **fail** to compress.

Formally, for any known anomaly $a \in \mathcal{V}_A$ and its role-matched normal counterpart:

$$\mathcal{L}(a) \geq \mathcal{L}(v_{\text{normal}}, \text{role}(a)) + \delta$$

where $\delta > 0$ is a safety margin. This constraint ensures that the codebook does not learn to encode known anomalies efficiently, while avoiding the pitfall of assuming that all future anomalies will resemble known ones.

This is implemented via a margin-based ranking loss:

$$\mathcal{L}_{\text{anti}} = \sum_{a \in \mathcal{V}_A} \max\left(0, \mathcal{L}(v_{\text{normal}}) + \delta - \mathcal{L}(a)\right)$$

### 4.4 Role-Aware Calibration

Different structural roles naturally require different description lengths, even among normal nodes:

| Role | Expected $L$ | Reason |
|------|:------------:|--------|
| Typical nodes | Low | Simple, common patterns |
| Hub nodes (high degree) | Higher | Complex neighborhood, naturally requires more bits |
| Bridge nodes (high betweenness) | Higher | Cross-community position, less "typical" |
| Low-degree nodes | Potentially higher | Scarce neighborhood information |

Without calibration, hub and bridge nodes would be systematically flagged as anomalous. NoCo calibrates scores within each structural role group:

$$s(v) = \frac{\mathcal{L}(v) - \mu_{\text{role}(v)}}{\sigma_{\text{role}(v)}} + \beta \cdot \Delta_{\text{cross-view}}(v)$$

where $\mu_{\text{role}(v)}$ and $\sigma_{\text{role}(v)}$ are the mean and standard deviation of description lengths for normal nodes sharing the same structural role as $v$, and $\Delta_{\text{cross-view}}(v)$ quantifies the discrepancy between attribute-based and relation-based reconstructions.

### 4.5 Codebook Regularization

To prevent **codebook collapse** — the degenerate case where all nodes are assigned to a single code — NoCo employs:

- **Entropy regularization** on the assignment distribution: $-\sum_k p_k \log p_k$, encouraging uniform code usage.
- **Codebook resetting**: periodically re-initialize underutilized codes with embeddings of nodes that are poorly reconstructed.

---

## 5. Training Objective

The full training loss for a node $v$ is:

$$\mathcal{L}(v) = \underbrace{\mathcal{L}_{\text{assign}}(z_v \mid \mathbf{c}_v)}_{\text{Code assignment cost}} + \underbrace{\mathcal{L}_{\text{attr}}(\mathbf{x}_v \mid z_v, \mathbf{c}_v)}_{\text{Attribute reconstruction}} + \underbrace{\mathcal{L}_{\text{rel}}(\mathbf{r}_v \mid z_v, \mathbf{x}_v, \mathbf{c}_v)}_{\text{Relational reconstruction}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{agree}}(\mathbf{x}_v, \mathbf{r}_v, z_v)}_{\text{Cross-view consistency}}$$

with the anti-code constraint applied to known anomalies:

$$\mathcal{L}_{\text{total}} = \frac{1}{|\mathcal{V}_N|} \sum_{v \in \mathcal{V}_N} \mathcal{L}(v) + \alpha \cdot \mathcal{L}_{\text{anti}} + \gamma \cdot \mathcal{L}_{\text{reg}}$$

where $\mathcal{L}_{\text{reg}}$ includes entropy regularization and any additional priors.

### Notation Summary

| Symbol | Meaning |
|--------|---------|
| $z_v \in \{1,\dots,K\}$ | Discrete code assigned to node $v$ |
| $\mathbf{e}_k \in \mathbb{R}^{d_c}$ | The $k$-th entry in the normality codebook $\mathcal{C}$ |
| $\mathbf{c}_v$ | Structural context / role descriptor of node $v$ |
| $\mathbf{x}_v \in \mathbb{R}^{d}$ | Node evidence (intrinsic attributes) |
| $\mathbf{r}_v \in \mathbb{R}^{d_r}$ | Relational evidence (neighborhood behavioral pattern) |
| $\text{Dec}_x$ | Attribute decoder |
| $\text{Dec}_r$ | Relational decoder (conditioned on code + attributes) |

---

## 6. Anomaly Scoring

At inference time, the anomaly score for node $v$ is computed as:

$$s(v) = \frac{\mathcal{L}(v) - \mu_{\text{role}(v)}}{\sigma_{\text{role}(v)}} + \beta \cdot \Delta_{\text{cross-view}}(v)$$

where:

- **Term 1**: The role-calibrated description length surplus — how many more bits does $v$ require compared to its role peers?
- **Term 2**: The cross-view disagreement penalty — are the attribute-based and relation-based explanations of $v$ consistent?

The cross-view disagreement $\Delta_{\text{cross-view}}(v)$ can be operationalized as:

$$\Delta_{\text{cross-view}}(v) = \|\hat{\mathbf{x}}_v^{\text{(from rel)}} - \mathbf{x}_v\|^2 + \|\hat{\mathbf{r}}_v^{\text{(from attr)}} - \mathbf{r}_v\|^2$$

where $\hat{\mathbf{x}}_v^{\text{(from rel)}}$ is the attribute prediction from the relational pathway and $\hat{\mathbf{r}}_v^{\text{(from attr)}}$ is the relational prediction from the attribute pathway. A high value indicates that the two views of the node cannot be reconciled under the normality codebook.

---

## 7. Algorithm Summary

```
Algorithm: NoCo Training

Input: Graph G = (V, E, X), labeled normals V_N, labeled anomalies V_A
Output: Codebook C, anomaly scores s(v) for all v ∈ V

1. Initialize codebook C = {e_1, ..., e_K} randomly
2. Initialize decoders Dec_x, Dec_r
3. Compute structural roles c_v for all v ∈ V

4. for epoch = 1 to T:
5.     // Forward pass
6.     for each node v:
7.         z_v ← argmin_k ||x_v - Dec_x(e_k)||^2   // Hard assignment (or Gumbel-Softmax)
8.         L_assign(v) ← -log P(z_v | c_v)
9.         L_attr(v) ← ||x_v - Dec_x(e_{z_v})||^2
10.        L_rel(v) ← ||r_v - Dec_r(e_{z_v}, x_v)||^2
11.        L_agree(v) ← cross-view disagreement
12.        L(v) ← L_assign + L_attr + L_rel + λ·L_agree
13.
14.    // Anti-code loss on known anomalies
15.    for each a ∈ V_A:
16.        L_anti ← max(0, L(v_normal_same_role) + δ - L(a))
17.
18.    // Update with total loss
19.    L_total ← mean(L over V_N) + α·L_anti + γ·L_reg
20.    Update C, Dec_x, Dec_r via gradient descent
21.
22. // Inference
23. for each node v:
24.     s(v) ← (L(v) - μ_role(v)) / σ_role(v) + β·Δ_cross-view(v)
25.
26. return C, {s(v)}
```
