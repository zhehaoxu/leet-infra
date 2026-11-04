# PEFT：Lily 跨层互联专家组合与路由

## 题目描述

**Lily**（Low-Rank Interconnected Adaptation across Layers, arXiv:2407.09946，ACL 2025 Findings，官方仓 `yibozhong/lily`）拆掉 LoRA 的逐层 A-B 配对：低维投影器 **LP** 留在层组局部（每 stride 层换新），高维投影器 **HP** 从各层剥离为**模型级共享 expert 池**（\(n_e\) 个 \((hidden, out)\) 矩阵），每个 LP 连接全部 HP。层间互联让梯度投影不再被单一固定投影矩阵钉死在低秩子空间——以相同/更少参数获得**更高秩**的 \(\Delta W\)。

`lily_adapter.forward` 的确定性链（官方 `natural-language-understanding/lily.py`）：

```
hidden = non_linear(lp(x))                 # MLP 位 ReLU、attention q/v 位 Identity
logits  = router(hidden)                   # nn.Linear(hidden, ne, bias=False)
probs   = softmax(logits, dim=-1)          # [B, N, ne]，数据依赖
p_mean  = probs.mean(dim=(0, 1))           # batch+token 维全局平均 -> (ne,)
combined_hp = einsum("e,eio->io", p_mean, hps)
delta   = hidden @ combined_hp
```

router 是**纯 dense softmax 加权**（无 top-k、无噪声），锚点全确定。`monoscale` 变体去掉 router：`combined_hp = sum(hps, 0) * (1/ne)`。`set_lily` 集成账：`stride = n_layers // ne`、`idx % stride == 0` 处放新 LP（层组共享）；HP `torch.zeros(ne, dim, out)` 零初始化（训练起点 \(\Delta W=0\)）。

## 规则

- `mode="router_probs"`：hidden 与 router 权重入参，返回 softmax 后 \((\dots, n_e)\) 分布（每 token 概率和 1）。
- `mode="expert_mix"`：`probs_mean (n_e,)` 与 `hps (n_e, hidden, out)` 入参，返回组合 HP \((hidden, out)\)。
- `mode="forward_delta"`：全链，`mlp=True` 时 hidden 过 ReLU；`probs_mean` 是 batch+token **双轴**平均。
- `mode="monoscale_delta"`：均匀变体，\(1/n_e\) 缩放。
- `mode="lp_groups"`：返回 `{"stride", "lp_group", "num_lp"}`——尾组收纳余数层（如 \(n=5, n_e=2\)：stride=2、组 [0,0,1,1,2]、3 个 LP）。
- LP/router 权重与 HP 全部显式入参；全程 float64。

## 示例

\(in=8, hidden=4, out=6, n_e=3\)：

```
forward_delta(x, lp_w, router_w, hps) -> (2,3,6)
  = hidden @ Σ_e p_mean[e]·hps[e]，hidden = x @ lp_wᵀ
monoscale_delta(x, lp_w, hps) -> hidden @ (Σ_e hps[e] / 3)
lp_groups(12, 4) -> {stride: 3, lp_group: [0,0,0,1,1,1,2,2,2,3,3,3], num_lp: 4}
```

## 约束

- softmax 输出每 token 概率和 1；不同 token 的 expert 分布可不同（数据依赖）。
- forward 与显式展开 \(\sum_e \bar p_e\,(hidden\cdot hps_e)\) 数学等价（测试互证）。
- `lp_groups` 满足：num_lp = 不同组 id 数 = \(\lceil n/st\rceil\) 向上取整语义由尾组收纳实现。
