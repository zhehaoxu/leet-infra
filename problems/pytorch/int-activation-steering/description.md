# Activation steering：向量注入与方向账

## 题意

Activation steering（表示工程 / steering vector，回源 `nrimsky/steering-vectors`）在推理时把一个方向向量按系数加进某些层的残差流，从而把生成行为拉向目标方向。本题把注入过程做成确定性账本：给定逐层隐态 `hiddens [L, T, D]`、steering vector `s [D]`、注入序列 `(alphas[k], layer_indices[k])` 与可选的预归一化，输出注入后的隐态与每层每 token 相对 `s` 的方向账（注入前后余弦、投影系数、每层累计注入范数）。

## 求解

```
s_eff = s/|s|（normalize=True）或 s
按 k = 0..K-1 顺序：steered[layer_indices[k]] += alphas[k] * s_eff   # 同层累加
账本（全部 float64）：
  layer_delta[l] = |Σ_{k: idx=l} α_k| · |s_eff|     # 未注入层为 0
  cos_before[l][t] = cos(h, s)      # 用原始 hiddens，与注入顺序无关
  cos_after[l][t]  = cos(h', s)
  proj_coef[l][t]  = (h·s)/(s·s)    # 使 h' ⊥ s 的 α；注入 −proj_coef 即消去 s 方向分量
```

关键点：同层多次注入是**累加**不是覆盖；`steered` 必须是副本（不得原位修改入参，float64 输入下 `.double()` 不拷贝）；`cos_before` 必须取自原始隐态；`proj_coef` 的分母是 `|s|²` 不是 `|s|`。

## 数据规模

- 1 ≤ L ≤ 8，1 ≤ T ≤ 8，1 ≤ D ≤ 32，1 ≤ K ≤ 8。
- 形状不匹配、`s` 为零向量、层号越界/非整数、alpha 非有限/bool、K=0 → `ValueError`。
