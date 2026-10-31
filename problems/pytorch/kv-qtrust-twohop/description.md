# KV Cache：Q-Trust 两跳注意力传播打分

## 题目描述

prefill 之后按注意力挑 KV 时，SnapKV 那一套只算「最后 W 个 query（也就是请求本身）对每个 key 的注意力」。Q-Trust 认为这只看了一跳：请求真正关注的**中间 query**，它们的注意力也应该算进 key 的分数。于是把打分拆成两跳，用 query 权重做一次传播。

一跳（`qtrust/reference.py::first_hop`）只算最后 `W` 个 query 的因果注意力，并在 GQA 组内求和：\(s(j) = \sum_{t \in \text{last } W} \sum_{h \in \text{group}} A_h(t, j)\)。这就是一个不做 pooling 的 SnapKV 分数。

query 权重（`query_weights`）：请求段位置的原始权重恒为 1，其余位置取 `s(q)`，整体归一化到 \(\sum_q w(q) = 1\)。那个 1 是**归一化前的原始权重**——请求段通常只有十几个 query，靠人手一个 1 才压得过成百上千个文档 query。

选择性计算（`tau`）：归一化之后 \(w(q) < \tau / N\) 的 query 被剔除（`w` 置 0，它的注意力行根本不算），但请求段的 query 永远保留；`tau = 0` 是不做筛选的全量第二跳。

二跳（`second_hop`）：`score(i) = Σ_{q ≥ i} w(q) Σ_{h∈group} A_h(q, i)`，因果掩码让 \(q < i\) 的项为零。上游注释指出 \(\sum_q s(q) = W \cdot G\)，所以它正比于 \(s(i) + \sum_{q \notin W} s(q) A(q, i)\)，即「请求的直接注意力 + 请求所关注的那些 query 的注意力」。

选 key（`select_keys`）：每个 KV 头恒保最后 `recent` 个 key，剩下的 `budget - recent` 个名额从 `[0, N - recent)` 里按 `score` 取。

三个容易写错的地方：

1. **因果边界在请求段上不放松**：`q` 与 `k` 是同一次 prefill 的 N 个位置，列 \(i\) 只对行 \(t \ge i\) 可见；请求段在末尾，因此它看得见全部 N 个 key，但它自己的行不会被更早的 key 看到。
2. **归一化发生在剔除之前**：被剔除的 query 不是「把权重重新分给活下来的人」，`Σq w(q)` 会小于 1。
3. **请求段恒保**：`keep[:, request_mask] = True` 是在阈值比较之后做覆盖，`tau` 再大也剔不掉请求段的 query。

## 任务

实现 `solve(q, k, request_mask, window=16, tau=1.0, recent=16, budget=32)`：

- `q`：`(N, H, D)` float32/float64，NHD 布局；
- `k`：`(N, KV, D)` 同 dtype，\(H = KV \cdot G\)；
- `request_mask`：`(N,)` bool，请求段；
- `window`、`recent`、`budget`：整数，`tau`：非负浮点数；
- 缩放系数为 \(D^{-1/2}\)，内部一律 float64，返回按输入 dtype。

返回 5 键字典：

- `first_hop`：`(KV, N)`，一跳分数 `s`；
- `query_weight`：`(KV, N)`，归一化并过阈值之后的 `w`（被剔除的位置恰为 0）；
- `kept_query`：`(M,)` int64 升序，**任一** KV 头保留的 query 下标并集——选择性计算真正要跑的注意力行；
- `score`：`(KV, N)`，二跳分数；
- `selected`：`(KV, budget)` int64、每行升序：`[0, N-recent)` 内按分数降序取 `budget-recent` 个（同分取更小下标），再并上 `N-recent … N-1`。

非法输入抛 `ValueError`：非三维、序列长度或 head_dim 不一致、dtype 不同或不是 float32/float64、`H` 不是 `KV` 的正整数倍、`request_mask` 不是 `(N,)` 的 bool、`window`/`recent`/`budget` 不是合法整数、`tau < 0`、`window > N`、`recent > budget`、`budget ≥ N`。

## 示例

```
q = ones(6, 1, 1)             # N=6, H=KV=1, D=1
k = zeros(6, 1, 1)            # 所有 logit 都是 0 → 因果注意力均匀
request_mask = [F,F,F,F,T,T]
window = 2                    # 一跳只看行 4、5
tau = 1.0                     # 阈值 = tau / N = 1/6
recent = 1, budget = 3        # 恒保 [5]，再从 [0,5) 里选 2 个

A(t, i)      = 1/(t+1) for i <= t     # 全同 logit 的因果 softmax
first_hop    = [11/30, 11/30, 11/30, 11/30, 11/30, 1/6]
               # j <= 4 拿 1/5 + 1/6，j = 5 只拿 1/6
raw_w        = [11/30, 11/30, 11/30, 11/30, 1, 1]      # 请求段原始权重是 1
sum(raw_w)   = 104/30
query_weight = [0, 0, 0, 0, 30/104, 30/104]
               # 11/104 = 0.1058 < 1/6 → 前四个 query 被剔除
kept_query   = [4, 5]
score(i)     = 30/104 * (A(4,i) + A(5,i))
             = [11/104, 11/104, 11/104, 11/104, 11/104, 5/104]
selected     = [[0, 1, 5]]    # [0,5) 内五个位置同分，取最小的两个
```

`k` 全 0 让因果 softmax 退化成 `1/(t+1)`，示例因此可以手算到分数；注意 `query_weight` 前四位是 0——它们归一化后是 11/104，差一点点没够到 1/6。

## 约束

- **请求段原始权重是字面的 1**：不是 `s(q)` 的倍数，也不是归一化后的常数。换成「全都用 `s(q)`」会让请求段被文档 query 的数量淹没。
- **阈值是 `tau / N`，分母是总长 N**：不是非请求 query 的数量，也不是 `window`。
- **请求段永不剔除**：阈值比较之后强制覆盖。`tau` 取极大时只剩请求段，此时 `Σw = n_req / Σ_j raw_w(j) < 1`。
- **`kept_query` 是跨 KV 头的并集**：每个头有自己的 `w`；一个 query 只要被任一 KV 头保留，它的注意力行（含组内所有 query 头）就必须算。
- **尾部恒保、不可竞标**：`selected` 的后 `recent` 位恒为 `N-recent … N-1`；`budget = recent` 时没有竞标名额，`selected` 就是这段尾巴。
- **同分取更小的 query / key 下标**：上游 `torch.topk` 不定义同分顺序，本题把它钉成确定性约定，示例与测试都用全同分构造局面。
