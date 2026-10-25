# KV Cache：FINCH 因果归一化窗口注意力打分

## 题目描述

SnapKV 用观察窗口（最近若干条 query）对每个上下文 key 收到的 attention **取平均**来打分。FINCH（Zhu et al., 2024, [arXiv:2407.09112](https://arxiv.org/abs/2407.09112)；kvpress `NVIDIA/kvpress` 的 `kvpress/presses/finch_press.py` 中 `FinchPress`，`normalize_scores=True` 路径）注意到一个偏差：**窗口里每条 query 的因果可见 token 数不同**——最老的窗口 query 只给少数 key 投票，最新的那条几乎看遍全场；softmax 分母不同，不同 query 行的 attention 数值尺度根本不可比。FINCH 在跨 query 平均**之前**，把第 \(i\) 行（从窗口最老一条数起）乘以该行可见的 token 数 \(C+i\)（\(C\) 为上下文长度），把「按行平均」修正为尺度一致的加权累计，再做 GQA 组内平均；最后观察窗口本身以全局最大分 +1 保底（窗口永不被剪）。

## 规则

输入：`attn` 形状 \((H_q, W, C)\)——\(W\) 条窗口 query 对 \(C\) 个上下文 token 的已 softmax 注意力权重，**行 0 是最老的窗口 query（全局位置 \(C\)）**，行 \(i\) 的全局位置为 \(C+i\)；`num_kv_heads` \(=H\)（\(H_q\) 整除 \(H\)，组数 \(G=H_q/H\)）；`compression_ratio` \(r\in[0,1)\)。

1. 行缩放：对每条 query head、每个窗口行 \(i\in[0,W)\)，整行乘以 \((C+i)\)。
2. 窗口维取平均（除以 \(W\)）→ \((H_q, C)\)。
3. GQA 分组取平均：head \(h\) 的分数来自 query heads \([hG, hG+G)\) 的均值 → \((H, C)\)。
4. 右侧拼接 \(W\) 个 pad 位：pad 值 = **所有 \(H\) 行 \((H,C)\) 分数的全局最大值 + 1**，得 `scores` 形状 \((H, L)\)，\(L=C+W\)。前 \(C\) 列是上下文位，后 \(W\) 列是窗口位。
5. \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\)；每行按 `scores` 取 top-\(n\_kept\)（分大者优先，**平局取位置小者**），保留位置升序输出。

返回两个张量：`scores`（float32，\((H, L)\)）与 `kept`（int64，\((H, n\_kept)\)）。

## 示例

```
Hq=4, W=2, C=3, num_kv_heads=2, G=2, r=0.4 → L=5, n_kept=3

query head 0: rows×[3,4]  [[1.2,0.6,1.2],[0.4,2.0,1.6]] → 行均值 [0.8,  1.3,  1.4]
query head 1: rows×[3,4]  [[0.9,0.9,1.2],[2.4,0.8,0.8]] → 行均值 [1.65, 0.85, 1.0]
kv head 0 = mean(head0, head1) = [1.225, 1.075, 1.2]
（另两个 query head 同理得 kv head 1 = [1.3125, 0.9125, 1.275]）
全局 max = 1.3125 → pad = 2.3125

scores = [[1.225,  1.075,  1.2,   2.3125, 2.3125],
          [1.3125, 0.9125, 1.275, 2.3125, 2.3125]]
kept   = [[0, 3, 4],
          [0, 3, 4]]
```

## 约束

- CPU float32；输入 `attn` 已经 softmax 且只含窗口 query 对上下文 key 的分量（窗口内部相互可见性由上游处理）。
- 行乘数是 \(C+i\)（\(i\) 从 0 起），即**不含 query 自身位置**的可见 key 计数——照抄上游实现，不是 \(C+i+1\)。
- pad 用的是跨全部 \(H\) 行的全局最大值，不是逐行最大值。
- \(r=0\) 时全部保留；\(W=1\) 时行缩放退化为统一常数（不改变上下文内排序）。
- 所有平局按位置小者优先；输出升序。
