# KV Cache：Q-Filter 学习滤波器负点积打分

## 题目描述

Q-Filter（Zang et al., 2025, [arXiv:2503.02812](https://arxiv.org/abs/2503.02812)；kvpress `NVIDIA/kvpress` 的 `kvpress/presses/qfilter_press.py` 中 `QFilterPress`）抛弃一切启发式：为**每个 (层, kv head) 预训练一个滤波器向量** \(f\)（长度 = head_dim），token 的分数定义为

\[ \text{score}_t = -\, f \cdot k_t \]

即 key 在滤波器方向上的投影**越小**（越负相关）分越高、越优先保留。滤波器不来自注意力也不来自范数——它是在下游 QA 任务上端到端学出的"重要性方向"，随模型发布（kvpress 从 HF hub 加载 `nthngdy/<model>_qfilt`），本题直接把它作为输入张量给出。

## 规则

输入：`q_filters` 形状 \((N, H, D)\)——每层每 kv head 一个滤波器；`keys` 形状 \((N, H, L, D)\)——已完成 RoPE 的缓存 key；标量 `compression_ratio` \(r\in[0,1)\)。

1. 逐 \((n, h, t)\)：\(\text{score} = -\sum_{d} \text{q\_filters}[n,h,d]\cdot \text{keys}[n,h,t,d]\)（点积后取负，无 \(\sqrt{D}\) 缩放）。
2. \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\)。
3. 每个 \((n,h)\) 行按分数取 top-\(n\_kept\)（分大者优先，**平局取位置小者**），保留位置升序输出。

返回两个张量：`scores`（float32，\((N, H, L)\)）与 `kept`（int64，\((N, H, n\_kept)\)）。

## 示例

```
N=2, H=1, D=2, L=4, r=0.5 → n_kept = 2

layer0: f = [1.0, -2.0]
        keys = [[0.5, 1.0], [0.0, -1.0], [2.0, 0.5], [-1.0, -0.5]]
        dot  = [-1.5, 2.0, 1.0, 0.0]      → scores = [1.5, -2.0, -1.0, -0.0]
        top-2 → kept = [0, 3]
layer1: f = [-0.5, 0.25]
        keys = [[1.0, 1.0], [2.0, 0.0], [0.0, 4.0], [-1.0, 2.0]]
        dot  = [-0.25, -1.0, 1.0, 1.0]   → scores = [0.25, 1.0, -1.0, -1.0]
        top-2 → kept = [0, 1]
```

## 约束

- CPU float32；\(N,H,D,L\ge1\)；\(0\le r<1\)。
- 分数是**负点积**：与滤波器同向的 key 得低分被驱逐，反向的得高分保留——符号是算法本体，漏掉负号会得到相反的选择。
- 滤波器为零向量时该 (层, head) 的所有分数为 0，按平局规则保留最左 \(n\_kept\) 个位置。
- 平局一律位置小者优先；输出升序。
