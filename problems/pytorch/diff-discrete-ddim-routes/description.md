# 离散扩散：D3PM 均匀腐蚀的后验三路路由

## 题目描述

离散状态空间的扩散模型（D3PM）不像连续扩散那样输出噪声或速度，而是直接在每个位置输出词表上的 logits，再从中采样一个「干净 token」\(x_0\)。均匀腐蚀核下，从时刻 \(t\) 回退到更干净的 \(s<t\) 时，后验 \(q(x_s\mid x_t,x_0)\) 有三条**互斥路径**：跳到预测的干净 token、原地不动、跳到一个均匀随机的 token。离散 DDIM 的一步就是在这三条路径上做一次多项采样。

本题要求实现这一步的**确定性核心**：给定 logits 与当前 token，算出每个位置的贪心去噪结果 \(x_0\)、三条路径的质量（归一化后），以及可选的 leave-one-out（LOO）去噪修正。采样本身交给 RNG，不在本题范围内。

## 规则

输入 `logits` 形状 \((B, L, K)\)（每位置 \(K\) 类词表），`tokens` 形状 \((B, L)\) 的整数张量（当前 block 的 token，取值 \([0,K)\)）。设 \(N=`num_inference_steps`\)，时间账按「alpha 越接近 1 越干净」：

\[ \alpha_t = \frac{\text{step\_index}}{N},\qquad \alpha_s = \frac{\text{step\_index}+1}{N},\qquad a = \frac{\alpha_t}{\alpha_s},\qquad b = \alpha_s \]

先取 \(x_0=\arg\max\) 的 logits。**LOO 修正**（`use_loo=True`）在取 argmax 之前把「已观测 token」所在位置的 logit 减去

\[ \delta = \operatorname{log1p}\!\left(\frac{K\,\alpha_t}{1-\alpha_t}\right),\qquad \text{仅当 } 0<\alpha_t<1 \]

记 `same = (tokens == x0)`。三条路径的质量为

\[ m_{\text{clean}} = \frac{b\,(1-a)}{K} + a\,b\cdot\text{same},\qquad m_{\text{stay}} = \frac{a\,(1-b)}{K},\qquad m_{\text{noise}} = \frac{(1-a)(1-b)}{K} \]

**注意这三元组并不总是归一化**：`same` 为假时它等于 \((1-ab)/K\)，为真时恰好等于 1。上游是对三元组除以自身之和再采样的，所以这个除法必须照做，不能省。

返回 `(route_probs, pred_tokens)`：

- `route_probs`：float32，形状 \((B, L, 3)\)，最后一维依次是 clean / stay / noise；
- `pred_tokens`：int64，形状 \((B, L)\)，即（可能经 LOO 修正后的）argmax。

## 示例

\(B=1\)、\(L=2\)、\(K=4\)，`logits` 使 \(x_0=[2,1]\)，`tokens=[[0,1]]`，`N=4`：

```
step_index=1 -> routes = [[0.333333, 0.333333, 0.333333], [0.714286, 0.142857, 0.142857]]
step_index=0 -> routes = [[0.250000, 0.000000, 0.750000], [0.250000, 0.000000, 0.750000]]
step_index=3 -> routes = [[1.000000, 0.000000, 0.000000], [1.000000, 0.000000, 0.000000]]
```

`step_index=1` 时 \(\alpha_t=0.25,\alpha_s=0.5\)：第一个位置 `same=False`，三条路径未归一化各为 \(0.0625\)，归一化后均分；第二个位置 `same=True`，clean 拿到额外的一份 \(ab=0.25\)。`step_index=0` 时 `stay` 恰好为 0；最后一步（`alpha_s=1`）只剩 clean=1，即确定性提交。

LOO 的例子（\(K=4\)、`logits=[[0,1,0,0]]`、`tokens=[[1]]`、`N=4`）：

```
step_index=2          -> pred=1, routes = [[0.900000, 0.066667, 0.033333]]
step_index=2, use_loo -> pred=0, routes = [[0.500000, 0.333333, 0.166667]]
```

\(\delta=\log(5)\approx1.609\)，把观测 token 的 logit 从 \(1.0\) 压到 \(-0.609\)，argmax 因此翻到 0 号 token，`same` 从真变假、clean 质量随之掉下来。

## 约束

- `logits` 必须是三维张量、`tokens` 必须是二维整数张量且形状等于 `logits` 的前两维；`tokens` 为浮点时抛 `ValueError`。
- `tokens` 的取值必须落在 \([0,K)\)，越界或负数抛 `ValueError`；\(K\ge 2\)。
- `step_index` 为整数（`bool` 不算）且 \(0\le \text{step\_index}<N\)，\(N\ge 1\)。
- `use_loo=True` 时，`step_index=0`（\(\alpha_t=0\)）不做任何平移，结果应与 `use_loo=False` 完全一致。
- 返回的 `route_probs` 每行之和为 1（在 float32 精度内）。
- 计算内部可用 float64，`route_probs` 返回 float32、`pred_tokens` 返回 int64。
