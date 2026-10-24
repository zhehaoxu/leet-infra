# 解题说明 · Min-p 采样

## 核心思想

Min-p 不是"选前 k 个"也不是"累积到 p"，而是设**相对于最高概率的地板**：
只保留 $p_i \geq p \cdot p_{\max}$。模型自信时 $p_{\max}$ 大→地板高→砍得猛；
困惑时 $p_{\max}$ 小→地板低→宽松。一个超参自适应置信度。

## 等价判据：logit 域

因为 softmax 单调：
$$p_i \ge p \cdot p_{\max} \iff \text{logit}_i \ge \text{logit}_{\max} + \ln p$$
llama.cpp 实现直接在 logit 域操作，避免先做 softmax。本实现按概率域写，二者数学等价。

## 逆变换采样

同 top-k：在 filtered（原顺序，非保留位置=0）的 CDF 上取首个 $\ge u$ 的位置即可。

## 与 top-p / top-k 的对比

| 方法 | 保留依据 | 对平坦分布 | 对尖锐分布 |
|------|---------|-----------|-----------|
| top-k | 固定个数 k | 太多噪声词仍保留 | 可能砍掉有用候选 |
| top-p | 累积质量 p | 截断核大 | 截断核小 |
| **min-p** | p·p_max 地板 | 地板高→砍更多 | 地板低→保留更多 |

三者互补，llama.cpp 默认管线里三者**串联**：先 top-k 粗筛 → top-p → min-p 精调。
