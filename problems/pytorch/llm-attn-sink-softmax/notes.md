# 详解：SoftMax1 的 sink 分母

## 出处

- StreamingLLM: Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024, arXiv **2309.17453**。第 3.3 节提出 **SoftMax1**：

  \[
  \text{SoftMax1}(x)_i=\frac{e^{x_i}}{1+\sum_j e^{x_j}}
  \]

  论文的原话是这一项等价于"往 keys 里塞一个 logit 恒为 0 的 token"；他们把它作为缓解 attention sink 的基线（实测能降低却无法完全消除对首 token 的依赖，因此最终方案仍是显式保留 sink token 的 KV）。
- 与库内相邻题的边界：`kv-streaming-llm-sink` 是 kvpress 的**缓存驱逐打分**（二值保留表），`nu-online-softmax-rescale` 是**分块 softmax 的 (max, sum) 重标定合并**；本题只做 softmax 本身——分母多一项、行和小于 1 的前向归一。三者没有公式重叠。

## 核心与易错点

1. **sink 必须参与取最大**。每行参考值取 \(r=\max(\max_{\text{保留位}} x, \texttt{sink\_logit})\)。少算 sink 时有两类翻车：整行被屏蔽时 `max` 得到 \(-\infty\)，`exp(0-\infty)` 与整体归一都会失效；整行远低于 sink（如 `[-1000, -999]`、sink 0）时 \(e_{\text{sink}}=e^{999}\) 直接上溢成 inf，权重变 NaN。
2. **屏蔽位既不进 max 也不进分母**，输出必须严格是 `0.0`，而不是"很小的数"。实现上先把屏蔽位置换成行参考值再 `exp`，最后用 `where` 置零，既避免 `exp(大正数)` 也能保证精确零。
3. **缺口就是 sink 权重**。`weights.sum(-1) + sink_weight = 1` 是硬约束；把 `weights` 再按行和归一（"补回 1"）是本模型最典型的错解，等于悄悄退回普通 softmax。
4. **整行屏蔽的约定**：`weights` 全 0、`sink_weight` 为 1，不返回 NaN。`torch.softmax` 在这种行上给 NaN，本题故意取"sink 兜底"，与 `nu-online-softmax-rescale` 里对 ±inf 行的工程化约定一致。
5. **float64 中间量**。`(1e5, 1e5 - 1e-3)` 这种"大偏置 + 极小差"在 float32 下会被量化成 `(1e5, 1e5)`，权重直接退化为 0.5；float64 才能保住 3 位有效差。
6. **sink_logit 只是整体缩放**：行内任意两个权重的比值与 sink_logit 无关，只由分数差决定；sink_logit 越大，所有可见权重同步变小、sink 权重单调变大。sink_logit = 0 就是论文的 zero token，取大负数则退化为普通 masked softmax。
7. **mask 广播**：`(S,)` 的 mask 要能作用在 `(B, H, S)` 上，等价于显式 `expand`；但屏蔽仍按行进行，不同 batch 行可以有不同的保留集合。

## 对拍口径

- `_oracle` 是逐行标量 numpy 重算：对每行先扫保留位求最大值，再逐个算 `math.exp`，被屏蔽位置**连 exp 都不调用**，与向量化解法没有共享代码。
- 独立等价性：把 `sink_logit` 当作多拼一列的 logit，直接调 `torch.softmax`（float64）对拍，前 S 列必须是 `weights`、最后一列必须是 `sink_weight`——这正是论文"多加一个零 token"的说法。整行屏蔽时 `torch.softmax` 会给 NaN，所以该检查要求每行至少保留一位，NaN 约定单独由 `check_all_masked_rows` 断言。
- 结构化断言：行和缺口、sink 比值不变性、sink_logit 单调性、掩码广播等价、sink 极端参数量级、float64 精度探针；另有 17 种非法输入（scores 非张量/整型/bool/0 维/空尾维/NaN/±inf，mask 非张量/float/不可广播，sink_logit 为 bool/str/None/NaN/±inf）必须抛 ValueError。
- 11 个 `check_*` 全部通过；对参考解做过变异测试（sink 不参与 max、分母漏 sink、屏蔽位变 1、内部降 float32、忽略 mask、sink_weight 恒 1）均被至少一个检查抓住。

## 复杂度

- 时间 \(O(\text{元素数})\)，空间 \(O(\text{元素数})\)；只有一次最大值扫描、一次逐元素 `exp` 与一次求和，没有 \(O(S^2)\) 的结构。
