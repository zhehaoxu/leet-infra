# 详解：因果 LM 移位标签

## 出处
标准 GPT 式自回归目标（Radford et al. 2018；HF `CausalLM` 在内部对 `labels` 做 `shift_labels = labels[:, 1:]` 再与 `logits[:, :-1]` 对齐、`ignore_index=-100`）。本函数把「左移 + 掩码」显式化，是 SFT pipeline 里 `label = next_token if supervised else -100` 的规范写法。

## 核心与易错点
1. **看后继位的掩码**：`labels[b,i]` 对应「预测 \(i+1\)」，因此**判定用 `supervised_mask[b, i+1]`**（下一个 token 是否是被监督目标），而不是 `i` 本身。写成 `mask[i]` 是最常见的错位。
2. **末列恒 -100**：\(i=L-1\) 无行内后继，必须 ignore；不能把 batch 里下一行的 `input_ids[b+1,0]` 接过来当目标（跨样本泄漏）。
3. **`ignore_index` 可配**：默认 `-100`，按参数填；不要用 0 或 pad 顶替，会与真实 token 混。
4. **值来源**：正位填的是 `input_ids[b, i+1]`（被预测的 token），不是 `i`。

## 对拍口径
参考解「双层循环，`if mask[b,i+1]>0: labels[b,i]=ids[b,i+1]`，末列保持 ignore」。判题 `_oracle` 用**逐行 python list** 独立构造：复制整行为 `-100`，再对每行的 `1:` 切片按 `mask[1:]` 选择回填 `ids[1:]`（等价于「先移位再按后继掩码挑选」），结构与参考解不同、逐元素一致。另设「全 0 掩码→全 ignore」「全 1 掩码→纯移位、末列 ignore」「example」「多行末列不跨行」断言。

## 复杂度
\(O(B L)\)。
