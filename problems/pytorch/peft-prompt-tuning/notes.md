# PEFT：Prompt-Tuning 软提示嵌入拼接——算法详解

## 1. 出处与核心思想

Lester et al.（EMNLP 2021, arXiv:2104.08691, *The Power of Scale for Parameter-Efficient Prompt Tuning*）：prefix-tuning 每层都要插前缀、实现重；prompt-tuning 砍到只剩输入嵌入层的 \(L\) 个可学 token。T5 系列上的关键观察是**规模效应**：模型越大，prompt-tuning 与全量微调的差距越小（T5-XXL 上几乎消失），而每任务只需 \(L\cdot d\) 参数（\(L=20,d=1024\) → 约 2 万参数）。多任务部署时模型只有一份、每任务一张提示表——"提示即任务标识"的服务化范式由此起步。

## 2. 前向与池形态（源码口径）

HF transformers 里 prompt-tuning 的实现（`PromptEmbedding`）：模型 embedding 冻结，前向

```python
inputs_embeds = torch.cat([prompt_embedding.expand(B, -1, -1),
                           embedding(input_ids)], dim=1)
```

提示在前、原嵌入在后，序列长度变 \(L+S_0\)。本题的多任务池形态：所有任务的提示行存一张 \(P\times d\) 的池表，`task_ids[t]` 是任务 \(t\) 的 \(L\) 个行索引，前向即 gather。判题考点：

- `check_gather_indices`：\(X[t,:L]\) 逐行等于 `pool[task_ids[t]]`——索引语义（不是排序、不是连续切片）。
- `check_prepend_order`：\(X[t,L:]=E\)——前缀在前；"嵌入在前"是常见实现 bug，会被钉住。
- `check_param_account`：\(n_\text{params}=T\cdot L\cdot d\)，且换池大小（\(P=6\to 9\)）结果不变——**每任务独占视角**的可训参数与存储形态解耦。面试口径：prompt-tuning 每任务可训参数 \(L\cdot d\)；池+索引是 serving 的存储优化，不改变参数账。

## 3. 与 prefix-tuning 的对比（易混淆点）

| | prompt-tuning | prefix-tuning |
| --- | --- | --- |
| 拼接位置 | 仅输入嵌入层 | 每层注意力的 KV |
| 每层/每任务参数 | \(L\cdot d\)（一次） | 直接 \(2Ld\)（单层），重参数更多 |
| 深层影响 | 仅通过第 0 层传导 | 每层直接注入 |
| 多任务池 | 天然支持 | 每层每任务一套，池化复杂 |

## 4. 与本合集其他题的分工

- `peft-prefix-tuning`：KV 侧前缀 + 两形态参数账。
- `peft-bitfit-count`：另一种极简 PEFT（bias 子集）的参数账。
- 本题：嵌入侧前缀 + gather 路由 + 每任务参数账。

## 5. 验证方法

`check_example` 手算：\(4\times2\) 池、两任务分别取第 0/1 行与第 3/2 行、\(E=[[9,9]]\) → \(X_0\) 头两行是池第 0/1 行、\(X_1\) 是第 3/2 行、尾行都是 \([9,9]\)，\(n=8\)；`check_matches_oracle` 与纯 python 双循环对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
