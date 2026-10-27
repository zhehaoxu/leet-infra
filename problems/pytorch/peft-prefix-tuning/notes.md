# PEFT：Prefix-Tuning 虚拟 KV 前缀注意力——算法详解

## 1. 出处与核心思想

Li & Liang（ACL 2021, arXiv:2103.10385）面对两类昂贵的适配方式：全量微调每任务一份权重；adapter 在层间插串行模块带来额外延迟。Prefix-tuning 的洞察是**注意力本身就是条件注入点**——给每层的注意力额外 \(L\) 个虚拟 token 的 K,V，真实 token 的注意力会自动学会"参考前缀"，从而在生成时被"软提示"。GPT-2 table-to-text、BERT XSum 等任务上用约 0.1% 的参数达到全量微调水平，低数据场景反超。

与 prompt-tuning 的差别（面试常问）：prompt-tuning 只在**输入嵌入层**拼前缀（只影响第 0 层的输入），prefix-tuning 在**每层**注意力都拼独立前缀，表达力更强、实现更重。

## 2. 重参数化 MLP（源码口径）

直接训练 \(P_k,P_v\) 优化不稳（论文 §4.1：重参数化在小数据上明显更稳），所以先训一个小张量、过 MLP 后写回前缀。PEFT `prefix_tuning/model.py` 的 `PrefixEncoder` 重参数形态：

```python
self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
self.transform = nn.Sequential(
    nn.Linear(token_dim, encoder_hidden_size),
    nn.Tanh(),
    nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
)
```

参数账（PEFT 默认 \(h=d\)）：

- embedding：\(L\cdot d\)
- 第一层 Linear：\(d\cdot h+h\)
- 第二层 Linear：\(h\cdot(2Ld)+2Ld\)

合计 \(n_\text{reparam}=Ld+dh+h+2Ldh+2Ld\)（本题 \(h=d\)）；直接形态 \(n_\text{direct}=2Ld\)。**注意重参数形态参数更多**——收益是优化稳定性，不是省参数（易错点）。`check_param_counts` 用两组 \((L,d)\) 断言公式，防止硬编码。

## 3. 前向语义（判题考点）

- 拼接在 **K 与 V 两侧同时**发生，且**前缀在前**：\([P_k;K]\)、\([P_v;V]\)。因果语言模型里真实 token 位置平移 \(L\) 后每个位置都能看到全部前缀；本题单轮注意力不涉及因果 mask，但拼接顺序决定 softmax 权重的语义（前缀行对应 \(P_v\) 行）。
- `check_prefix_zero_degenerates`：\(L=0\) 时 \([P_k;K]=K\)，退化为标准注意力 \(\text{softmax}(QK^\top/\sqrt d)V\)，同时 \(n_\text{direct}=0\)、\(n_\text{reparam}=d^2+d\)（embedding 消失、MLP 还在）。
- `check_prefix_dominates`：\(Q\) 与 \(P_k\) 强对齐、与 \(K\) 正交时，输出逼近 \(P_v\)——"虚拟 KV 是可学习的软提示 KV"这个直觉的最小可判验证。

## 4. 与本合集其他题的分工

- `peft-prompt-tuning`：嵌入层拼接 + 每任务 \(L\cdot d\) 参数账（输入侧）。
- `peft-multi-adapter-batch`：LoRA 侧的 batch 内多 adapter 路由。
- 本题：注意力内 KV 前缀 + 两形态参数账。

## 5. 验证方法

`check_example`（\(d=L=S=N_q=1\)）：\(P_k=2,P_v=3,K=5,V=7,Q=1\)，logits \([2,5]\)，attn \(=(3+7e^{3})/(1+e^{3})\approx 6.8103\)、\(n_\text{direct}=2\)、\(n_\text{reparam}=7\)；`check_matches_oracle` 与纯 python softmax 对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
