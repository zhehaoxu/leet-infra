# Transformer 参数量估算

## 题目描述

面试常要求「不看代码、按结构估出模型有多少参数」（GPT-2/GPT-3 是多少亿参数怎么来的）。
本题按一个标准 decoder-only Transformer 的结构，从每层维度精确清点参数量。

## 规则

- 实现 `solve(layers, hidden, ffn, vocab, tied_embedding=True)`，返回参数总数（整数）。
- **每个 Transformer block 的参数量**（矩阵按 `in×out` 计，含 bias，不含位置编码）：
  $$ P_{\text{layer}} = \underbrace{4h^2}_{\text{attn } q,k,v,o\ (W_{\text{proj}}: h\to h)} + \underbrace{2hf}_{\text{FFN } h\to f,\ f\to h} + \underbrace{4h}_{\text{2× LayerNorm（weight+bias，各 } h\text{）}} $$
- **词嵌入** `vocab × hidden` 记一份；输出 LM head 也是 `vocab × hidden`：
  - `tied_embedding=True`：输入/输出共享同一张嵌入表，只计 `vocab·h` 一次；
  - `tied_embedding=False`：再额外计一份 `vocab·h`（不共享的输出投影）。
- 末尾再加一个 **final LayerNorm**：$2h$（weight+bias）。
- 合计：
  $$ \text{total} = \text{layers}\cdot P_{\text{layer}} + \text{vocab}\cdot h + (\text{tied?}\,0:\text{vocab}\cdot h) + 2h $$
- 头数 `heads` 不改变参数量（MHA 的 $q,k,v,o$ 恒为四个 $h\times h$），故本题不需要 `heads`。

## 示例

```
输入：layers=12, hidden=768, ffn=3072, vocab=50257, tied_embedding=True
  P_layer = 4*768² + 2*768*3072 + 4*768 = 2,359,296 + 4,718,592 + 3,072 = 7,080,960
  layers 项 = 12 * 7,080,960 = 84,971,520
  嵌入项    = 50257 * 768 = 38,597,376
  final LN  = 1,536
  total     = 84,971,520 + 38,597,376 + 1,536 = 123,570,432   （≈123.6M）
输出：123570432
```

## 约束

- 全部为正整数；`tied_embedding` 为布尔
- 本模型**不含可学习位置嵌入**（`max_seq × h`）与 dropout 等无参数项；若要计入请自行另加，本题口径固定
- 返回**整数**参数个数（Python int）
