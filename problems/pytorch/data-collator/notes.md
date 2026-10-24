# 数据整理（collate）详解

## 三个张量为什么缺一不可

- **input_ids**：模型唯一的输入。变长没法堆叠成矩形张量做批量矩阵乘，所以补齐到 `L`。
- **attention_mask**：pad 是「假 token」，若不加掩码，softmax 会把注意力分给 pad，稀释真实 token 的表示、污染下一条生成。掩码在 attention 里把 pad 位置的分数置 `-inf`。
- **labels 的 -100**：算交叉熵时 pad 也要排除。`F.cross_entropy(ignore_index=-100)` 会跳过值为 `-100` 的位置——这是 PyTorch 里 `ignore_index` 的**默认值**，所以全行业约定用 `-100` 标「不算 loss」。

## 右补 vs 左补

训练用**右补**（本题）：teacher forcing 下未来位置本就 mask 掉，pad 在尾巴无所谓。**生成/推理常用左补**：自回归要从最后一个真实 token 接着续写，pad 若在后会接在 pad 上。两者 attention_mask 规则一样，只是 pad 落在哪端。本题固定右补，别自作主张左补。

## 两个高频漏点

1. **labels 忘了把 pad 设成 -100**：直接复制 input_ids 当 labels，于是模型被逼去「预测 pad」，loss 里混入无意义项、指标虚高。
2. **attention_mask 与 labels 的 mask 不一致**：两处 pad 判定必须同源（都基于真实长度）。本题用同一 `n` 切前缀保证一致。

## 为什么 dtype 用 long

token id 和 label 要做 embedding 索引 / cross_entropy 目标，必须是整数张量（`torch.long`/int64）。float 化的 id 会报错。

## 关联

- 上游：`bpe-encoding`（文本→子词→id，本把它按 batch 整理）。
- 下游：`causal-language-modeling` 之类需要 label shift + ignore 的 loss 题；`attention-mask`（若有）用这里的 mask 再扩成 4D。
- 判题：整数张量、无随机；`check_example` 用 dump 的三矩阵，另有全等长、单条、自定义 pad/ignore 用例。
