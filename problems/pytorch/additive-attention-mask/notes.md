# 详解：4D 加性注意力 mask 构造

## 出处

- 加性 mask 与 finfo.min 口径：HuggingFace transformers 的 masking 工具与 flash-attn 的 add-to-logits mask 约定——遮蔽值取 \(\mathrm{finfo(dtype).min}\) 而非 \(-\infty\)（全遮行加 −inf 后 softmax 出 NaN，加 finfo.min 只会均匀摊分）。本题 starter docstring 把这条理由写成了契约的一部分。
- 因果背景：Vaswani et al., *Attention Is All You Need*（arXiv:1706.03762）——自回归解码中位置 \(i\) 不可见 \(j>i\)；本题是该约束在"KV 缓冲左侧已垫 kv_offset 个旧 token"时的加性 mask 推广。

## 完整推导与下标约定

**两套坐标**：\(i\) 是查询窗内 0-based 位置（0..q_len−1），\(j\) 是 KV 缓冲内的**绝对**位置（0..kv_len−1），kv_offset 是 KV 缓冲左侧已缓存的旧 token 数。解码时"当前查询行"恰能看全从缓存起点到自己的所有 token，对角线因此落在 \(j=i+\text{kv\_offset}\)。带约束用**相对坐标** \(r=j-\text{kv\_offset}\)：\(i-r<\text{window}\)，即从当前查询往回最多 window−1 个 KV 位置、对角线恒可见。j 保持绝对语义，因果/窗口条件才能与"缓冲左侧垫了旧 token"的布局自然复合。

**四条约束的代数形状**：causal 给出上界 \(r\le i\)，window 给出下界 \(r>i-\text{window}\)，两者夹出带宽为 window 的对角带；kv_lens / seq_lens 是与 (i,j) 结构无关的列裁剪 / 行裁剪。全开时可见集 = 对角带 ∩ 有效列 ∩ 有效行——四个条件 AND，缺哪个去哪个，彼此不做任何豁免。

**kv_offset=0、纯 causal 时塌回三角**：\(j\le i\) 正是下三角含对角，与 torch.tril 的布尔模式逐位一致——`causal-mask` 一题的上三角掩码就是本 mask 的一个特例。kv_offset=k 时整体右移 k 列（每行可看集恰为 \(j\le i+k\)），这正是"KV 缓冲里已垫 k 个旧 token、查询窗内第 i 行"的解码步进图景：同样的 i，能看到更靠右的绝对位置。

**finfo.min 的数值账**：fp32 的 finfo.min ≈ \(-3.4028\times10^{38}\)。logits 量级 O(1..100)，相加不会溢出（fp32 上限 3.4e38）；被遮位与未遮位的分数差 \(\ge10^{38}\)，softmax 下 \(e^{-10^{38}}\) 下溢为 0——被遮位概率严格为 0，与 −inf 的可见位结果在数值上不可区分。**唯独全遮行不同**：−inf 行的 softmax 是 \(-\infty/-\infty\) 形态，输出 NaN；finfo.min 行的 logits 加完仍是有限大负数，softmax 给出 \(e^{x_i-\max}/Z\) 的有限均匀摊分（\(1/L\)）。这就是"全遮行必须不 NaN"的全部机制，也是 HF / flash-attn 不用 −inf 的原因。

**window 与 causal 的独立性**：causal=False、window=w 时只施加 \(i-r<w\)——未来位置不受限，只限制"往回看得太远"，得到一条以对角线为中心的双向局部带。判题专门覆盖 causal=False + window + kv_offset 的组合。

**batch 广播语义**：B = max(len(查询行 lens), len(KV 列 lens), 1)；长度 1 的序列沿 batch 复制（编码器对整个 batch 共享同一裁剪时省一个维度）；两者都 >1 时必须等长。长度既不是 1 也不是 B、或都 >1 且不等 → ValueError。

## 核心与易错点

1. **j 是绝对坐标、i 是相对窗内坐标**：causal 与 window 都要把 kv_offset 放在 j 侧（\(j\le i+\text{kv\_offset}\)、\(i-(j-\text{kv\_offset})<\text{window}\)）；写成 \(j\le i\) 或 \(i-j<\text{window}\) 会在 kv_offset>0 的对拍里全线崩掉。
2. **遮蔽值随 dtype 变**：fp32 ≈ \(-3.4028\times10^{38}\)、fp64 ≈ \(-1.7977\times10^{308}\)，一律 \(\mathrm{finfo(dtype).min}\)；写死 −inf 或写死 fp32 的 min 都过不了 float64 探针。
3. **可看处恰为 0.0**：mask 是加到 logits 上的，可看处加 0 才是恒等；写成 1.0（布尔思路）或夹带其他 bias 值都会挂逐位 equal。
4. **window ≤ 0 拒绝；causal=False 时 window 仍生效**：两者是独立开关，"window 只在 causal 下有意义"是错误理解。
5. **pad 行/列是整行/整列全遮**：\(i\ge\text{seq\_len}_b\) 的行不看任何列（对角线也不例外），\(j\ge\text{kv\_len}_b\) 的列对所有行不可见；lens 裁剪不与因果/窗口条件互相豁免。
6. **长度 1 广播**：长度 1 的 lens 复制到每个 batch；不要误解为"只作用于 batch 0"。
7. **空维合法**：q_len=0 或 kv_len=0 时返回 (B,1,0,kv_len)/(B,1,q_len,0) 的空张量，不报错。
8. **校验面**：负数/非整数的 q_len、kv_len、kv_offset，window ≤ 0，非法 dtype，lens 负数/越界/长度不匹配——全部 ValueError。

## 与邻近题的区别

- **vs `causal-mask`**：那边在 (L,L) 分数张量上保留三角、上三角置 float("-inf")（kv_offset=0 的特例，且掩的是分数本身）；本题是加 kv_offset 平移 + padding 裁剪 + 窗口带的 4-D **构造**，输出恒为 0.0/finfo.min 两档。kv_offset=0、无 lens 时本 mask 的下三角判定面与那边逐位一致，但值域不同（那边上三角 −inf 且保留下三角原分数）。
- **vs `sliding-window-attention`**：那边实现窗口注意力的**前向计算**（带内 softmax 加权求和）；本题只构造 mask 本身，不做任何注意力运算。
- **vs `alibi-bias`**：ALiBi 是另一种加性 bias 来源（按距离线性衰减的连续偏置，可与本 mask 叠加使用）；本题的加性项只有 0 / finfo.min 两档，是"开关"不是"打分"。
- **vs `kv-cache-update`**：那边管 KV 缓冲的写入（追加 token）；本题管"缓冲已有 kv_offset 个旧 token 时"的读侧可见性，二者是解码步进的写侧/读侧。

## 对拍口径与复杂度

- 独立重算 oracle：三重 for 循环逐 (b,i,j) 显式判定四条规则，lens 长度 1 时按 `bi % len(lens)` 广播——与解法的向量化广播路径完全不同。mask 只有两档取值，判题可直接 torch.equal 逐位比较。10 组全组合对拍（causal × window × lens × kv_offset，含全可看、整 batch 全遮、seq 广播）+ 专项探针：tril 分解、offset=3 平移、window=1 严格对角、大 window 退化、全额 lens 等价、pad 行/列、float64、零维。
- 复杂度：时间与空间均 \(O(B\cdot\text{q\_len}\cdot\text{kv\_len})\)——可见性布尔张量一遍逐条件 AND、一遍 masked_fill；单对 (i,j) 的判定本身 O(1)。
