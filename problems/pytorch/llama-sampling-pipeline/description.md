# 采样：llama.cpp 全管线阶段顺序装配

## 题目描述

单个采样算子（top-k、top-p、min-p、DRY、重复惩罚……）在本合集里各自成题；但真实推理引擎里它们**串成一条链**，而链的**阶段顺序**本身就是行为：llama.cpp 把温度放在**全部截断之后**（截断概率由未过温度的 logits 归一化得到），HuggingFace 则把温度放在最前——同一个 top-p 在两种口径下存活集合不同。本题要求你把 llama.cpp 的完整采样管线按固定顺序装配起来：

\[\text{bias(+banned)}\to\text{penalties}\to\text{dry}\to\text{top-k}\to\text{top-p}\to\text{min-p}\to\text{xtc}\to\text{temp}\to\text{dist}\]

并逐步记录**每个阶段执行完后的存活集合**。随机性以显式入参注入（`u_xtc` 驱动 xtc 的掷币、`u` 驱动最终逆变换采样），保证判题确定性。CPU float32、全程无随机数调用。

## 规则

- 完整签名、参数名与默认值只写在 starter 的注释里，那是唯一口径。`history` 是已生成 token 序列（list），其余标量/张量含义同 starter。
- **返回三元组 `(masks, probs, chosen)`**：
  - `masks`：bool 张量 (8, V)。行下标固定 `0=bias 1=penalties 2=dry 3=top-k 4=top-p 5=min-p 6=xtc 7=temp`，第 \(i\) 行 = 第 \(i\) 个阶段执行完后的存活集合；被 `banned` 的 token 自第 0 行起即 `False`。
  - `probs`：(V,) float，dist 步使用的最终分布——存活集合上归一化、其余位置为 0。
  - `chosen`：标量 int64。`u` 在 `probs` **原顺序** CDF 上逆变换（前缀和**首次 \(\ge u\)** 的下标，取等即命中）；兜底取存活集合在降序里最后一个。
- **各阶段语义**（窗口类参数统一口径：`last_n<0` 取全 history、`=0` 关闭、`>0` 取末尾 \(n\) 个）：
  1. **bias**：`bias_tokens` 逐条把对应 logit **累加** `bias_values`（越界 id 忽略）；`banned` 强制掩码（不改 logit）。
  2. **penalties**：窗口内计数；对每个存活且计数 \(c>0\) 的 token：logit \(x\le0\) 时乘 `penalty_repeat`、\(x>0\) 时除 `penalty_repeat`，再减 \(c\cdot\text{penalty\_freq}+\text{penalty\_present}\)。`penalty_repeat=1, freq=0, present=0` 时整体短路。
  3. **dry**：在**逆序窗口**上对每个偏移 \(k\) 求与开头的最长公共前缀（重复段必须贴着窗口末尾），长度 \(rl\ge\) `dry_allowed_length` 才登记，每 token 只记**严格更长**的重复；命中 token 的 logit 减 `dry_multiplier · dry_base^(rl−al)`。`multiplier=0`、`base<1` 或窗口为空时关闭。
  4. **top-k**：`top_k>0` 且存活数 \(>k\) 时，按 logit 降序（并列取下标小者）保留前 \(k\)。
  5. **top-p**：`top_p<1` 且存活数 \(>1\) 时，在**当前存活集合**上归一化的概率里降序累积，首次 \(\ge p\) 且已保留数 \(\ge\) `min_keep` 处截断。这份概率在 `temp_first=False` 时**未过温度**。
  6. **min-p**：`min_p>0` 时 logit 域地板 = 最大 logit \(+\ln(\text{min\_p})\)，降序扫描到首个**低于**地板且已保留数 \(\ge\) `min_keep` 的位置为止。
  7. **xtc**：`xtc_probability>0`、`xtc_threshold\le0.5`、存活数 \(\ge2\) 且 `u_xtc\le\text{xtc\_probability}` 时触发：概率 \(\ge\) 阈值的**头部前缀**（降序）里，除**最后一个**达标者之外全部砍掉（尾部保留），砍完存活数仍须 \(\ge\) `min_keep`。
  8. **temp**：`temp>0` 时存活 logit 逐个除以 `temp`；`temp\le0` 时贪心——只留最大 logit 的一个 token。**执行时机**由 `temp_first` 决定：`False`（llama.cpp 口径）在 xtc 之后、dist 之前；`True`（HF 口径）紧跟 dry、先于全部截断，此时 top-p/min-p/xtc 看到的概率已过温度，且第 7 行快照是"temp 刚执行完、截断还没跑"的状态。
- 每个截断阶段的 `min_keep` 是**兜底**：宁可少砍/不砍，也不能把存活集合砍到 `min_keep` 以下（top-k 除外，llama.cpp 口径它不受 `min_keep` 保护）。
- 全程 float32、CPU、确定性；不许调用现成采样器。

## 示例

`logits=[5,4,3,2,1,0,-1,-2,-3,-4]`，history `[3,4,5,3,4,5]`，`bias={2:+0.7}`，`banned={0}`，`penalty_repeat=1.1, penalty_freq=0.2, penalty_present=0.1, penalty_last_n=-1`，`dry_multiplier=0.8, dry_base=1.4, dry_allowed_length=2, dry_penalty_last_n=-1`，`xtc_probability=0, xtc_threshold=0.05`，`min_keep=1`，`top_k=8`，`top_p=0.6`，`min_p=0`，`temp=0.5`，`temp_first=False`，`u_xtc=0.5`，`u=0.55`。

- 第 0 行：token 0 被 ban → 存活 9 个；token 2 的 logit 变 3.7。
- 第 1 行：tokens {3,4,5} 各出现 2 次，正 logit 除以 1.1 再减 \(2\cdot0.2+0.1=0.5\)：token 3 → 1.318，token 4 → 0.409，token 5 → −0.5。
- 第 2 行：DRY 在逆序窗口上发现 token 3 贴尾重复长度 3 ≥ 2 → 减 \(0.8\cdot1.4^1=1.12\) → token 3 = 0.198。存活仍 9。
- 第 3 行：top-k=8 砍掉最低的 token 9 → 存活 8。
- 第 4 行：未过温度的存活概率降序累积（token 1 ≈0.552、token 2 累计 ≈0.961）→ 首次 \(\ge0.6\) 保留 2 个。
- 第 5/6 行：min-p=0、xtc 关闭 → 存活集合不变。行和 `[9, 9, 9, 8, 2, 2, 2, 2]`。
- 第 7 行：temp=0.5 把两个存活 logit 变 8 与 7.4，最终分布 `probs[1]=0.645656, probs[2]=0.354344`，其余为 0；`u=0.55` 落在 token 1 的 CDF 段内 → `chosen=1`。

```
masks 行和: [9, 9, 9, 8, 2, 2, 2, 2]
probs:      [0.0, 0.645656, 0.354344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
chosen:     1
```

## 约束

- `masks` 用 `torch.equal` 比对、`probs` 用 `atol=rtol=1e-5`、`chosen` 用 `torch.equal`
- 两个判别口径评测直接断言：**同一参数下** `temp_first=False` 与 `True` 的 top-p 存活集合不同（llama 的截断看到的是未过温度的概率，高温不放大截断），且 llama 口径下 temp 行的存活集合包含于 xtc 行（temp 是最后一个截断）；**所有旋钮关掉**时 `probs` 等于原 logits 的 softmax
