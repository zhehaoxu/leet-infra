# 扩散：FasterCache 注意力外推广播步（batch 舞步 + 双槽线性外推）

## 题意

FasterCache 在 PAB 的「缓存复用」上再进一步：广播时不是直接用旧注意力输出，而是用**最近两次输出的线性外推** `t_1 + (t_1 − t_2)·weight` 逼近当前注意力状态，并配合 denoiser 侧的 uncond 分支跳过形成 batch 舞步。实现 `solve(attention_output, state, current_timestep, block_skip_range=2, timestep_skip_range=(-1, 681), weight=0.5, is_guidance_distilled=False)`：

```text
# 1. 跳过门控
within = lo < current_timestep < hi                 # 开区间 (严格)
区间外 -> 不跳
区间内 -> skip = iteration % block_skip_range != 0  # 第 0, N, 2N... 次真算
skip 复核: 仅 is_guidance_distilled 或 state.batch_size != 当前 batch 时才真跳
           (cache 只缓存 conditional 半批; denoiser 整批前向时形状对不上, 必须真算)

# 2. 线性外推 (skip 时)
t_2, t_1 = cache  # 最近两次 conditional 输出
t_2/t_1 若 size(0) != 当前 batch, 先切 conditional 半批 [batch:]
output = t_1 + (t_1 - t_2) * weight                # 非 skip 时 output = attention_output

# 3. 缓存维护
cache_output = output
非 distilled 且 output.size(0) == state.batch_size -> chunk(2, dim=0)[1] 取条件半批
cache = [cache[1], cache_output]                   # 首步 [c, c]; iteration += 1
```

返回 `dict(output, computed, state)`，state 三键 `{iteration, batch_size, cache}`（新 dict，纯函数式；batch_size 首调用记录后保持）。

## 关键语义

- **区间只「许可」广播**：端点 t==lo/t==hi（官方默认 −1/681）在区间外，连条件半批调用也恒 compute。
- **batch 失配复核是第二道门**：区间内 + 取模命中还不够——cache 只存 conditional 半批，当前调用若还是整批（batch == state.batch_size）就必须真算。蒸馏模型（无 uncond 分支）豁免此复核。
- **外推参照是前两次条件输出**：cache 条目若意外携带整批（2B），先切 `[batch:]`（conditional 在后一半），不是 `[:batch]`。
- **双槽滚动**：新 `cache = [旧 cache[1], 新条件输出]`——第 N+2 次外推的参照是第 N 次的输出与第 N+1 次的外推近似，不是更早的条目。
- **缓存写入的半批条件**：非 distilled 且 `output.size(0) == state.batch_size`（整批）才 chunk 取后半；条件半批调用（batch 失配时）原样入缓存。
- **官方 `iteration > 0 and ...` 的 `>0` 是死代码**（0%N==0 恒真），判题域不收入。

## 判题域约定

- 官方 hook 挂注意力模块、weight/current_timestep 由 callback 提供；判题域显式入参（weight 缺省 0.5 同官方 callback 缺省）。
- 官方分发层（identifier 匹配）与 767 PAB 同族，不入本题；skip_range 直接入参。
- 官方 cache 支持 tuple 多输出分支；判题域单张量。
- 官方不校验参数；判题域补张量/区间/weight/state 三键等显式校验。

## 例子

非蒸馏完整舞步（整批 2B=4、条件半批 B=2、N=2、区间内、w=0.5）：iter0 整批 compute（缓存切后半 ca）；iter1 条件半批 skip（外推 = ca，两槽同值）；iter2 整批 compute（cache=[ca, cb]）；iter3 条件半批 skip（外推 = cb + (cb−ca)·0.5）；iter4 整批 compute（cache=[approx3, cd]）；iter5 skip（外推参照 approx3 而非 cb）。computed 序列 `[T,F,T,F,T,F]`。

数据规模：张量 `(2, 4)`–`(4, 4)` float32，N ≤ 8，weight ∈ [0, 2]。
