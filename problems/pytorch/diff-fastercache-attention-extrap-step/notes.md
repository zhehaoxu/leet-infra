# 解题笔记：FasterCache 注意力外推广播步

## 回源定位

- 源文件：diffusers `hooks/faster_cache.py`（654 行，FasterCache 论文 arXiv 2410.19355）。本题取 `FasterCacheBlockHook`（注意力块钩子）：`new_forward` 门控 L407-432、`_compute_approximated_attention_output` 外推 L392-405、缓存维护 L459-478。denoiser 侧 CFG-Cache（uncond 分支跳过 + 频域 delta 逼近，`FasterCacheDenoiserHook` L261-361 + `_split_low_high_freq` L638）机制独立且张量格式分支繁重（BCFHW/BFCHW permute/flatten），可另立一题。
- 官方门控双重否定：`should_compute = iteration > 0 and iteration % N == 0; should_skip = not should_compute`——`>0` 被 0%N==0 完全子sumed（死代码）。判题域直写 `skip = iteration % N != 0`。
- batch 失配复核：skip 初判后 `skip = distilled or state.batch_size != batch`——cache 只缓存 conditional 半批，denoiser 整批前向（uncond+cond）时注意力输出必须真算（否则形状对不上）。

## 数学要点

1. **线性外推**：`t_1 + (t_1 − t_2)·weight` 是对「注意力状态随时间步线性漂移」的一阶逼近；weight=0 退化为 PAB 式直接复用，weight=1 为 2·t_1 − t_2。
2. **区间只许可广播**：`lo < t < hi` 双侧严格。端点测试必须用**条件半批调用**——整批调用即使落在区间内也会被 batch 失配复核挡回 compute，测不出区间端点（第一版 open_interval 全用整批，三个区间变异全 SURVIVED）。
3. **双槽滚动的分化条件**：新 cache = [旧[1], 新值]。只有当旧 cache[0] ≠ 旧[1] 时，`[旧[1], …]` 与 `[旧[0], …]` 才分叉——而首两次外推因两槽同值（[c,c] → 外推 = c）无法分化；必须走到「compute(A≠B) → skip(外推 C) → compute → skip」的第四层，外推参照才是 C 而非 B。第一版 full_dance 只到 iter3，m-cache-slot0 SURVIVED。
4. **半批切片方向**：`[batch:]` 取后半（uncond 前、cond 后的 batchwise 拼接约定）；`[:batch]` 变异用整批 cache + 条件半批调用的专用场景抓。
5. **缓存写入条件**：`not distilled and size(0) == batch_size` 才 chunk——条件半批调用（size 失配）原样入缓存；`==` 翻转成 `!=` 会把整批原样入缓存、把半批错误切块。

## 实现要点

- 校验顺序：张量（≥1 维，size(0) 要用）→ current_timestep → block_skip_range ≥1 → 时间步区间二元整数组 → weight 有限数值 → distilled bool → state 三键（未知键先于缺键；cache 是「两个张量的 list 或 None」）。
- batch_size 语义：首调用记录 `attention_output.size(0)`（真·uncond+cond 整批），之后保持；当前 batch 每次从输入取。
- 纯函数式：新 dict + cache 条目 clone；入参 state 不动。

## 测试要点

- **完整舞步六层**：整批 compute → 条件半批 skip（两槽同值外推=c）→ 整批 compute（cache=[ca,cb]）→ 条件半批 skip（cb+(cb−ca)w）→ 整批 compute（cache=[approx3, cd]，**槽滚动分化点**）→ 条件半批 skip（参照 approx3）。逐值断言。
- 蒸馏序列：batch 恒定、缓存不切块、skip 不需失配。
- weight=0 / weight=1 两端点闭式验证。
- 开区间端点 × 条件半批调用（恒 compute）+ 区间中点条件半批（真 broadcast）。
- 切片方向专用场景：手工 state 整批 cache + 条件半批调用（iteration=1，注意 0 会先 compute）。
- cache 槽链接断言（新[0]==旧[1] 逐值）、purity、校验 battery（weight=inf/bool、cache 长度 1/含非张量）。

## 变异清单（27 条，全部 CAUGHT，无等价）

门控 8：开区间左/右端 <=、within 条件删、取模反转、取模错位、skip 复核丢 distilled/丢失配、batch_size 记录失效。外推 5：取负、参照换 t_2、weight 丢弃、切片方向两处。输出/缓存 7：skip 输出换真算、compute 输出换缓存、chunk 取前半、写入条件翻转、cache 停更、槽链接用旧[0]、iteration 不增。校验 7：张量、维度、skip_range 下界、区间长度、weight 有限、state 缺键 + 消息。

经验：① **「许可型」条件（区间）与「守卫型」条件（batch 失配）叠加时，测试必须构造能让许可真正生效的输入**——否则许可侧变异全被守卫遮蔽（第一版 3 个区间变异 SURVIVED 的根因）；② 双槽/多槽状态机的变异分化点在状态首次「分叉」之后，序列用例要走到那一层（本题为第 4-5 层调用）；③ 手工 state 的 iteration 要与目标分支对齐（iteration=0 会先被 0%N==0 打到 compute）。
