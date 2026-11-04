# 解题笔记：First Block Cache 残差门控步

## 回源定位

- 源文件：diffusers `hooks/first_block_cache.py`（258 行，ParaAttention 的 First Block Cache，docstring 自述基于 TeaCache 思想但实现更简）。核心：`FBCHeadBlockHook.new_forward`（首块残差 + 门控分发 + 两种输出路径）、`_should_compute_remaining_blocks`（absmean 相对 L1 比较）、`FBCBlockHook`（中间/尾块在 cache 决策下透传）。
- 官方共享 `FBCSharedBlockState`：head_block_output（首块输出缓存）、head_block_residual（首块残差缓存）、tail_block_residuals（尾块相对首块输出的残差）、should_compute（步内瞬态）。`reset()` 只清 tail 与 should_compute，**保留 head 两项**——判题域 state 三元组独立可 None。
- 官方 cache 路径的输出在 **head hook 内**就合成完毕（tail_residual + 当前首块输出），中间块透传、尾块透传——所以判题域把一次前向压缩为 head/tail 两个张量是数学等价的。

## 数学要点

1. **相对 L1 的结构**：diff = mean|residual − prev| / mean|prev|。分子分母同聚合（mean/abs），任何「同聚合替换」都是恒等比值（mean→sum 严格等价：sum = N·mean，2 的幂缩放在浮点除法下逐位不变——本题唯一的等价变异，直接剔除记录）。
2. **abs 的必要性**：残差变小（去噪后期常见）时残差差整体为负，无 abs 的 diff 为负 → 永远 ≤ threshold → 永远 cache（错误）。`check_decreasing_residual` 用残差 (0.15, 0.05) 对 prev (0.2, 0.1) 的负差值场景专测。
3. **stale 比较**：cache 步不写 state，diff 的参照冻结在上次 compute——连续 cache 步的漂移累积到超阈值才恢复 compute。用「步 A 小漂移 cache、步 B 大漂移 compute（与步 A 无关）」的双步场景断言。
4. **尾残差的参照**：`tail_block_residuals = tail_output − head_block_output`（本次 compute 的首块**输出**），不是 head_input——变异 m-tail-res 抓的就是这个。
5. **零分母**：prev 残差全零时官方 nan/inf 行为规范化为 0.0/inf（nan>x False ≡ 0.0 ≤ threshold cache ✓；x/0 inf ≡ inf compute ✓）。

## 实现要点

- 校验顺序：三个张量类型 → 形状一致 → threshold（非 bool、数值、≥0）→ state 结构（三键白名单、值张量或 None）。
- cache 分支 tail 缺失显式拒绝（官方 TypeError，判题域 ValueError 统一）——官方 reset 后（head 保留、tail 清空）首步因 residual 非 None 不走强制 compute，若漂移小则撞 cache → 官方崩、判题域 raise。这是官方 reset 语义的真实角落。
- 纯函数式 state：新 dict + clone，purity 用例断言入参快照不变。

## 测试要点

- 首步：computed=True、diff=None、state 三项填充（tail_residuals = tail − head 逐位）。
- compute/cache 双路径 + diff 数值对照（独立 _absmean_np oracle）。
- **边界用 2 的幂精确构造**：prev (0.5, 0.5)、差 (0.25, 0.25)、thr=0.5 → diff 精确等于阈值 → cache；+2⁻⁴⁰ → compute。第一版用 0.21−0.2 这类非精确数构造「恰等」被浮点误差破坏（0.21−0.2 = 0.00999…98），`>=` 变异抓不到——**边界类用例必须用二进制精确数**。
- 负差值用例（abs 必要性）、零分母双分支、stale 双步、cache 步 state 逐项不变、compute 步 state 三项更新、purity。
- 残差构造全部用显式两列 `_r(a, b)`——`_t` 的线性第二列模式（v·0.5−0.3）会把残差弄成意外值（第一版 5 个用例全因此挂）；同理浮点上 `1.0+0.6−1.0 ≠ 0.6`，期望残差要用 `(ho2 − hi)` 表达而非字面量。

## 变异清单（26 条，全部 CAUGHT，无等价）

决策核心 6：residual 参照错（tail_output）、分子不带 prev、分母用当前残差、>→>=、>→<、首步 diff 非 None。公式 7：cache 加 head_input、尾残差参照 head_input/自身、head_block_output 用 tail、残差缓存不更新（stale 失效）、compute 最终输出用 head_output、首步强制翻转。零分母 2：全 cache 化、全 compute 化。聚合 2：mean→max、abs 删除。防御/校验/消息 11：首步 cache 化、tail 缺失不拒、形状删、threshold 负/bool、state 未知键、值类型、三条错误消息。

**等价变异记录**：mean→sum 在 diff = num/den 的比值结构下是严格等价（分子分母同聚合、sum = N·mean、N 是 2 的幂时浮点乘除逐位精确）——剔除并记录，不强行抓。

经验：① 边界类断言（>= vs >）必须用二进制精确数构造「恰等」；② 带符号聚合（abs 删除）需要负差值场景才可观测；③ 比值结构的聚合替换可能是恒等等价，识别后剔除比硬造用例诚实；④ 残差/输出构造用显式逐列 helper，杜绝线性模式与浮点不可逆表达（`a+b−b ≠ a`）。
