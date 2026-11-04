# 解题笔记：TaylorSeer 缓存步

## 回源定位

- 源文件：diffusers `hooks/taylorseer_cache.py`（348 行，TaylorSeer 论文 arXiv 2503.06923，2025-03）。核心三块：`TaylorSeerCacheHook._measure_should_compute`（门控）、`TaylorSeerState.update`（差分递推）、`TaylorSeerState.predict`（泰勒外推）。
- 官方 `update` 里有一个防御分支值得注意：形状变化时 break 重启展开（注释解释是 Qwen-Image 的 KV prefill 步会返回 prefix+target token，形状不同无法做差分）——这是判题域可观测行为，必须考。
- 官方 `_measure_should_compute` 每次调用都 `state.current_step += 1`（hook 内部递增）；判题域把 step 显式化，state 只保留 taylor_factors 与 last_update_step。
- 官方 factor dtype 默认 bfloat16（`taylor_factors_dtype`），判题域固定 float64 省 dtype 参数。

## 数学要点

1. **差分表即导数近似**。`f[j+1] = (f[j]_new − f[j]_prev) / delta` 是数值微分的多步差分：f[1] 是一阶差商（近似一阶导数），f[2] 是二阶差商（近似二阶导数），delta 是 compute 步间距而非 1——这是与「逐帧差分」最大的语义差别（m-delta-step 变异专测）。
2. **泰勒外推**。`out(s) = Σ_j offset^j/j!·f[j]`，offset = 当前 predict 步 − 上次 compute 步。max_order=1 退化为线性外推 f0 + offset·f1；max_order=2 加曲率项 offset²/2·f2。间隔 5 步、offset 可达 4，曲率项贡献 8·f2 量级——不是小修正。
3. **门控的算术**。`(step − before − 1) % interval == 0` 的语义是「warmup 结束后的第 interval 步开始对齐」：before=3、interval=5 时 compute 步 0,1,2,4,9,14——**step 3 是 predict**（(3−4)%5=4）。off-by-one 的两个方向（减不减 before、减不减 1）各对应一个可观测的 compute 步移位，全循环断言 computed_steps 序列一抓一个准。
4. **cooldown 覆盖 interval**。after 条件是 or，所以 cooldown 段每步都 compute（即使不对齐 interval），且 update 继续（递推 delta 回到 1）。

## 实现要点

- 纯函数式 state：返回新 dict、张量 clone、不改入参（官方是 StateManager 原地维护）——判题域约定写入 docstring，purity 用例直接断言入参快照不变。
- 校验顺序：outputs → step（int 非 bool、≥0）→ interval（≥1）→ before（≥0）→ after（非负或 None）→ max_order（≥1）→ state 结构（keys 白名单、factors 键为非负 int 阶、值为张量、last_update_step 非负 int 或 None）。`-2` 的 after 漏负数检查是第一版 bug（「非负整数」消息骗了自己）。
- predict 的 raise 条件是 `last_update_step is None or not factors`（or）——官方语义两个不一致方向都 raise；只判 factors 会漏 lus=None + factors 非空（m-predict-lus 首轮 SURVIVED 即此，补不一致 state 用例后 CAUGHT）。
- compute 步的 delta==0 raise 在首更新检查之后（首更新 lus=None 不受影响）。

## 测试要点

- **全循环对拍**（before=3, interval=5, max_order=2, step 0..11）：oracle 独立实现门控+递推+外推，逐断言 computed 标志、compute 步的因子集合与数值、predict 步的输出——门控 off-by-one 与递推错误同时被覆盖。
- interval 对齐用例直接断言 computed_steps 序列 `[0, 2, 5, 8]`（before=1, interval=3）——注意 before=1 时 compute 步是 2,7,12 不是 5,10（第一版想当然写成 [0,5,8]，跑基线时被抓）。
- cooldown 用例断言 `[0,1,2,4,6,7,8]`（after=6 后每步 compute）。
- 手工 state 直塞 predict（跳过递推）：一阶/二阶外推的数值公式直接对照（`f0 + offset·f1 + offset²/2·f2`）。
- 形状变化重启：compute 步给不同形状输出，断言 factors 只剩 order 0。
- state 一致性双向：factors 空但 lus 非 None、lus None 但 factors 非空，predict 都 raise。
- purity：同入参两次调用一致 + 入参快照不变。

## 变异清单（31 条，全部 CAUGHT，无等价）

门控 8 条：warmup 边界松 1、warmup 删、interval off-by-one 两向、interval 删、cooldown 边界紧、cooldown 删、or→and。递推 6 条：delta0 校验删、delta 用 step、max_order−1、形状重启条件删、差分符号、差分不除 delta。外推 6 条：offset 用 step、offset 恒 1、阶乘 +1、幂 +1、求和符号、lus 检查删。校验/消息 11 条：step 负/bool、interval 0、max_order 0、before 负、after 负、因子值非张量、lus bool、state 未知键、delta 消息、predict 消息。

经验：门控 off-by-one 用「computed_steps 序列断言」比逐点断言覆盖密度高；不一致 state 的双向用例（factors 空/lus None 各自与组合）要显式构造——官方 raise 条件的 or 结构决定了删除任一分支都必须有专属用例可抓。
