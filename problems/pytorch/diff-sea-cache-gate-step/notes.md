# 解题笔记：SeaCache 频域指示器门控步

## 回源定位

- 源文件：diffusers `hooks/sea_cache.py`（1097 行，SeaCache 论文 arXiv 2602.18993，Cosmos3/Wan 视频变换器）。核心：`_apply_sea_filter`（三轴 Wiener 型频域增益 + 密度归一化 L362-389）、`_build_indicator`（`latent.movedim(0,-1)` 后滤波，L392-405）、`SeaCacheSharedState.resolve_gate`（累积相对 L1 预算 + 五条强制 compute + 非相邻重置 + (step,sigma) 备忘，L182-245）、`SeaCacheRootHook.prepare_decoder_stack`（Cosmos 栈路径：残差复用/外推/双 fail-open，L606-724）、`SeaCacheLeaderBlockHook.new_forward`（generic 路径同款，L790-878）、`_record_full_execution`（history 截断到 residual_order+1，L288-311）。
- 官方两套入口（Cosmos `_run_decoder_stack` 替身 / generic leader 块钩子）共享 `resolve_gate`；skip 输出都是 `hidden + cached_residual`。Cosmos 路径残差形状不符只清 history 不清 accumulated，generic 路径两者都清——判题域取 generic 语义（归零）。
- 官方 gate 备忘 `gate_key == (step, sigma)` 直接返回缓存决策：那是一次 forward 内 root+leader 双入口的幂等保护，判题域单调用无此需求。

## 数学要点

1. **Wiener 型增益的形状**：`axis_gain = s·cp/(s²·cp + n² + 1e-16)`，s=1−σ、n=σ、cp=1/(|f|^p+1e-16)。低频 cp→1e16 时 axis_gain→1/s（放大信号），高频 cp→0 时→0（压噪）——典型的去噪增益。三轴连乘后除以均值（密度归一）保证跨步距离可比。
2. **T=H=W=1 的恒等性**：fftfreq(1)=[0]，cp=1e16，axis_gain≈1/s，三轴积 (1/s)³，均值归一后恒 1——指示器≈恒等（float32 下误差 ~1e-7）。这给了 sea_filter 一个免 oracle 的快速健全性测试。
3. **`>=` 语义**：`candidate >= threshold` 即 compute。threshold=0 + 零距离（同 latent 同 sigma 逐位复现）构造出 candidate=0.0 的精确边界——`>` 变异会错误 skip。
4. **distance 参照 filtered previous**：`num/den`，den=mean|prev|+1e-16。注意 float32 下 `den + 1e-16 == den`（无操作），且 num 的 abs 交换律使 current/prev 互换在分子上不可观测——**分母参照（prev vs cur）才是可观测变异**，delta=0.3 的 latent 对给出 ~9% 偏差（delta=0.02 只有 3.7e-9，不可观测——变异用例的 delta 必须够大）。
5. **外推 scale**：`(step−latest_s)/(latest_s−prev_s)`。相邻 history（步差 1、当前步 +1）恒为 1——必须构造非相邻 history（步 0、2，当前步 3 → scale=0.5）才能抓分子分母互换变异。
6. **非相邻重置先于一切门控**：重置后 previous_indicator=None → 首观测强制 compute → distance=None。回跳（last=3 → step 1）也是非相邻——`!=` 不能弱化成 `>`。

## 实现要点

- sea_filter 是独立模块级函数（官方同名语义）；power_exp/sigma 校验放 sea_filter 内（**solve 里不要重复校验 power_exp**——双份校验互为掩护，单副本变异不可抓）。
- solve 校验顺序：三张量 → latent 4 维 → gen 形状 → step/N → sigma 域 → threshold → residual_order ∈ {0,1} → retention/cache_end/max_consecutive → state 五键（未知键先于缺键、history 条目白名单）。
- history 条目 `{"step", "residual"}`；append 后 `[-(residual_order+1):]` 截断；skip 步 history 不变、consecutive+1；compute 步 consecutive=0。
- 纯函数式：`list(state["history"])` 浅拷贝 + 新条目；previous_indicator 存 `.detach().clone()`。

## 测试要点

- 指示器三层验证：T=H=W=1 恒等性（免 oracle）→ numpy float64 独立 oracle 对拍（atol 1e-4，float32/64 差异 ~1e-6）→ sigma=0/1 clamp 端点。
- 精确边界：threshold=0 + 零距离（同 latent 同 sigma 逐位确定）抓 `>=`→`>`。
- **distance 数值用 rel 容差**（`< 1e-3·d`）：float32 实现对 float64 oracle 的绝对误差随 d 增大，abs 容差要么过松（抓不到分母变异）要么过紧（浮点噪声翻转）；delta=0.3 的 latent 对让分母变异偏差达 ~9%。
- 全循环：6 步 oracle 状态机对拍 computed 序列，先自检所有 oracle 距离离 threshold 边界 ≥30% 余量（`d < 0.7·thr or d > 1.3·thr`），保证 float32/64 差异不翻转决策。
- 邻接性：cache_end 用例必须逐相邻步推进到窗内（step 3 直接跳会先触发非相邻重置、测不到窗）；外推 scale 用例必须先补一个 skip 步维持邻接。
- fail-open 双通道 + 指示器形状不符（distance=inf）+ 非 None 残差形状/dtype 不符 + purity（**history 快照逐条比 step + torch.equal residual**——`list[dict] ==` 会因张量歧义布尔报错）。
- state 校验断言子串避开 `state['key'] 必须...` 的引号边界（用键名或去掉前缀的短语匹配）。

## 变异清单（33 条，全部 CAUGHT；另有 6 条识别为等价剔除）

sea_filter 7：signal/noise 互换、power_exp 幂删除、clean_power 的 1e-16 删除、三轴乘→加、密度归一删除、.real→.imag、movedim 删除。门控 12：>=→>、retention <→<=、cache_end >=→>、max_consec >=→>、max_consec 条件删除、非相邻 !=→>、重置保留 indicator、distance 分子 mean→sum、分母参照 cur、形状检查删除、accumulated 不归零。残差 9：无 history fail-open 删、形状不符 fail-open 删、外推条件 order 反转、scale 分子分母互换、外推取负、skip 输出用 gen_output、残差取负、consecutive +2、compute 不清零。校验 5：step 负、sigma 域、order 集合、power_exp 域、state 缺键 + 消息。

**等价变异记录（剔除）**：① sigma clamp 端点删除——float32 下 1e-6 偏差低于一切可行容差；② 分母 +1e-16 删除——float32 中加法无操作；③ 记录侧 history 截断删除——读取侧切片已覆盖；④ `.contiguous()` 删除——语义无操作；⑤ 返回 dtype `.to(dtype)`→`.to(float32)`——判题域输入全 float32；⑥ 分子 current/prev 互换——abs 交换律。

经验：① 冗余双守卫（solve 与 sea_filter 各验一遍 power_exp）让单副本变异永 SURVIVED——删掉 solve 侧副本，让守卫唯一；② 数值类变异的可观测性要用真实数据实测（3.7e-9 与 6e-2 之间隔着一个 delta 的选择）；③ rel 容差是「实现 float32 vs oracle float64」场景的正确容差形式；④ 涉及步邻接的语义（非相邻重置、外推 scale）测试必须让步序列合法推进，否则会被别的分支抢先。
