# 解题笔记：MagCache 累积误差门控步

## 回源定位

- 源文件：diffusers `hooks/mag_cache.py`（469 行，MagCache 论文/仓库 github.com/Zehong-Ma/MagCache）。核心：`MagCacheConfig`（threshold=0.06、max_skip_steps=3、retention_ratio=0.2 默认值；ratios 缺失或 calibrate=False 时 raise；长度不等时 `nearest_interp` 插值 L110-137）、`nearest_interp`（最近邻索引插值 L69 起）、`MagCacheState`（previous_residual + 三累积器 + step_index，reset 全清 L140-168）、`MagCacheHeadHook.new_forward`（scale 越界回退 → retention_step → 累积 → 判定 → 双输出路径 L200-280）、`MagCacheBlockHook`（is_tail 时 previous_residual = 尾块输出 − 首块输入 L330-360）。
- 官方 skip 输出公式：`output = hidden_states + previous_residual`（L232/240，首块输入 + 上次尾残差）；compute 走 original_forward 后由尾块 hook 写残差。判题域把「尾块输出」压缩为 tail_output 入参——残差公式 `tail_output − hidden_states` 与官方逐位一致。
- 官方判定发生在 head hook（块执行前），累积器先加本步贡献再判 skip；skip 时尾块 hook 直接透传输入（`return hidden_states`）并推进 step。所以 skip 步的累积器**保留**、compute 步的重置在判定分支内完成——判题域单函数逐步调用与官方「head 判定 + tail 写残差」两段式等价。

## 数学要点

1. **累积误差的结构**：err += |1 − ratio| 其中 ratio = Πscale（连乘）。与「err += |1 − scale|」的单步误差不同：0.98² 的累积贡献是 |1−0.9604|=0.0396 而非 0.02。`check_accumulated_error` 手算三步（0.02 → 0.0596 → 0.1184 爆阈值）对拍。
2. **retention 四舍五入**：`int(r·N + 0.5)` 是正数上的 round-half-up。截断变异 `int(r·N)` 只在 frac(r·N) ≥ 0.5 时可观测——0.2·8=1.6（round 2 / trunc 1）。**反截断用例的关键构造**：retention 判定内的步必须带非 None 残差 + 断言累积器保持初值——若只断言 `computed is True and steps == 0`，截断变异下 step 落入累积区后因 res=None 触发 compute+重置，steps 同样是 0，断言被重置路径骗过（第一版正是这么 SURVIVED 的）。
3. **先累积后判定**：即使判定结果必然 compute（res None / err 超限 / steps 超限），ratio/steps/err 也已更新；只有 compute 分支才重置为初值。变异 m-skip-res（删 res None 条件）抓的就是「res None 时也算 skip」。
4. **err ≤ threshold 恰等**：官方是 `<=`（恰等仍 skip）。边界用 2 的幂构造：scale=0.5、ratio=1.0 → err=0.5 恰等于 threshold=0.5 → skip。
5. **scale 越界回退**：插值后 len==N，但 step ≥ N（判题域允许显式传大 step）时官方回退 scale=1.0——ratio 不变、err 仍按 |1−ratio| 累积。注意回退不等于「什么都不做」。

## 实现要点

- 校验顺序：张量类型/形状 → step/N/threshold/retention/max_skip 数值域 → mag_ratios（None 拒绝、list 转张量、一维非空、浮点、长度插值）→ state 结构。state 校验**未知键先于缺键**（`{"foo":1}` 应报未知键而非缺键）、缺键先于值类型。
- nearest_interp 独立函数：src 校验（一维非空浮点张量）+ target_length 校验（非 bool、≥1）→ target==1 取末元素 → scale/grid/round 索引。
- 纯函数式 state：新 dict + clone；compute 分支残差 `(tail_output − hidden_states).clone()`，skip 分支保留旧残差 clone。

## 测试要点

- 全循环对拍：独立 `_oracle_loop`（同状态机）生成 computed 序列。
- retention_rounding：两个舍入组合（0.2·8、0.25·10）+ **反截断组合（0.2·8 的 step 1 带残差断言零累积）** + skip 步累积器保留（ratio=1.5/err=0.5）。
- accumulated_error：三步手算 + 重置断言 + skip 输出 = hidden + 旧残差（用 `(tail − hidden)` 表达式而非字面量，浮点上 `1.0+0.6−1.0 ≠ 0.6`）。
- max_skip_steps：scale=1（err 恒 0）下 steps 1,2,3 skip、第 4 次 compute+重置、再 1,2 skip 的完整序列。
- stale_residual：skip 步换新 tail_output 也不影响输出；compute 步残差更新。
- nearest_interp 经 solve 间接验证（判题器独立命名空间下 `import solution` 不可用）：err = |1 − 插值后 ratios[step]| 反推索引，覆盖 round 银行家舍入（0.5→0）、target==1、list 入参。
- 校验用例注意：`solve` 有默认参数，`_expect_err` 传 `**kw` 时不能与显式关键字重复（第一版 `**kw` 与内联 threshold 冲突）。

## 变异清单（31 条，全部 CAUGHT，无等价）

retention/门控 3：+0.5 截断、>=→>、retention 删除。累积器 10：ratio 赋值替换连乘、err 用单步 scale、err 去 abs、err <=→<、steps <=→<、res None 条件删除、err 条件恒真、重置删除、重置 ratio=0、scale 越界回退 0.0。输出路径 6：skip 输出用 tail、compute 输出用 hidden、残差参照自身/取负、skip 步残差不保留。nearest_interp 4：scale 公式、round→floor、target==1 取首、插值删除。校验/消息 8：step 负数、N≥1、threshold 负、retention 范围、max_skip≥1、ratios None 消息、整型 ratios、state 缺键 + 错误消息。

经验：① 「retention 内不累积」类断言必须防重置路径伪装——带非 None 残差 + 断言累积器初值；② compute 步的 output 值也要断言（只测 skip 路径输出会让 compute 输出替换 SURVIVED）；③ docstring 与代码同串时变异片段用缩进/上下文定位；④ 判题器独立命名空间下 test_cases 不能 `import solution`，间接验证（经 solve 的可观测副作用）是正道。
