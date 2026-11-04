# 解题笔记：PAB 注意力广播步

## 回源定位

- 源文件：diffusers `hooks/pyramid_attention_broadcast.py`（314 行，PAB 论文 arXiv 2408.12588）。核心：`_apply_pyramid_attention_broadcast_on_attention_class`（三类匹配 + 优先级分发 L241-287）、`PyramidAttentionBroadcastHook.new_forward`（开区间 + 取模门控 L156-174）、`apply_pyramid_attention_broadcast`（全 None 回退 spatial=2 L220-230）、`_common.py` 的三类 identifier 默认值（spatial 含 "blocks"/"layers" 等宽泛词，temporal 只有 "temporal_transformer_blocks"，cross 与 spatial 前两个重叠）。
- 官方把 hook 挂在每个注意力模块上，`new_forward` 里 `iteration % block_skip_range == 0` 决定重算；`or self.state.iteration == 0` 被 0%n==0 完全覆盖（死代码）——判题域剔除并记录。
- 分发是 hook 创建期一次性决策，门控是逐步决策；判题域合一后 block_type 从 dispatch 流出，便于对拍。

## 数学要点

1. **开区间**：`lo < t < hi` 双侧严格。t==lo 或 t==hi 都算区间外——cache 已建立时强制 compute。变异 m-open-lo/m-open-hi 分别抓两端。
2. **取模序列**：iteration 从 0 起，每次调用 +1（broadcast 也算）；`iteration % N == 0` 给出 compute 位置 0, N, 2N…。变异 %→!=0（反转）与 %N==1（错位）都被序列对拍抓。
3. **cache 更新语义**：`state.cache = output` 无条件执行——broadcast 步 cache 被写成旧 cache 的别名（值不变），compute 步立即换新。m-cache-noedit（cache 不更新）在「区间外 compute → 再回区间内 broadcast」场景暴露（旧值 vs 新值）。
4. **优先级链**：spatial → temporal → cross 的 if/elif 顺序 + `not is_cross`/`is_cross` 互斥守卫。官方默认 spatial_ids 的 "blocks" 用 re.search 会吞掉 temporal 层名——这不是 bug 而是官方行为（temporal 只有 spatial_range=None 才生效），优先级反转变异被 11 个用例抓住。

## 实现要点

- 校验顺序：name/is_cross/张量/current_timestep → 三个 block_skip_range（None 或 ≥1 非 bool int）→ 三个时间步区间（二元整数组）→ 三组 identifiers（字符串序列）→ state 两键（未知键先于缺键、iteration 非 bool ≥0、cache 张量或 None）。
- 全 None 回退 spatial=2 必须在 dispatch 之前。
- 未匹配层：`dict(state)` 原样透传（iteration/cache 不动），output=hidden_states、computed=True、block_type=None。
- identifiers 的 `isinstance(identifiers, str)` 检查是死代码（字符串本就不是 list/tuple，后面的检查报同样的错）——等价变异记录。

## 测试要点

- 分发四象限：spatial 命中 + cross 排除、temporal 被 spatial 抢（官方默认 ids）、spatial_range=None 时 temporal 接管、cross 需 is_cross=True + 命中 + 未命中层 None。
- 门控序列：N=2 交替 [T,F,T,F,T]、N=3 [T,F,F,T,F]、broadcast 步 output 逐值对照（上一次 compute 的输出）。
- 开区间两端点 t==lo/t==hi 恒 compute；区间中点才可 broadcast。
- cache=None 强制 compute（手工 state iteration=5 + cache=None）；iteration=0 + 非 None cache 也 compute（0%n==0）。
- 出区间 compute 后 cache 立即换新（m-cache-noedit 的观测点）。
- temporal/cross 各自独立的 timestep 区间（temporal (0,50)、cross (50,100) 互不串）。
- purity + 未匹配层 state 透传 + 校验 battery（skip_range=0/True/-1、区间长度/dtype、identifiers 传字符串/含非字符串）。

## 变异清单（23 条全部 CAUGHT；1 条等价剔除）

分发 6：spatial 丢 cross 排除、cross 标志取反、优先级对调、any→all、search→match、默认回退删除。门控 6：cache None 检查删、开区间左端 <=、右端 <=、within 条件删、取模反转、取模错位。输出/状态 5：broadcast 输出换 hidden、compute 输出换 cache、cache 不更新、iteration 不增、未匹配层 computed 翻转。校验 6：name、is_cross、skip_range 下界、时间步区间长度、identifiers、state 缺键 + 消息。

**等价变异记录（剔除）**：① identifiers 的字符串前置检查——被 list/tuple 检查完全子sumed（同报错）；② 官方 `iteration == 0` 分支——0%n==0 恒真，不入实现。

经验：① 薄门控题靠分发层（regex + 标志 + 优先级）补足机制密度，判题域把「创建期决策」与「逐步决策」合一后 block_type 成为天然对拍通道；② 官方死代码（0%n==0 分支、字符串前置检查）识别后剔除比照抄更诚实；③ docstring 与代码同串的变异片段用缩进定位（`or iteration % ...` → `\n        or iteration % ...`）。
