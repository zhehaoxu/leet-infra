# 扩散：PAB 注意力广播步（三类分发 + 迭代广播门控）

## 题意

Pyramid Attention Broadcast（PAB）利用相邻时间步注意力状态的高度相似性跳过注意力计算：cross-attention 相似度最高（跳得最频繁）、temporal 次之、spatial 再次。实现 `solve(name, is_cross_attention, hidden_states, state, current_timestep, *_block_skip_range, *_timestep_skip_range, *_block_identifiers)`，每步两段：

```text
# 1. 注意力层分发 (spatial > temporal > cross 优先级, re.search 任意命中)
spatial  = match(spatial_ids)  and spatial_range  is not None and not is_cross
temporal = match(temporal_ids) and temporal_range is not None and not is_cross
cross    = match(cross_ids)    and cross_range    is not None and is_cross
全不匹配 -> 不挂 hook (恒 compute, state 原样透传)
三个 block_skip_range 全 None -> 官方回退 spatial_attention_block_skip_range=2

# 2. 广播门控
within         = lo < current_timestep < hi          # 开区间 (严格比较)
should_compute = cache is None or not within or iteration % block_skip_range == 0
compute   -> output = 真实注意力输出
broadcast -> output = cache (上次输出)
state.cache 每次调用更新为本次 output; iteration 每次调用 +1
```

返回 `dict(output, computed, block_type, state)`，state 两键 `{iteration, cache}`（新 dict，纯函数式；未匹配层原样透传）。

## 关键语义

- **优先级即语义**：官方默认 spatial_ids 含 `"blocks"`，而 `re.search("blocks", "temporal_transformer_blocks.0.attn1")` 命中——temporal 层名会先被 spatial 认领；只有 spatial_range=None 时才轮到 temporal。cross 层靠 `is_cross_attention` 标志从 spatial/temporal 中排除。
- **开区间**：`lo < t < hi` 严格比较——t 恰等于 lo 或 hi 时不在区间内，恒 compute（官方默认 (100, 800) 的两个端点都算区间外）。
- **取模广播**：第 0、N、2N… 次调用 compute，其余 broadcast；iteration 对每次调用（含 broadcast）都 +1。
- **cache 每步更新**：compute 步 cache=本次输出；broadcast 步 cache=本次输出（即旧 cache）——出区间恢复 compute 时 cache 立即换新。
- **官方 `iteration == 0` 分支是死代码**（被 `0 % n == 0` 完全子sumed），判题域不收入。

## 判题域约定

- 官方 hook 挂在每个注意力模块、current_timestep 由 callback 提供；判题域单函数显式传层名/is_cross/当前时间步，分发与门控合一。
- 官方不校验参数；判题域补 name 字符串、is_cross bool、张量、state 两键、block_skip_range ≥ 1 或 None、时间步区间二元整数组、identifiers 字符串序列等显式校验。
- 官方 broadcast 存引用；判题域存 clone（纯函数式）。

## 例子

spatial 层、block_skip_range=2、区间内 t=500：调用序列 computed = `[T,F,T,F,T]`；第 2 次调用输出 = 第 1 次的输出（缓存），第 3 次（iteration=2，2%2==0）恢复 compute。t=100（恰为 lo）或 t=800（恰为 hi）时序列恒 `[T,T,T,T,T]`。block_skip_range=3 时 `[T,F,F,T,F]`。

数据规模：张量 `(1, 2)` float32，N ≤ 8，时间步区间任意整数对。
