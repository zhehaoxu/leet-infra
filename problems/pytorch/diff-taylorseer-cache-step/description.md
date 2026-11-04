# 扩散：TaylorSeer 缓存步（门控 + 差分递推 + 泰勒外推）

## 题意

TaylorSeer（arXiv 2503.06923）让扩散 Transformer 的模块在部分去噪步**跳过前向**、用泰勒外推近似输出。实现 `solve(outputs, state, step, cache_interval=5, disable_cache_before_step=3, disable_cache_after_step=None, max_order=1)`：

**门控**（决定本步是 compute 还是 predict）：

```text
should_compute = (step < disable_cache_before_step)                     # warmup
              or ((step − disable_cache_before_step − 1) % cache_interval == 0)
              or (disable_cache_after_step is not None and step >= disable_cache_after_step)
```

**compute 步**执行差分递推（delta = step − last_update_step）：

```text
f[0] = output
f[j+1] = (f[j]_new − f[j]_prev) / delta        # j = 0..max_order−1
```

首次更新只存 f[0]；若 f[j]_prev 缺失或**形状变化**则 break（重启展开，只留 order 0）；delta == 0 时报错。

**predict 步**做泰勒外推（offset = step − last_update_step）：

```text
output = Σ_j (offset^j / j!) · f[j]            # j 遍历已有阶数
```

返回 `dict(output, computed, state)`，state 为新 dict（纯函数式，不修改入参）。

## 关键语义

- **warmup 步也要 update**：`step < disable_cache_before_step` 的步强制 compute 且更新因子（官方注释 "factors are updated, caching not applied"）——没有这些因子，warmup 后的第一步 predict 会因 factors 为空报错。
- **interval 的 off-by-one**：compute 步满足 `(step − before − 1) % interval == 0`。before=3、interval=5 时 compute 步是 0,1,2（warmup）+ 4, 9, 14…，而 step 3 是 predict（(3−4)%5=4≠0）。
- **delta 是 compute 步间距**（如 4→9 为 5），不是恒为 1——差分要除以它才是「每步导数」。
- **offset 是 predict 步距上次 compute 的偏移**（1..interval−1 变化），不是恒为 1。
- **形状变化重启**：输出形状在两次 compute 之间变化时（如 Qwen-Image 的 KV prefill），旧因子无法做差分，展开从 order 0 重新开始。
- **不一致 state**：`last_update_step is None` **或** factors 为空，predict 都报「Cannot predict without prior initialization/update.」（官方 raise 条件是 or）。

## 判题域约定

- 官方 hook 内部递增 `state.current_step`；判题域 step 显式入参、state 不含 current_step。
- 官方支持 tuple 多输出；判题域单张量输出。官方因子 dtype 默认 bfloat16；判题域固定 float64。
- `cache_interval ≥ 1`（官方 0 会 ZeroDivisionError）与 `max_order ≥ 1` 为判题域显式校验。
- before=0 时 step 0 落入 predict 且无初始化——官方可达路径，保留报错分支。

## 例子

before=3、interval=5、max_order=2、after=None 的前 12 步：compute 步 = 0,1,2,4,9；predict 步 = 3,5,6,7,8,10,11。step 3 的 offset=3−2=1；step 8 的 offset=8−4=4。cooldown after=6 时 step 6/7/8 也强制 compute。

数据规模：输出张量 `(1, 2)`–`(2, 2)` float64，因子阶数 ≤ 3。
