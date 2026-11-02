# notes（quant-fp8-delayed-scaling）

## 回源

- 官方仓：`NVIDIA/TransformerEngine` main 分支。旧的 `transformer_engine/pytorch/fp8.py`
  已标记 DEPRECATED，现逻辑在 `transformer_engine/pytorch/quantization.py`：
  - `_default_sf_compute(amax, scale, fp8_max, margin)`：
    `sf = (fp8_max / amax) / (2**margin)`，然后三条 `torch.where`——
    `amax > 0` 否 → 用旧 `scale`；`isfinite(amax)` 否 → 用旧 `scale`；`isinf(sf)` 是 →
    `_fp32_max`（`torch.finfo(torch.float32).max`）。
  - `_default_get_amax_and_update_history(amax_history, algo)`：`max` → 窗口最大值，
    其余（`most_recent`）→ `amax_history[0]`，随后 `_update_amax_history`。
  - `_update_amax_history`：`shape[0] > 1` 才 `torch.roll(h, -1, 0)`，然后 `h[0].fill_(0.0)`。
- 默认值（`DelayedScalingRequest`）：`margin=0`、`amax_history_len=1024`、`algo="max"`。
  本题把窗口缩到 1–8 便于手算，公式不改。
- `fp8_max`：E4M3 = 448、E5M2 = 57344（`Format` 决定），题面作显式入参。

## 官方实现里的坑

1. **`inf`/`nan` 的 amax 不缩小 scale**：`torch.where(isfinite(amax), sf, scale)` 直接回退到
   旧 scale。也就是说 cast 溢出产生了 `inf` amax 之后，scale 原地不动，下一步会继续溢出——
   这是官方行为，不是 bug，题目照抄。
2. **`amax == 0` 保留旧 scale，不回落到 1.0**：docstring 里写「previous scale (or 1)」，
   代码里只有 `scale`（初值 1.0）；实现要按代码走。
3. **夹取在最外层**：三条 `where` 的顺序是「先用旧 scale 替换不可用分支」再「isinf → FP32
   max」。所以 `amax == 0` 时算出的 `inf` 不会触发夹取；只有「可用但极小」的 amax 才会被夹。
4. **history 左移后槽 0 清零**：滚动方向写反会让窗口永远留下同一个大 amax，scale 长期偏小。
5. **窗口长度 1 不滚动但照样清零**：`shape[0] > 1` 才 roll；清零无条件执行，等价于
   「每步只留本步 amax」。
6. **float32 窄化**：`amax_history` / `scale` 都是 float32 张量，`1234567.89` 进窗口就变成
   `1234567.875`；用 float64 复刻会在最后几位上对不上。

## 本题的规范化

- 账本显式给出 `scales`（cast 用的）与 `next_scales`（更新后的）两列，再加上 `histories`
  快照，使「延迟一拍」和「窗口滚动」都可观测，不靠浮点差猜。
- `kept` / `clamped` 两个布尔列把两条容易混淆的分支分开记录：`kept` 是「amax 不可用」，
  `clamped` 是「可用但 sf 溢出」。
- 校验门：`history_len` 正整数（`bool` 拒收）、`algo` 白名单、`margin` 非负整数、
  `fp8_max` / `init_scale` 有限正数；空序列返回空账本且 `final_scale == init_scale`。
- 所有标量运算在 float32 下做，返回时 `double()`，保证逐位可复现。

## 变异

17 条全部 CAUGHT：

```
m-margin-mul                margin 用成 margin+1        → margin 用例抓
m-kept-invert / m-kept-drop 保留分支反向 / 删除          → zero/inf 用例抓
m-clamp-drop / m-clamp-value 夹取删除 / 阈值改 1e38      → 1e-40 用例抓
m-roll-dir                  滚动方向反向                → 窗口淘汰用例抓
m-no-zero-slot              槽 0 不清零                 → histories 断言抓
m-no-roll                   永不滚动                    → 窗口淘汰用例抓
m-algo-swap                 max / most_recent 互换       → 对比用例抓
m-scale-recorded-after      scales 记成更新后的值        → 延迟一拍用例抓
m-hist-fp64                 窗口用 float64              → 窄化用例抓
m-usable-finite-only        只判有限不判 >0              → amax=0 用例抓
m-usable-positive-only      只判 >0 不判有限             → inf/nan 用例抓
m-validate-history/margin   两条校验门改坏              → validation 抓
m-final-history-stale       final_history 用旧快照       → 尾部断言抓
m-validate-fp8max           fp8_max 门放宽到 >=0        → validation 抓
```

## 查重

- FP8 往返(125)、E5M2 分块缩放(445)：元素格式与「按块共享指数」的静态口径，不涉及
  跨步递推。
- `quant-fp4-e2m1-grid`(740) / `quant-fp6-dual-subformat-pack`(741)：元素位域与打包。
- 本题只考「scale 递推状态机 + 窗口 + 两条回退分支」，与上述均不重叠。
