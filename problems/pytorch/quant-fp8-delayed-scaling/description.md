# FP8 延迟缩放递推账（DelayedScaling）

生产环境里的 FP8 训练几乎都不是「按当前 batch 现算 scale」，而是把最近若干步的 amax 存成
一个滑动窗口，用**上一步**算出的 scale 去做**这一步**的 cast。延迟带来两个后果：scale 一定
比数据慢一拍，而窗口里的旧 amax 会让 scale 迟迟不放大。

## 接口

```python
def solve(amax_seq, history_len=3, fp8_max=448.0, margin=0, algo="max", init_scale=1.0) -> dict
```

`amax_seq` 是逐步观测到的 amax 序列（1 维 `float64` 张量，允许 `0` / `inf` / `nan`）。
返回逐步账本：

| 字段 | 含义 |
| --- | --- |
| `scales` | 第 t 步 cast 实际使用的 scale（= 上一步更新后的 scale） |
| `next_scales` | 第 t 步更新后的 scale |
| `amax_used` | 第 t 步归约出的 amax（写入 history 后、滚动前） |
| `histories` | 第 t 步的 history 快照（`T × history_len`） |
| `kept` | 该步是否因 `amax <= 0` 或非有限而保留旧 scale |
| `clamped` | 该步算出的 scale 是否溢出、被夹到 FP32 上限 |
| `final_scale` / `final_history` | 序列结束时的 scale 与窗口 |

## 输入输出

```
输入：amax_seq=[10.0, 20.0], history_len=2, fp8_max=448.0
输出：scales=[1.0, 44.8], next_scales=[44.8, 22.4], amax_used=[10.0, 20.0],
      histories=[[10.0, 0.0], [20.0, 10.0]]
```

第一步还没有历史，只能用 `init_scale=1.0`；第二步归约出 `20` 得 `448/20 = 22.4`，但这一步
cast 用的仍是第一步算出的 `44.8`。

## 规则

- **五步顺序**（每一步）：① 本步 amax 写进 `history[0]`；② 归约（`max` = 窗口最大值，
  `most_recent` = 槽 0）；③ 算 `sf = (fp8_max / amax) / 2^margin`；④ 更新 scale；
  ⑤ 窗口左移一格（`history_len == 1` 不滚）并把槽 0 清零。
- **不可用的 amax 保留旧 scale**：`amax <= 0`（含恰好 0）或 `amax` 为非有限值时，
  `scale` 原样保留——**不是**回落到 `1.0`，也**不是**放大到无穷。溢出导致 `amax = inf`
  时 scale 不会自动缩小，这是延迟缩放最反直觉的一条。
- **溢出夹取**：可用的 amax 极小（例如 `1e-40`）时 `fp8_max / amax` 在 float32 下溢出为
  `inf`，此时 scale 夹到 `float32` 上限 `3.4028234663852886e38`；夹取发生在「保留旧 scale」
  之后，所以保留分支不会触发夹取。
- **float32 语义**：amax 写进窗口时窄化成 float32（`1234567.89` 变成 `1234567.875`），
  scale 也按 float32 递推；账本以 `float64` 返回这些值。
- **`margin`**：`sf` 除以 `2^margin`，`margin=0` 为默认（TE 的默认值）。
- **`fp8_max`**：E4M3 为 `448`，E5M2 为 `57344`；本题只要有限正数即可。
- **校验**：`history_len` 为正整数（`bool` 拒绝）、`algo ∈ {max, most_recent}`、
  `margin` 为非负整数、`fp8_max` / `init_scale` 为有限正数；空序列合法（返回空账本）。

## 提示

- 不要为了「更及时」把 scale 更新放到 cast 之前——那就不是延迟缩放了，第一步的
  `scales[0]` 必须还是 `init_scale`。
- 窗口滚动的方向很容易写反：左移一格意味着最老的 amax 从末尾掉出，槽 0 留给下一步。
