# Paraformer CIF 预测器 fire 发射账

Paraformer（FunASR）不做自回归解码，而是用 **CIF（Continuous
Integrate-and-Fire，Paper 1905.11235）** 从 encoder 隐态里按权重"积攒"
出声学嵌入：每帧带一个 `[0,1)` 的发射权重 alpha，累积越过阈值 1.0 即
"fire"——发射一枚 token 嵌入并把跨阈余量留给下一枚。这是全库唯一的
integrate-and-fire 机制题。官方实现：FunASR
`funasr/models/paraformer/cif_predictor.py` 的 `cif()` 与
`tail_process_fn`。

## 接口

```python
solve(alphas: list[float], hidden: list[list[float]], threshold: float = 1.0,
      tail_threshold: float = 0.45, use_tail: bool = True) -> dict
```

- `alphas`：每帧发射权重（官方为 sigmoid 输出经 smooth/noise 阈值，
  合法域 `[0, 1)`）；`hidden`：`(T, D)` encoder 隐态，与 `alphas` 同帧数
  （官方不变量 `len(hidden) == len(alphas)`）；
- 返回 `{"embeds": 发射嵌入序列, "fire_positions": [fire 判定帧...],
  "token_num": int}`。

## 状态机（逐条照抄官方 `cif()`，单序列视角）

每帧 `t`，`alpha = alphas[t]`：

1. `distribution_completion = 1 - integrate`——**在加 alpha 之前**取，
   是当前 token 尚缺的权重；
2. `integrate += alpha`；
3. **fire 前值快照**（官方 `list_fires.append` 在减 1 之前）；
4. `fire = integrate >= threshold`——**闭区间**，恰好等于阈值也发射；
5. fire 时 `integrate -= 1`，`cur = completion`（补齐当前 token 的缺口
   部分）；否则 `cur = alpha`；`remainds = alpha - cur`（跨阈余量）；
6. `frame += cur * hidden_t`；fire 时**发射 frame**（当前 token 的累计
   加权和），并将 `frame` 重置为 `remainds * hidden_t`。

发射嵌入序列 = 各 fire 帧的 frame 快照；`fire_positions` 升序。

## token_num 的两处官方口径

- `use_tail=True`（官方推理路径）：`tail_process_fn` 在 `alphas` 末尾
  **追加一帧 `alpha = tail_threshold`**，并给 `hidden` 追加一行
  **零向量**——tail 帧恰对该零行（官方不变量 `len(hidden) ==
  len(alphas)`），补缺口的权重乘零向量，末 token 的嵌入只由真实帧
  累计而成。`token_num = floor(扩展后总权重和)`。单序列下发射数恒等于
  该 floor 值（可自证：每次 fire 后余量 < 1，追加 tail 后要么恰好凑满
  一枚要么余量 < 1）。
- `use_tail=False`（官方 `cif()` 内 `len_labels` 口径）：
  `token_num = round(总权重和)`。此时发射数由快照决定，**可以与
  token_num 不同**（尾部余量丢弃）。

## 样例

```python
h = [[1,0],[0,1],[2,2],[3,3]]
solve([0.5,0.5,0.5,0.5], h, use_tail=False)
# fire at 1、3；embeds=[[0.5,0.5],[2.5,2.5]]；token_num=round(2.0)=2

solve([0.6,0.6,0.6], h[:3], use_tail=False)
# fire 仅 at 1；embeds=[[0.6,0.4]]（余 0.2 丢弃）；token_num=round(1.8)=2
# ——发射数 1 ≠ token_num 2

solve([0.6,0.6,0.6], h[:3], use_tail=True)
# tail 帧对零行；fire at 1、3(0.8+0.45=1.25)
# embeds=[[0.6,0.4],[1.2,1.4]]（第 2 枚 = 0.2h1+0.6h2，零行不贡献）
# token_num=floor(2.25)=2

solve([0.55], [[1.0,2.0]], use_tail=True)
# 尾余 0.55 + tail 0.45 = 1.0 恰好 fire → fire at 1
# embeds=[[0.55,1.1]]；token_num=floor(1.0)=1
```

## 提示

- `cur` 在 fire 时是**缺口**而非整段 alpha；`remainds = c_t - k`（cumsum
  跨过第 k 个整数后的超出量）会作为下一枚 token 的起始权重。
- 快照时机错了（用减 1 后的值）会让恰达阈值的发射丢失（快照 0.0 <
  1）。
- 无 tail 时的 `round` 与发射数无关——它只影响官方 batch pad 宽度，
  本题仅要求输出该口径值。
