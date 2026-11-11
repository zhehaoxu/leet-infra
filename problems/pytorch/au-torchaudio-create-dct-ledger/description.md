# torchaudio `create_dct` 构造账：DCT-II 余弦矩阵、`n_mels × n_mfcc` 转置、`norm=None` 全局 ×2 与 `norm="ortho"` 首列 ×1/√2 再 ×√(2/n_mels)、非法 `norm` 精确报错

## 1. 背景

`torchaudio.functional.create_dct` 构造 MFCC 用的 DCT-II 变换矩阵。它只做 `cos`、转置与两个 `norm` 口径的缩放，**不含 RNG、不含音频 IO、不含 torch 外的任何库**，也没有 warning。本题把它用纯 Python 复刻成三个可独立判题的 mode（`matrix` / `norm_error` / `census`），把全部数值用官方常量快照钉死。
全库查重：函数名 `create_dct` 在 `problems` 的全部 `meta.json` / `description*.md` / `notes*.md` 中零命中。`audio-mfcc` 只是把正交 DCT-II 当作 MFCC 流水线的一步（含 lifter / CMVN、单 `norm` 变体、作用于 logmel），并不构造 `create_dct` 独立矩阵；855 的 notes 提到 `create_dct` 仅指抽取手法，未覆盖该函数。867 直接判 `create_dct` 的矩阵中间量与 `norm` 口径，与既有题边界清晰。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  create_dct(n_mfcc, n_mels, norm)   (lines 636-667)

  if norm is not None and norm != "ortho":
      raise ValueError('norm must be either "ortho" or None')
  n = torch.arange(float(n_mels))
  k = torch.arange(float(n_mfcc)).unsqueeze(1)
  dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # (n_mfcc, n_mels)
  if norm is None:
      dct *= 2.0
  else:
      dct[0] *= 1.0 / math.sqrt(2.0)
      dct *= math.sqrt(2.0 / float(n_mels))
  return dct.t()   # (n_mels, n_mfcc)
```

`official867.py` 用 `ast.get_source_segment` 抽出顶层 `create_dct` 并 `exec` 进同一共享 NS，官方函数体**一字未改**；注入对象只有真实的 `torch` / `math` / `typing` 与类型别名，全局默认 dtype 固定为 `float64`，因此 `torch.cos` 与纯 Python 参考解可对拍到浮点精度。矩阵由 `official867.py` 用真实 torch 复算，是判题常量的唯一权威来源。

## 2. 口径约定

### 三个 mode

`solve(mode, **kw)` 分发三路：`matrix` 返回 `n_mfcc` / `n_mels` / `norm` 决定的矩阵，`norm_error` 判非法 `norm` 是否抛错（或合法 `norm` 返回 `"NO_ERROR"`），`census` 判源码级事实。除 `unknown mode` 抛 `ValueError` 外，所有异常（含非法 `norm`、`n_mels=0`）都收敛成 `{type, msg}` 字典返回，消息逐字对齐官方。

### 矩阵与缩放

构造矩阵是 DCT-II：`dct[i][j] = cos(pi/n_mels * (n+0.5) * k)`，形状 `(n_mfcc, n_mels)`；其中 `n` 是 mel 轴索引、`k` 是 mfcc 轴索引。函数最后 `return dct.t()`，所以**返回矩阵的形状是 `(n_mels, n_mfcc)`**，而不是构造时的 `(n_mfcc, n_mels)`。`norm=None` 把整张矩阵乘以 `2.0`；`norm="ortho"` 先把**构造矩阵的第 0 行**（`dct[0]`）乘以 `1/sqrt(2)`，再整张乘以 `sqrt(2/n_mels)`，而构造矩阵的第 0 行在转置后变成返回矩阵的**第 0 列**。

## 3. 四个官方 quirk

① `norm` 只接受 `None` / `"ortho"`，大小写敏感：其他任何值（含 `"ORTHO"` / `""` / `"none"`）都抛 `ValueError`，消息逐字为 `norm must be either "ortho" or None`。② `norm=None` 是**全局乘以 2.0**，包含所有 `i, j`，没有例外。
③ `norm="ortho"` 的首列 `1/sqrt(2)` 缩放作用在**转置后的第 0 列**，而不是第 0 行；整张 `sqrt(2/n_mels)` 缩放才是全局。④ `n_mels=0` 在求 `math.pi / float(n_mels)` 时**立即**抛 `ZeroDivisionError`（不是逐元素惰性求值），属于退化输入，明确排除在题目输入域外；`n_mfcc=0` 则返回形状 `(n_mels, 0)` 的空矩阵（0 列），不抛异常。

## 4. 判题账本

`fixtures867.py` 从逐字官方环境导出 31 个常量，`cases867_checks.py` 提供 37 个 `check_*(solve)`：既有数值容差比较，也有针对矩阵的逐元素精确相等断言（参考解用与官方相同的 DCT-II 公式与转置，全局默认 dtype 锁定 `float64`，与 `torch.cos(float64)` 实测差异在 1e-12 量级）。另有形状必为 `(n_mels, n_mfcc)`、四个角点 `cos(0)=1 → 2.0`、转置公式、正交性（`norm="ortho"` 且方阵时 `D @ D.T == I`）、错误字典与 `n_mels=0` / `n_mfcc=0` 边界等硬化检查。

## 5. 验证流水线

```text
official867.py smoke          3 modes, EXIT=0
probe867.py                   1390/1390 ALL GREEN (12 segments)
probe_solution867.py          35593 checks / 0 bad (2 layers)
verify867.py                  87272 checks / 192 configs / 0 fails
fixtures867.py                31 constants
driver867.py                  37/37 ALL ACCEPTED
mut867.py                     42 mutants / 0 equivalents / 42 CAUGHT
```

三层对拍的第三层与参考解**不共享任何算法**：用泰勒级数（`-x^2/(2k+2)(2k+1)` 递推）独立计算 `cos`，不依赖 `math.cos`；再用代数性质校验——方阵 `norm="ortho"` 时 `D @ D.T == I`，这一性质与具体的余弦公式无关，能抓出任何转置方向、缩放或公式错误。42 项变异全部被 `check_*` 捕获，无任何被判为真等价的变异（不以删除变异的方式让测试变绿）。

## 6. 工程教训与边界

第一个坑是**转置方向**：构造矩阵是 `(n_mfcc, n_mels)`，返回的是其转置 `(n_mels, n_mfcc)`，所以 `returned[i][j] = cos(pi/n_mels * (i+0.5) * j)`，两个索引的归属很容易写反，必须用 12 段探针的转置公式检查与正交性性质双重钉死。第二个坑是**正交首列缩放落在第 0 列而非第 0 行**——`dct[0]` 是构造矩阵的第 0 行，转置后才变成返回矩阵的第 0 列，直接断言第 0 行会漏掉这个语义。
第三个坑是 `n_mels=0` 的 `ZeroDivisionError` 来自 `math.pi / float(n_mels)` 的**顶层立即求值**，而不是逐元素惰性求值；参考解必须把这一步提到循环外，否则 `n_mels=0` 会错误地返回空矩阵而非抛异常。第四个坑是 `n_mfcc=0` 返回的是形状 `(n_mels, 0)`（0 列）而非 `(0, n_mels)`，判题边界检查需按 0 列处理。

## 7. 注册与护栏

`register867.py` 最小侵入注册 `audio` 18→19、中英 description 各追加 867 口径账长段，复核 `845 registered = 845 disk / max_order=867 / 无 dangling / 无 unregistered`。`todos.md` 更新为 845 题 / audio 19 / 下一批从 868 起，并把 867 从「已立项」改为「已落地」。落地后跑 selftest `submit` / `run` 双模式、全库双语校验、短护栏与独立 `--basetemp` 全量护栏。按约定改完**先不 git commit**，等用户确认。
