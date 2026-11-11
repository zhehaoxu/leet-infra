# 实现笔记：867 `create_dct` 构造账

## 1. 官方环境

`create_dct` 从 `functional.py` 636-667 行用 AST 抽取后 `exec` 进共享 NS，`torch.set_default_dtype(torch.float64)`。无 Proxy、无 RNG、无音频 IO、无 warning。`comp` 实现 `matrix` / `norm_error` / `census` 三路，与参考解一一对应。

## 2. 参考解

纯 Python 只用 `math`。`pi_over_nm` 提到循环外（复刻官方顶层求值，使 `n_mels=0` 抛 `ZeroDivisionError`）；构造矩阵 `(n_mfcc, n_mels)`，再转置成 `(n_mels, n_mfcc)`；`norm=None` 全矩阵 `×2`，`norm="ortho"` 先 `dct[0]`（构造第 0 行）`×1/sqrt(2)` 再整张 `×sqrt(2/n_mels)`。三个 mode 与官方 `comp` 一一对应。

## 3. 独立 oracle

`verify867.py` 第三层用泰勒级数（递推 `term *= -x^2/((2k+2)(2k+1))`）独立计算 `cos`，不依赖 `math.cos`；再用代数性质校验方阵 `norm="ortho"` 时 `D @ D.T == I`。192 组配置 × 3 mode 共 87272 项比较零差异。

## 4. 常量与判题

31 个官方常量 + 37 个 `check_*`：默认三 mode、双 `norm` 全部矩阵、错误字典、转置形状、四个角点、正交性、精确余弦与 `n_mels=0` / `n_mfcc=0` 边界。

## 5. 变异

42 项变异，42 项全 CAUGHT，0 项被判为真等价；硬化重点是转置方向、正交首列与 `n_mels=0` 顶层报错。无等价留证脚本（零存活）。

## 6. 查重与划界

`create_dct` 零命中；`audio-mfcc` 仅把正交 DCT-II 应用于 MFCC 流水线一步，855 notes 只提抽取手法，867 判矩阵与 `norm` 口径。`n_mels=0` 的 `ZeroDivisionError` 与 `n_mfcc=0` 的空矩阵属实现细节 / 退化输入，明确排除或按 0 列处理。

## 7. 交付与约定

落地 11 文件、selftest 双模式、双语校验、注册 845 题、短护栏与独立 `--basetemp` 全量护栏、日志回填。改完先不 git commit，等用户确认。
