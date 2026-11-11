# 实现笔记：866 Mel / 线性滤波器组刻度账

## 1. 官方环境

五个函数从 `functional.py` 425-633 行用 AST 抽取后 `exec` 进共享 NS，`torch.set_default_dtype(torch.float64)`。无 Proxy、无 RNG、无音频 IO。`_call` 用 `warnings.catch_warnings(record=True)` 捕获 warning 与异常。

## 2. 参考解

纯 Python 只用 `math`。对称 `linspace` 逐位复刻 torch；`_safe` 把异常收敛为 `{type,msg}`；`_fbank` 同时产出矩阵、warning 与完整 ledger；十个 mode 与官方 `comp` 一一对应。

## 3. 独立 oracle

`Fraction` 有理数 `linspace`、`log1p`/`expm1`/`pow` 变换、几何分段三角形、集合成员列普查。69 组配置 × 10 mode 共 37578 项比较零差异。

## 4. 常量与判题

121 个官方常量 + 123 个 `check_*`：默认十 mode、双刻度全部断点、错误字典、三角矩阵、floor division、`norm`×`mel_scale` 四组合、warning 文本、精确 `linspace` 与端到端重算。

## 5. 变异与等价

168 项变异，160 项全 CAUGHT；8 项真等价由 `equiv866.py` 在 270 配置 × 10 mode 上逐项证明（21600 次比较零差异）。硬化重点是精确相等断言与 `linear_fbanks` 的两个静默分支。

## 6. 查重与划界

五函数零命中；342 把 Slaney 与归一绑定，852 是 Kaldi/VTLN 兼容层，866 判刻度中间量、`linear_fbanks` 与 warning。`n_freqs=0` 的 torch `max()` 细节排除在输入域外。

## 7. 交付与约定

落地 11 文件、selftest 双模式、双语校验、注册 844 题、短护栏与独立 `--basetemp` 全量护栏、日志回填。改完先不 git commit，等用户确认。
