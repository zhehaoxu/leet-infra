# FP8 E4M3 量化往返 详解

## 为什么是 FP8、为什么是 E4M3

FP8 一字节存浮点，显存/带宽减半、tensor-core 吞吐翻倍，是 H100 `transformer-engine`、DeepSeek 训练的做法。两档子格式：**E5M2**（范围大、精度低，多用于梯度）与 **E4M3**（精度高一档、范围小，多用于前向激活/权重）。E4M3 尾数只有 3 位 → 相对精度约 $2^{-4}=6\%$，靠 FP8 前向 + 高精度累加（FP32 accumulator）保住总误差。

## 手工格点量化的三块硬骨头

1. **按 binade 变步长**：规格化值落在 $[2^e, 2^{e+1})$ 时，格点间距 $2^{e-3}$（3 位尾数 → 一段 8 格）。`frexp` 取实际指数最稳，别用 `log2` 再 floor（边界浮点误差会错段）。
2. **次正规固定步长**：$y<2^{-6}$ 后不再随段缩小，统一间距 $2^{-9}$（$=2^{E_{\min}-1-2}$ 的标准结论），这保证 0 附近等距、RTNE 良定义。把次正规也按段算是最常见 bug。
3. **饱和 vs 进 NaN**：`float8_e4m3fn` 的 `fn` 表示「无 inf」，$E{=}15,M{=}7$ 被 NaN 占用，最大有限 $=448$。RTNE 本想舍到 $480$（不存在）→ torch **饱和到 448**。所以 `465 → 448` 而非更大。

## 平局取偶（RTNE）

Python 内建 `round()` 就是「四舍六入五取偶」，正好是 IEEE round-to-nearest-even：`round(0.5)=0`、`round(1.5)=2`、`round(2.5)=2`。用它对 `a/step` 取整即与 torch cast 对齐。用 `int(x+0.5)` 之类的「一律向上」会在平局处与 oracle 差一格。

## 与整型量化的对照

`symmetric/per-channel/group-wise` 把范围映射到**等距整数格点**（均匀量化）；FP8 是**非均匀浮点格点**——小值密、大值疏，天然适配动态范围大的数据，也无需 scale。`fp16-loss-scaling` 处理的正是「低位宽浮点的上/下溢」，与本题的饱和/次正规是同一主题的 fp16 vs fp8 两面。

## 判题

- 内置 oracle：`x.to(torch.float8_e4m3fn).to(dtype)`，`check_matches_oracle_*` 对拍大随机网格 + 次正规 + 饱和 + 平局（**精确相等**，两侧都落到同一格点值）。
- `check_example` 用 dump 的具体映射；另有 float64、零、不修改输入用例。
