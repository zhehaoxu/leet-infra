# notes（quant-fp4-e2m1-grid）

## 回源

- 元素格式：OCP **MX v1.0** 规范的 FP4 `E2M1`——1 符号 + 2 指数 + 1 尾数、`bias=1`、
  **无 inf / 无 NaN**、最大有限值 `6`、次正规只有 `0.5` 一个非零码。位域解码：
  `e=0` 时 `mag = m/2`（`0.5`），`e>0` 时 `mag = (1 + m/2)·2^(e-1)`。
- 缩放：MXFP4 是「每 32 元素一个 E8M0 共享指数」（单级）；NVFP4（Blackwell）是
  「每 16 元素一个 E4M3 块 scale × 每张量一个 FP32 scale」（两级）。本题把**已解码的
  scale 值**作入参，只考元素格式与两级乘积口径——E8M0 编码本身归 `quant-mx-microscaling`(612)，
  E4M3 往返归 FP8 题(125/445)，都不重复出。
- 判题环境无 FP4 硬件，也没有 `torch.float4_e2m1fn_x2`，故按位域公式自建参照表对拍。

## 官方实现里的坑

1. **`6.0` 不饱和，`6.0000001` 才饱和**：判据是 `|scaled| > 6`，用 `>=` 会把精确的 6
   也记成饱和，`sat_mask` 与 `block_sat` 全错。
2. **没有 `inf` 码也没有 `NaN` 码**：`inf` 只能饱和到 `6`；`NaN` 无码可编，官方量化的
   输入契约是「不得含 NaN」，本题显式 `ValueError`。
3. **次正规只有 `0.5`**：`e=0` 的尾数只贡献 `m/2`，所以 `0.1 / 0.2 / 0.25` 只能回 `0`；
   把网格写成「`0.25` 起、步长 `0.25`」是最常见的错。
4. **RTNE 的「偶」是尾数位偶数，不是幅值偶数**：`1.25 → 1.0`（尾数 `0`）、`3.5 → 4.0`
   （`4.0` 的尾数位也是 `0`）。恰好因为码值下标奇偶 == 尾数位，判「平局取偶」时可以直接看下标。
5. **`-0.0` 有独立码 `8`**：`v < 0` 判符号会漏掉负零，`torch.signbit` / `math.copysign`
   才分得清；负数幅值小于 `0.25` 全部落 `-0.0`。
6. **两级 scale 是乘积**：`eff = global · block`，不是 `(v / block) / global`。两者在
   非 2 的幂 scale 下差最后几个 ULP，本题按乘积口径钉死（`eff_scale` 字段让差异可观测）。

## 本题的规范化

- 码值布局固定 `code = (sign << 3) | (exp << 1) | mant`，`dequant = ±GRID[idx] · eff`。
- 平局判据用**精确浮点相等**（中点 `0.25/0.75/1.25/1.75/2.5/3.5/5.0` 全部二进制精确），
  并把 `tie_mask` 暴露出来，避免「结果碰巧对但过程不同」的假通过。
- 校验门：`block_size` 正整数（`bool` 拒收，且单独立用例防被长度检查掩盖）、
  `block_scales` 长度 == 块数且每项有限正数、`global_scale` 有限正数、`values` 无 `NaN`；
  空输入合法（0 块 + 空 scale）。
- 返回的 8 个张量各自独立存储，`dequant` 原地改动不影响 `codes` / `eff_scale`。

## 变异

16 条全部 CAUGHT，无等价变异：

```
m-tie-odd / m-tie-to-lo      平局取奇 / 取低   → RTNE 全表用例抓
m-sat-ge / m-sat-code        饱和门含等号 / 码值回 6 → sat_mask + codes 抓
m-maxv                       6.0 → 6.5          → 6.0000001 不再饱和，抓
m-grid-subnormal             0.5 → 0.25         → 次正规与 grid 用例抓
m-no-negzero                 copysign → v < 0   → -0.0 符号抓
m-eff-sum / m-eff-no-global  eff 用加法 / 丢 global → 两级缩放抓
m-dequant-sign               反量化丢符号       → 负数 dequant 抓
m-nearest-lower              最近邻恒取下界     → 0.4 / 0.26 用例抓
m-block-wrap                 块号用取模         → 块壁用例抓
m-no-len-check               去掉长度校验       → validation 抓
m-bs-bool                    漏 bool 拒收       → 单独立用例抓（首轮曾被长度检查掩盖）
m-nan-ok                     放行 NaN           → NaN 用例抓
m-block-ties-sat             tie 计数记成 sat   → block_ties 抓
```

## 查重

- `quant-mx-microscaling`(612)：考 E8M0 共享指数编码本身，notes 明示不考下游元素格式；
  本题不碰 E8M0 编码，只吃已解码的 scale 值。
- FP8 往返/分块缩放(125/445)：E4M3、E5M2 的位宽与 scale 递推，非 4-bit 元素网格。
- `quant-uint4-asym-pack`：4-bit 非对称整数打包（整除字节）；本题不考字节重组。
- `quant-afpq-dual-scale-fp4`：把 FP4 码表当抽象集合做双侧 scale 搜索；本题是 E2M1 位级
  格式语义 + 两级 scale 的乘积口径，不做搜索。
