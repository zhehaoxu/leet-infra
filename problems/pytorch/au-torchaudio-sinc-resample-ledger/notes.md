# 859 解答笔记：torchaudio sinc 重采样账

## 1. 官方回源

### 源码清单与依赖

```text
.workbuddy/scratch/captum855/functional.py   2511 lines / 95067 bytes
.workbuddy/scratch/torchaudio859/official859.py   ast x3
WANT = [_get_sinc_resample_kernel,
       _apply_sinc_resample_kernel, resample]
NS injects only: math typing warnings torch Tensor Optional _CPU
```

### 逐字执行环境

三个函数整段 `exec` 到共享 `NS`，驱动层为 `comp_kernel` / `comp_apply` / `comp_resample` 与 `raw_kernel` / `raw_apply` / `raw_resample`；后者暴露 `dtype` 与 warning 行为，用于 P2 / P10 的分支探针。全部张量默认 float64。

## 2. 十二段官方机制探针

### 探针结论

P1 钉 gcd 约分、`base_freq=min(o,n)*rolloff`、`width=ceil(W*o/base)`、`K=2*width+o`；P2 钉弃用别名、非法方法、非整数频率、`W<=0` 与 `resample` 的 `<=0` guard；P3 钉恒等短路返回原对象。
P4–P5 钉 t 网格行偏移 `-j/n`、clamp 饱和与 Hann 边界的 `cos(pi/2)**2` 残差；P6 钉 beta 的 float32 量化、float32/float64 双 `torch.i0` 路径与 `1/i0(beta)` 边界；P7 钉 `t==0` 的 sinc 分支与 `scale=base/o`。
P8–P9 用独立定义式手算核对 Hann / Kaiser 核矩阵（含 44100→48000 的 160 相位）；P10 钉 float64 / `dtype=None` / float32 三条路径互不相同；P11 钉 pad、stride 卷积、交错、截断与 1D/2D/3D unpack；P12 钉端到端与恒等短路。
首轮两处断言写错并按官方实测订正：① 弃用别名 `sinc_interpolation` 并非映射到 Hann，而是落进 Kaiser 分支；② `<=0` guard 的判据字符串在核构造器里也有 "should be positive"（来自 `W<=0`），故改用 `Original frequency and desired` 唯一定位。最终 `PROBE 859 DONE ok=124 fail=0`。

## 3. 三层对拍

### official 层

逐字三函数 + torch float64 张量；`raw_kernel` 返回 `(n,1,K)` 核，对拍时 squeeze 掉恒 1 的通道维，与判题口径的 `(n,K)` 对齐。

### reference 层

纯 Python `solution_staged.py`，五个 mode；`f32` 用 `struct` 复刻 `torch.tensor(float(beta))`，`i0_series` 用幂级数递推，`pack` / `unflatten` 处理任意前导维。

### oracle 层

独立定义式：`math.fsum` 累加级数项、显式逐 tap 生成、`zip` 切片点积、独立的形状/展平/折回 helper，不复用参考解任何函数。`t` / `window` / `sinc` 是账本视图，官方不暴露，故只做 reference↔oracle 两层核对。

### 对拍结论

```text
VERIFY 859 DONE checks=727 fails=0 ALL GREEN
9 parameter cases x (meta + kernel + 4 waveforms x apply/resample)
i0 float64 column: official / reference / oracle
```

## 4. 判题用例与 driver

`test_cases_staged.py` 有 35 个顶层 `check_*(solve)`，全部使用官方逐字环境实测的显式常量：`K84` / `KK84` / `K48` / `K69` / `K84B3` 五组核快照、`A84_8` / `A84_11` / `A48_8` / `R69` 四组输出快照，以及 meta / i0 / 别名 / 恒等 / 错误消息的结构断言。
首轮判题 2 红均为期望值写错（参考解始终与官方一致）：`base_freq` 在 8→4 时是 `min(2,1)*rolloff = rolloff` 而非 `2*rolloff`；`i0_float32` 列的判题口径是「精确级数再量化」，我误填了 `torch.i0(float32)` 的 Cephes 值。订正后 `DRIVER 859 35/35 ALL ACCEPTED`。

## 5. 变异测试

`mut859.py` 构造 196 项变异，覆盖常量、helper、`params` / `meta_view` 全部字段、五个 mode 的公式与校验、`solve` 分发。锚点重复处用 `ANCHOR_INDEX` 指定出现位置，预检保证 0 个坏锚点。

```text
round 1: 170 CAUGHT / 25 SURVIVED / 1 ERROR
patch  : +6 hardening checks, 8 equivalence rulings
round 2: 187 CAUGHT / 1 SURVIVED / 0 ERROR
round 3: 188 CAUGHT / 0 SURVIVED / 0 ERROR
196 total / 8 equivalent / 188 active -> ALL MUTANTS CAUGHT
```

首轮 25 项 SURVIVED 的成因：默认参数未测、错误消息只校验关键词不校验全文、类型分支（bool / 字符串 / 积分浮点）未测、sinc 账本列未与 window 列区分、恒等短路的 `meta` 未断言、未知 mode 未断言。追加 6 组硬化用例后逐一收口。
8 项真等价：判题域内 `shape_of` 永不接收空列表；`unflatten` 在 `shape==[]` 时切片恒为单元素；`require_numbers` 的第二道 guard 仍以同一消息拒绝空数组；`is_number` 已排除 bool 使 `require_freq` 的显式 bool 检查冗余；I0 级数用 `x*x` 故符号无关；`pack` 的输入必已过 `require_numbers` 故 rank guard 不可达；交换私有 `orig`/`new` 字段不改变任何公开输出（相等判断对称且 `_kernel` / `_apply` 重读原始 kw）。

## 6. 剥除口径

① 官方核 `(n,1,K)` 剥成 `(n,K)`，`meta.shape` 保留官方三维；② 判题域固定 float64，`dtype=None` / float32 两条降级路径只记录不判；③ Kaiser beta 仅取默认值或与 `torch.i0(float32)` 逐位一致的 3.0 / 12.0；④ `i0_float32` 定义为 `f32(I0_float64(x))`，不是 `torch.i0(float32)`；⑤ 警告文本简化为 `deprecated: old -> new` 标记串；⑥ `rolloff` 要求为正、`beta` 要求为数值，这两条是判题域约束而非官方 guard。

## 7. 官方 quirk 清单

与 `description.md` 第 9 节一致：弃用别名落错分支、beta 的 float32/float64 双路径、Hann 与 Kaiser 的边界残差、三条 dtype 路径、恒等短路返回原对象、`<=0` guard 只在编排层、`ceil(n*L/o)` 截断、以及与警告文本自相矛盾的行为。

## 8. 查重与落地

落地前确认 `max_order=858`、`sinc_resample` / `sinc_interp_kaiser` / `griffinlim` / `spectral_centroid` 全库零命中；与 344 多相重采样按算法族划界，与 855 / 857 / 858 同源但函数集合不重叠。11 文件由 `land859.py` 落地，三件套与 `variants/pytorch/` 用 `filecmp` 校验一致。
