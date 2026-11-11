# 实现笔记：torchaudio BS.1770-4 响度门控账

## 1. 选题与查重

order 861，合集 `audio` 12→13，slug `au-torchaudio-loudness-gate-ledger`。扫描全库：`max_order=860` / 838 题 / order 唯一 / 861 空闲 / slug 未占用；`loudness` / `LKFS` / `BS.1770` / `biquad` / `lfilter` 在全部 `meta.json` 零命中，description/notes 里只有 860 的一句候选避让说明。与 855（频谱后处理）、857（波形算子）、858（NCCF）、859（sinc 重采样）、860（Griffin-Lim）同源于 `functional.py` / `filtering.py` 生态但函数集合完全不重叠。

## 2. 逐字执行环境

`functional.py` 是 855 已落盘的 2511 行官方文件；`filtering.py` 于 2026-11-08 从 GitHub raw 端点回源，把九个函数/类逐字誊到 `filtering861.py`（只有模块级的 `_IS_TORCHAUDIO_EXT_AVAILABLE` 开关由执行环境注入，函数体不动）。`official861.py` 用 `ast.get_source_segment` 抽 `loudness` + 八个 filtering 定义，`exec` 到共享 `NS`，再令 `NS['_lfilter_core_loop'] = NS['_lfilter_core_generic_loop']`。
冒烟输出：sr=8000 时 treble `a=[3.625826704781014, -1.2110558364958972, 0.6938511881834482]`、highpass `a=[1.0298406997386809, -1.9991093343177666, 0.9701593002613191]`；方波经两级 biquad 后 `gate_samples=3200` / `step=800` / `n_blocks=3` / `LKFS=-9.97723766646316`。

## 3. 流水线结果

```text
12-segment probe      PROBE 861 DONE ok=261 fail=0 ALL GREEN
three-layer verify    VERIFY 861 DONE checks=931432 fails=0 ALL GREEN
judge cases           DRIVER 861 50/50 ALL ACCEPTED
mutation             199 total / 6 equivalent / 193 CAUGHT / 0 SURVIVED
precheck              PRECHECK 861 35 PASS / 0 FAIL
```

三层对拍的独立 oracle 用 `math.pow(10, gain/40)` 求 `A`、用显式 `xp=[0,0]+x` / `yp=[0,0]+zeros` padded 数组配合 `math.fsum` 重写 FIR + IIR，再用块优先的循环重算门限账；与官方逐字环境、纯 Python 参考解三者逐值一致。931432 是递归标量比较次数（含 16000 点波形的逐元素比对）。

## 4. 变异测试与真等价

首轮 197 项：177 CAUGHT / 17 SURVIVED / 3 ANCHOR-ERROR。锚点错误是三处重复字符串未加出现序号（`return out` 在 `_coeff_block` 与 `_filter_row` 各一次、`return led` 在 `_dual_gate` 与 `_mode_loudness` 各一次、8 空格缩进的 `must be rectangular` 同时是 12 空格版本的子串）；补 `idx` 后消除。硬化补丁新增 8 组用例：F7（-69.565 LKFS 卡在 -70 与 -60 之间）、F8（`T == gate_samples` 恰一块）、F9（有限能量但全块低于 -70）、F10（5 块被绝对门限剔除却仍高于 `gamma_rel`）、F3L（sr=35 银行家舍入 step=4）、sr=1 的报错消息、默认 clamp、空系数组消息，并给 F5 补 `n_gated_abs` / `ew_abs` 断言、给 R2/R3 补多声道 `shape` 断言。
判定 6 项真等价并逐条留证：① `MODES` 是文档常量，全文件仅出现 1 次（定义处）；② `math.log10(nan)` 本来就返回 nan，删掉 isnan 早退等价；③④ `_log10` 的三个调用点只可能是能量（平方和乘非负权重）或 nan，负数分支不可达，故 `v<0` 判断与其返回值均为死代码；⑤⑥ `require_matrix` 保证 `filtered` 非空，`len(filtered)<1` 分支不可达。最终 **193/193 CAUGHT，0 SURVIVED，0 ERROR**。

## 5. 剥除口径与工程教训

① 判题域只取二维 `(channels, time)`，官方批次维剥除，0 维 LKFS 张量剥成 float；② `blocks` 接收已滤波波形，使门限账可脱离 biquad 独立判题；③ float32 / 扩展 C++ kernel 路径只记录不判，数值统一 float64；④ `pad` / `batching` 等 `lfilter` 的 2-D 系数分支不在判题域；⑤ 判题辅助 `_near` 必须是递归 `assert` 而非返回 bool（order 860 的教训，本题从一开始就按此实现）；⑥ `sr>=4000` 才跑波形：更低采样率下 1500 Hz 架式滤波器的极点跑到单位圆外，官方 torch 与纯 Python 都会溢出，系数 mode 仍可判。

## 6. 官方 quirk 与查重结论

与 `description.md` 第 8 节一致：size(-2) 声道检查、alpha 的两种写法、A 的 exp 形式、clamp 不反馈、两级各 clamp 一次、g 的 dtype、count_nonzero 的形状、严格 -70、gamma_rel 只吃绝对门限块、logical_and、全剔除时 nan、四个字面量常量、不校验 Nyquist。另需订正 860 notes 的说法：`loudness` 与 `phase_vocoder` 没有任何调用关系。

## 7. 注册与护栏

`register861.py` 最小侵入注册 `audio` 12→13、中英 description 各追加 861 口径账长段；复核 `839 registered = 839 disk / max_order=861 / 无 dangling / 无 unregistered`。`todos.md` 更新为 839 题 / audio 13 / 下一批从 862 起。双语全库校验预期 `0 error / 15 warning`（15 条全为历史既有）；短护栏用独立 `--basetemp`，全量护栏后台运行后回填结果。
