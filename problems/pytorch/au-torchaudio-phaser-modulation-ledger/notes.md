# 872 · 实现笔记

## 划界与排除

- 只判 `phaser` + `_generate_wave_table`；`vad` / `_measure`（SoX 倒谱能量门控）、`overdrive` / `filtfilt` 留作 873+ 候选。
- 871 flanger、869 biquad、870 dither、864 speed/pitch、716 FSMN-VAD 均不重判；本题不引入真实音频 IO、不重判 IIR 递推。

## 数值与语义

- INT 波表先 `d ± 0.5` 再 `.to(int32)` 向零截断；FLOAT 分支强制 binary32，必须逐位模拟而不能只做数值近似。
- `delay_pos` 与 `mod_pos` 是两个独立相位，按不同模数推进；先读 `idx` 旧值再写 `delay_pos`，顺序不能颠倒。
- `delay_buf = delay_buf * decay` 让负 decay 下初始缓冲为 `-0.0`；`_clean` 递归归一 inf/nan 但保留零的符号。
- 非法 `wave_type` / `data_type` 走官方「无 else」静默分支；`clamp` 饱和 ±inf，只有 nan 传播。一维输入被 `view(-1,T)` 接受，通道数无上限。

## 工程教训

- `int(x+0.5)` 的判定必须挑跨舍入边界的小数参数，整数参数网格会让篡改存活。
- `mod_buf_len=2` 时 SINE 波表恒为常数，不同 `sample_rate` / `mod_speed` 会输出相同；硬化要用非常数波表尺寸与长波形。
- torch 向量化 `sin` 与 libm 在 n=7/13/14/... 差 1 ulp；精确 `Fraction` 推 binary32 也会在中点差 1 ulp，应把「精确代数性质」与「数值容差对拍」分开断言。
- 真等价变异不删除，而是对完整 `solve()` 输出做类型敏感、float 逐位比较留证。
