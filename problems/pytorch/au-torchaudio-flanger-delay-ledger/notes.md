# 871 · 实现笔记

## 划界与排除

- 只判 `flanger` + `_generate_wave_table`；`phaser`（全通链 + INT 波表）、`vad` / `_measure`（SoX 能量门控）、`overdrive` / `filtfilt` 留作 872+ 候选。
- 864 speed/pitch、716 FSMN-VAD、869 biquad、870 dither 均不重判；本题不引入真实音频 IO、不重判 IIR 递推。

## 数值与语义

- `torch.frac(x)=x-trunc(x)` 保留符号；`torch.floor(x)` 向负无穷取整。负 delay 时两者不一致，插值变成外插，这是最容易写错的一处。
- 波表 `FLOAT` 分支强制 float32；`INT` 分支不降精度且用 `d±0.5` 后截断。binary32 舍入必须逐位模拟，不能只做数值近似。
- 非法 `wave_type` / `data_type` 走官方的「无 else」静默分支，不抛异常；`clamp` 把 ±inf 饱和到 ±1，只有 nan 传播并归一为字符串。
- 环形缓冲 `pos=(pos+L-1)%L` 递减、L 步一循环；`lfo_pos` 每步 +1 并取模。前缀输出必须与完整输出的前 n 列逐位一致（因果性）。

## 工程教训

- 短波形会让延迟环形缓冲写不满，使不同 `delay` 扫描值输出恰好相同；mode 级检查必须配长波形 + 高采样率才能暴露扫描表被篡改。
- 负 phase 才能区分 `int()`（向零截断）与 `math.floor()`（向下取整）；非负参数网格会漏掉这类分支。
- 三层 oracle 用精确 Fraction 推导 binary32 时，官方先算 float64 中间值再降精度，舍入中点会出现 1 ulp 差异；应把「精确代数性质」与「数值容差对拍」分开断言。
- 真等价变异不删除，而是对完整 `solve()` 输出做类型敏感、float 逐位比较留证。
