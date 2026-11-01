# Moshi 多流 delay 账（notes）

回源：kyutai-labs/moshi `rust/moshi-core/src/lm_generate_multistream.rs`（Config::v0_1：
8 生成 + 8 输入 codebook、vocab 2049、audio_pad = vocab-1 = 2048、acoustic_delay 2、
text_pad 3、text_eop 0、text_start 32000）。

- **写侧 delay 不对称**：cb0 零延迟写（实时流）、cb1-7 滞后 2 写；输入半 cb8-15
  每帧直写（无延迟）。读侧（模型输入 codes）另有 cb0/cb8 零延迟读、其余读
  step-delay-1 帧的口径（本题不考读侧，入参剥离）。
- **覆盖链**：saturating_sub 使 step 0..2 的 cb1-7 都写 slot0——官方注释
  "Overwrite existing positions even if there are non-UNGENERATED values"。
  锚点 slot0 = [step0.cb0, step2.cb1-7]。
- **播放判定三重**：step ≤ delay 前 None；读帧 step-delay-1；帧内任一
  audio_pad（含 dep 直接采样出 2048 或 None 兜底写）→ 整帧静音。any→all 与
  阈值 >=→> 的变异都被固定锚点抓。
- **UNGENERATED**：官方矩阵初始 u32::MAX，读侧遇它会 bail（本题入参保证 dep/input
  覆盖所需格；残留格只出现在尾部与短输入流，播放判定 -1 < 2048 不触发静音——
  出题约定已写进题面）。
- 等价变异：`list(frame)`→`frame`（slot 记录后无后续写命中）。文本流（text_pad 3
  的 pad_mult 采样偏置、repetition penalty 的 non_pad 跳过）属采样侧，未入题。
