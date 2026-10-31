# notes — tts-rvq-residual-quant（order 673）

## 回源

- 出处：`facebookresearch/encodec` 的 `quantization/core_vq.py` / `bqrw`（ResidualVectorQuantization：`forward` 逐级 `encode` 残差；SoundStream 2107.03334 同构；Mimi/DAC/Qwen3-TTS 的 code-to-wave 同核）。本题取**推理侧逐级最近邻账**（不含码本学习/EMA/commitment loss）。
- 口径要点：每级最近邻用 **L2²**（等价 L2 排序，省开方）；并列取最小下标（torch.argmin 语义）；残差在码本条目**减法后**再算本级 residual_sq（本级吸收量已扣除）；重建向量 = 各级被选条目按级递增顺序求和。
- 与 quantization 合集划界：不考标量 quant/dequant、码本训练；与 #110 的 Qwen2-Audio token 展开划界：那是 token 数账。
- llama.cpp 关联：Qwen3-TTS（PR 20397 系）与 Pocket-TTS 的 RVQ decode 即本题 codes → reconstructed 的逆过程（编码账为其镜像）。

## 判题面

- solve 入参：latents [N,D]、codebooks [K,C,D]。返回 codes [K][N]、reconstructed [N,D]、residual_sq [K][N]。
- oracle：`test_cases._py` 独立实现（min+index 取码、recon 后组装），150 组随机对拍（N≤5、D≤3、K≤4、C≤5）codes 严格 + 浮点 1e-9。
- 锚点：1-D 两级手算（1.05 → 级0 取 1.0 残差 0.05 → 级1 归零；残差浮点 ~1.7e-33 用 1e-30 容差）；级间级联修正方向；等距并列取小下标；多样本独立；2-D 正交码本手算；残差能量逐级严格下降。
- 非法输入：空 latents/codebooks、空码本级、维度不齐、非数/NaN/Inf/bool → ValueError。

## 变异记录（9 条全抓）

tie-max-index（并列取大下标）、no-residual-update（残差不更新）、level-reverse（级序反转）、dist-no-sq（L2² → L1）、resid-no-update-sq（residual_sq 恒 0）、recon-minus（重建变减法）、codes-order-swap（级序输出互换）、dist-use-recon（距离丢码本项）、first-level-skip（跳过级 0）。

## 红线

码本学习/EMA/kmeans 不出（码本全入参）；真音频 IO 与波形合成不出；与 quantization 合集（标量量化）划界；与 #110 音频 token 展开划界。
