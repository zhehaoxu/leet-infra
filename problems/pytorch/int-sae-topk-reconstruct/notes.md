# 详解：SAE top-k 编码账

## 出处

`jbloomAus/SAELens` 的 TopK SAE 架构（`sae/sae.py` 的 `encode`/`decode`：`pre_acts = W_enc(x − b_dec) + b_enc` 后取 top-k、**不做 ReLU**；`decode = acts @ W_dec + b_dec`）。隐态与权重全显式入参，符合「需真实大模型激活的题不出」红线。JumpReLU 门槛变体附在 notes（用固定阈值 θ 替换 top-k 选择，其余账同构），未单独立题。

## 核心与易错点

1. **不做 ReLU 是 TopK SAE 与 ReSAE/普通 SAE 的分界**：负的 pre-act 照样入选、负值照常参与重构——「top-k 后 clamp_min(0)」与「top-k 前 clamp_min(0)」两个变异都被 `negative_pre_acts_kept` + 随机对拍抓。
2. **b_dec 的双重角色**：既在 encode 端被减（`x − b_dec`），又在 decode 端加回——只减不加（或反之）都被 `k_equal_F` 的恒等断言（k=F 时 acts=完整 pre_acts、recon=pre@W_dec+b_dec）与随机对拍抓。
3. **并列 tie-break**：`torch.topk` 值相等时取较小 index；oracle 用 `sorted(key=(-pre, f))` 复刻。`dead_features` 用四个等值 1.0 的 pre-act 钉死 `[0,1]`。
4. **l0 的边界**：`min(k, F)`（k=F 时全部非零——除非某些 pre-act 恰为 0）；`max_feat` 按**绝对值**最大（负激活幅度最大者也算），与 `argmax`（带符号）不同——变异 `abs-argmax-tie` 靠随机负激活抓。
5. **mse 对 D_in 求均值**（不是 sum）——`anchor_toy` 的 0.5 锚点钉死。

## 变异测试（8/8 全抓）

encode 去 `−b_dec`、top-k 后 ReLU、top-k 前 ReLU、decode 去 `+b_dec`、k 减半、mse 换 sum、max_feat 去绝对值、recon_error 符号反转。

## 复杂度

O(B·F·D) 编码 + O(B·F·D) 重构，规模 ≤ 2×10×6，毫秒级。
