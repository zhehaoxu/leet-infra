# notes — asr-cif-fire-emit (order 694)

## 出处

- 官方源码：modelscope/FunASR `funasr/models/paraformer/cif_predictor.py`
  的模块级 `cif(hidden, alphas, threshold)` 与 `CifPredictor.tail_process_fn`
  （GitHub `main`，2026-11 抓取）。论文：CIF 1905.11235。
- 目录探索：FunASR 无独立 predictor 包；`CifPredictor`（paraformer 系）、
  `CifPredictorV2`（bicif，双 fire 阈值）、`CifPredictorV3`（cnn 版）——
  本题取最经典的 CifPredictor + 模块级 cif()。前处理链
  （conv1d 平滑 → sigmoid → smooth_factor/noise_threshold → alphas）不考，
  alphas 显式入参。
- 调用语义（forward, target_length=None）：alphas → tail_process_fn
  （追加 tail 帧 + 零行 hidden，token_num=floor(sum)）→ cif() →
  截取 [:token_num]。单序列下 pad/截取无观察差异，接口直接输出 embeds。

## 算法逐条账

1. **fire 闭区间** `integrate >= threshold`：恰达 1.0 也发射（锚点
   [0.2]*5 与 [1.0] 单帧）。
2. **cur = 缺口**：fire 时 `cur = 1 - integrate_before`，即 cumsum 跨第
   k 个整数时权重取 `k - c_{t-1}`；`remainds = c_t - k` 成为下一枚的
   起始权重。非 fire 帧 cur = 整个 alpha。
3. **快照在减 1 前**：`list_fires.append(integrate)` 位于 `integrate +=
   alpha` 之后、减 1 之前。若误用减后值，恰达阈值的发射（快照 0.0）
   丢失——`snapshot-removed` 变异可观察（fire_positions 恒空）。
4. **frame 语义**：非 fire 帧 `frame += alpha*h_t`；fire 帧先加上缺口
   部分再发射，随后重置为 `remainds*h_t`。发射嵌入 = 发射前累计完成
   的加权和。
5. **tail 机制**：`tail_process_fn`（mask=None 分支）给 alphas 追加
   `tail_threshold` 一帧、给 hidden 追加一行零向量，且**帧号对齐**：
   tail 帧（index T）恰用补的零行。补缺口权重乘零 → 末 token 嵌入
   只由真实帧累计。**官方不变量 len(hidden) == len(alphas)** 是出题时
   最重要的输入约束——初版测试给了 hidden 比 alphas 多帧的输入，tail
   帧错位到真实帧上，断言失败后才钉死该不变量并写进 starter。
6. **token_num 双口径**：use_tail → `floor(sum_ext)`（tail_process_fn
   返回 floor）；无 tail → `round(sum)`（cif() 内 len_labels）。单序列
   use_tail 下恒等式 `发射数 == floor(sum_ext)` 成立（fire 后余量 < 1
   的归纳）；无 tail 时发射数可 < round(sum)（锚点 [0.6]*3：1 vs 2）。
   batch 场景 len_labels 还决定 pad 宽度、forward 再截 [:max(floor)]，
   单序列无观察差异，不出题。

## oracle 设计

解析法独立实现：cumsum c_t，帧 t 贡献半开区间 [c_{t-1}, c_t)；第 k 枚
token 的权重 = 区间与 [k-1, k) 的相交长度；fire 帧 = 首个 c_t ≥ k 的帧。
不经状态机。随机对拍用 **m/256 网格**（2 的幂分母，浮点精确，累积无
误差，边界判定无翻转）300 组，另断言 use_tail 下发射数 == token_num
恒等式与 fire_positions 严格递增。

## 变异记录

13 条全抓（首轮 1 条 + 修正后 13 条；首轮 `snapshot-after-decrement` 的
替换串截断 if/else 报 IndentationError，改为 `snapshot-removed` 删除快照
行——fire_positions 恒空立即可观察）。覆盖：fire 闭/开区间（两处判定）、
快照存在性与时机、cur 取缺口/整 alpha、缺口计算时序、余量重置/清零/
不重置、integrate 不减 1、tail 不追加/零行缺失、token_num floor↔round
互调、embeds 倒序、remainds 不跟踪。无等价变异。

## 教训记录

- 出题初期 3 个锚点手算错（[0.9]*6 的逐帧 fire 序列、tail 零行对齐、
  H 帧数与 alpha 数不匹配），全部靠打印实际值定位；`0.9` 连续帧每帧都
  跨阈（fire 后余 0.8 + 0.9 = 1.7 ≥ 1）是最反直觉的一个。
- tail 帧零行对齐要求 `len(hidden) == len(alphas)` 不变量成立后锚点才
  自洽——入参约束本身是账点。

## 查重

- todos ⑤ 原文"integrate-and-fire、fire 阈值 1.0 发射、尾部余量按比例
  分配"——实际官方无尾部分配，是 tail_threshold 追加帧补发；按源码
  纠正（todos 已同步）。
- 与 `ctc-loss`（loss 侧 DP）、RNN-T（候选 ④，前向 DP）机制不同：
  CIF 是推理侧的加权累积状态机。与 `spec-decoding` 无关。
- FunASR 的 CifPredictorV2（bicif）双阈值不考；`gen_frame_alignments`
  （batch 对齐矩阵）实现繁杂且单序列无意义，不考。
