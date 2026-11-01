# notes — asr-specaugment-mask (order 693)

## 出处

- 官方源码：modelscope/FunASR `funasr/models/specaug/mask_along_axis.py`
  的 `mask_along_axis`（GitHub `main`，2026-11 抓取；espnet 同名函数同源，
  FunASR 是 Paraformer 训练实际使用的版本）。配套 `specaug.py` 组合
  T/F 两轴与 time_warp。
- 论文背景：SpecAugment 1904.08779（Park et al.）定义 mT/mF 频谱掩蔽；
  FunASR 版本的差异点是 spec_lengths 未使用、LFR 变体
  (`mask_along_axis_lfr`) 按 lfr_rate 重复段——本题不含 LFR。
- 回源取舍：torchaudio `FrequencyMasking/TimeMasking` 也是常见出处，但
  pytorch/audio 仓在 v2.9 后 Python 侧迁走（contents API 404），且 asr
  合集语境下 FunASR 更对口、纯 numpy 可转写——立 FunASR。

## 算法逐条账

1. `D = spec.shape[dim]`（time=1 → T，freq=2 → F）；单条序列视角。
2. 段布尔：`(mask_pos <= aran) & (aran < mask_pos + mask_length)` 左闭
   右开；width=0 → 空段；越界由 aran 范围天然截断（aran 只到 D-1）。
3. `mask.any(dim=1)` 布尔并——**相邻段（[0,2)+[2,4)）在布尔向量上连成
   一节**，这就是 todos 里"相邻段合并规则"的机制本质：不是特殊逻辑，
   而是 any 的自然结果。`segments` 输出即该布尔向量的连续段分解。
4. `replace_with_zero=False` → `value = spec.mean()`：全谱标量（含将被
   mask 的元素，且官方 mean 在 masked_fill 前对原 spec 计算）。注意
   官方 `masked_fill_` 原地（requires_grad=False 时）——单序列无观察
   差异。
5. 官方随机采样 quirk（入参化，写进 description 背景）：
   - 宽度 `randint(lo, hi)` 上界开区间（0..29）；
   - 起点 `randint(0, max(1, D - mask_length.max()))` 用**全 batch 最大
     宽度**决定上限——批内耦合，单样本视角看就是"小宽度段的起点可能
     超过 D - 自身宽度"（必然越界截断）。这是官方实现一个真实反直觉点。
   - `MaskAlongAxisLFR` 的 lfr 重复段变体不考。

## oracle 设计

`_oracle` 走经典区间算法：段先 clip 到 [0, D]（width<=0 丢弃）→ 排序 →
区间合并 → 展开。与 solve（逐帧布尔 + 扫描）形成双路径对拍。300 组随机
（D 1..12、含负起点/越界段/width=0、replace_with_zero 二选一），另断言
segments 升序、互不重叠、右开、界内。

## 变异记录

13 条：12 抓 + 1 等价。等价变异 `skip-masked-copy`（`row[:]` →
`list(row)`，逐行拷贝语义完全相同）——分析记 notes。可观察的 12 条覆盖：
左/右端点开闭、宽度偏移、轴选择、time 轴改按列填充、均值时序（fill 后
再算）、均值口径（逐行）、区段右开端点、相邻段不并、区段倒序、
忽略 mask 全填。

## 查重

- `audio` 前端合集五题是信号处理（stft/frame 等），无数据增强题——
  全库首个增强账（todos 红线确认）。
- 与 `dt-*` token mask（如 attention mask）不同：那是序列 padding 语义，
  本题是频谱段增强。
- time_warp（FunASR specaug 另一组件，dtw 弹性形变）机制更重，若后续
  立项需单查重。
