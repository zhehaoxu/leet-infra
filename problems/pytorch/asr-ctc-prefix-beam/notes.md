# notes — asr-ctc-prefix-beam（order 670）

## 回源

- 出处：Graves et al. 1504.08129「Towards End-to-End Speech Recognition with Recurrent Neural Networks」的 prefix beam search DP（distill.pub/ctc 标准转写；wav2vec2/icefall 系通用）。本题取**无 LM、线性概率域**最小口径。
- 三个语义关键点：
  - **重复字符的非对称分支**：s 末尾恰为 c 时，`s+c` 只能从 p_b 扩展（blank 分隔两个同字符），同时 p_nb 延续在 s 上——这是 prefix beam search 区别于 path beam search 的全部实质；
  - **blank 分支用 (p_b + p_nb) 和**：blank 后两种结尾状态都汇入同一前缀；
  - **剪枝与 tie-break**：按总概率降序、并列按前缀字典序升序——无 tie-break 时并列顺序依赖 dict 插入序（不可复现）。
- 等价变异：初始 `(1.0, 0.0)` → `(0.0, 1.0)` 不可观察（首帧只消费 p_b+p_nb 之和，空 prefix 的划分随即被首帧重写）。已换 blank 二次扩展变异。
- 与 `ctc-loss`（loss 侧已占）、`ocr-ctc-greedy-decode`（greedy 路径账）划界：本题是**前缀级 beam DP**。

## 判题面

- solve 入参：probs [T,V]（线性概率）、beam_width、blank_id、k（None 取 beam_width）。返回 hypotheses（prefix list + prob）。
- oracle：**穷举 V^T 全路径按 collapse 求和**（与 beam DP 完全独立的语义路径），随机 120 组（T≤4、V≤3、softmax 归一 + 尖峰、bw=999 不剪枝）容差 1e-9 对拍；另用手算锚点钉剪枝行为（beam_width=1 时 (1,) 自吸收 blank 概率 0.32 vs 不剪枝 0.38）。
- 锚点：单帧三前缀排序；A blank A 合并到同前缀（T=3 手算 0.262/0.729/0.009）；AA 重复 P(≥1 个 A)=0.75；并列字典序；k 截断；全 blank 输出空；零概率假设过滤。
- 非法输入：probs 空/行不齐/负/NaN/bool、blank_id 越界/bool、beam_width/k 非正 int 或 bool → ValueError。

## 变异记录（9 条全抓 + 1 条等价）

repeat-from-both（重复分支混入 p_nb）、no-repeat-extend（延续丢失）、blank-from-pnb-only（blank 分支丢 p_nb）、no-prune、sort-ascending、k-plus-one、prob-no-blank（输出概率丢 p_b）、zero-kept（零概率假设保留）、blank-also-nonblank（blank 类再走非 blank 分支）。

**等价变异**：初始 (1.0,0.0) → (0.0,1.0)，已记 notes。

## 红线

LM 打分不出（无外部分数入参；如续做带 LM 版本需新立项查重）；真模型/音频不出；与 `ocr` 合集的 greedy 账划界：本题是 **beam DP**。
