# 解析：Qwen2.5-VL 的视觉窗口重排索引与 cu 账

## 实现思路

solution 是 transformers 5.20.2 `vision_utils.get_vision_window_index` 的逐行转写（`modeling_qwen2_5_vl.py` 正是从 `vision_utils` 导入这个函数），只在两处显式化：入参校验（`vws ≥ 1`、`h,w` 被 merge 整除、标量参数拒绝 bool）和末尾补上模型侧紧随其后的 `reverse_indices = torch.argsort(window_index)`。

实现主干照抄上游：`arange → reshape(t, llm_h, llm_w) → F.pad(-100) → reshape(t, nwh, vws, nww, vws) → permute(0,1,3,2,4) → reshape(t, nwh*nww, vws, vws)`；`seqlens = (index_padded != -100).sum([2,3]).reshape(-1)` 是每窗的 llm token 数；摊平后布尔选非 `-100`、加跨图偏移；cu 用 `seqlens.cumsum(0) * mu + 前图累计`，最后 `torch.unique_consecutive`。两个易错位：**F.pad 的 `(0, pad_w, 0, pad_h)` 顺序**（最后一维是 w 左右、倒数第二维是 h 上下）；**permute 把「窗行、窗列」挪到「窗内行、窗内列」之前**，与 Swin `window_partition` 同构（`mm-window-partition` 考的是不带 padding 与 -100 标记的静态切块，本题是带跨图偏移与两级尺度账的完整 varlen 前置账，分工已写明）。

## 语义锚点（用本地 transformers 实证）

开发期直接调用 `.venv-eval` 里 transformers 5.20.2 的 `get_vision_window_index` 实证了三个关键语义，避免凭 2402.10807 论文想当然：① 16×16 patch（8×8 llm）在 `vws=4` 下确实切出 3×3 窗格、5 个空窗、cu 去重后只剩 4 满窗的 5 个边界；② cu 步长是「每窗 llm token 数 × merge²」（patch 级），与 `window_index` 的 llm 级长度不同轴；③ `t>1` 时窗口按 `(ti, 窗行, 窗列)` 展开，帧间不重开窗格。

## 等价变异的发现

变异测试 7 个中 5 个有效变异全抓（漏 permute、cu 忘乘 merge²、跨图偏移用 patch 级、去掉 unique_consecutive、reverse 用 window_index 顶替）；**两个存活的是数学等价变异**：`pad` 改成「整除时 0」（clamp 版）与 `num_windows` 的恒等重写。原因：空窗对 `window_index` 贡献 0 项，对 cu 只追加重复边界，而重复恰是相邻的（序列非降），`unique_consecutive` 无差别抹平——所以 pad 公式在**最终输出上**与 clamp 版本恒等，上游那个「整除 pad=vws」的怪点是内部过程而非可观察行为。真正可观察的是**去掉去重这一步**（no-dedup 变异被 5 个用例抓住）。这与本仓库既往的等价变异记录（no-validation-r、CPU 并列稳定等）同类：转写题的某些上游「怪」在输出空间里是自恰的，notes 必须把它降级为「过程怪点」而不是「行为契约」。

## 独立验证

oracle 是纯 Python 五重循环：帧 → 窗行 → 窗列 → 窗内行 → 窗内列，窗口序与窗内序都按行主序手写，空位（逻辑坐标越界）跳过并只对非空位计数 ×`merge²` 累加 cu，最后手写「去相邻重复」与逆排列（`reverse[tok]=pos`）。不使用任何 torch 的 reshape/permute/cumsum/argsort。随机用例 25 组覆盖 merge∈{1,2,4}、patch∈{1,2,7,14}、window∈{28,56,112,224}、G∈1..4、t∈1..3；另有三方交叉验证（oracle = transformers = solution）36 组全零差异。整数账全部 `==` 精确断言，无浮点容差。

## 复杂度

设每图 LLM token 数 `T_i = t·llm_h·llm_w`、窗口数 `W_i`。摊平与布尔选择 `O(Σ(T_i + 垫后格点数))`，cu 累计 `O(ΣW_i)`，argsort `O(T log T)`。中间量只有垫后网格与窗口 reshaped 视图（`O(Σ(T_i + pad 格点))`）；整除尺寸下垫后格点最多放大 `(1+1/vws)²` 倍。

## 来源与范围

依据 `.venv-eval` 内 transformers 5.20.2 的 `transformers/vision_utils.py::get_vision_window_index`（`modeling_qwen2_5_vl.py:52` 导入、L442 调用），以及 `modeling_qwen2_5_vl.py` 的 `permute_input_for_window_attn`（`input_tensor[window_index, :, :]` 重排，L419-421）与 `reverse_indices = torch.argsort(window_index)`（L479）。逐行对齐点：`vit_merger_window_size = window_size // spatial_merge_size // patch_size`、`spatial_merge_unit = spatial_merge_size**2`、`pad = vws - 边长 % vws`、`F.pad(..., "constant", -100)`、`seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]`、`torch.unique_consecutive`。

刻意不在范围内：窗口 attention 本身（cu_seqlens 喂给 flash-attn varlen 的内核路径）；`kwargs` 里预计算 `window_index`/`cu_window_seqlens` 直接弹出的缓存分支（纯接线）；与 `mm-window-partition`（Swin 静态切块）、`mm-mrope-position-ids`（Qwen2-VL 位置编号）、`mm-flux-2d-rope`（2D RoPE）的分工均已在各自 notes 写明。

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-qwen25vl-window-index --mode submit
```

## 变异测试（5 有效全抓 + 2 等价）

漏 permute（5/8 用例抓）、cu 忘乘 merge²（7/8）、跨图偏移 patch 级（3/8）、去 unique_consecutive（5/8）、reverse 顶替（4/8）全部抓获；pad-clamp 与 num_windows 重写为输出等价变异，见上文。
