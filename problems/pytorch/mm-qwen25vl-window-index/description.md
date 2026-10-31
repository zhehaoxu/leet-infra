# 多模态：Qwen2.5-VL 的视觉窗口重排索引与 cu 账

## 任务

Qwen2.5-VL 的视觉塔在部分层用**窗口注意力**：把空间网格上的 token 重排成「窗口连续」的一维序列，让 varlen attention 只在窗内交互，算完再按逆排列还原。本题考的就是这套**纯索引账**：给定若干图/视频段的 patch 网格尺寸 `grid_thw`，逐行转写 transformers `vision_utils.get_vision_window_index`，产出重排索引、窗口边界累计账与逆排列。

全部输入都是整数网格参数，没有任何浮点前向——判题用精确相等。参数与异常约定见 starter 的双语契约，以下只定义计算规则。

## 规则

1. **两个尺度先分清**：`grid_thw` 的 `h,w` 是 patch 级；除以 `spatial_merge_size` 后得到 LLM token 级网格 `llm_h × llm_w`（每个 LLM token 对应 `merge²` 个 patch token，即 `spatial_merge_unit`）。LLM 侧的窗口边长 `vws = window_size // spatial_merge_size // patch_size`（官方配置 `112//2//14 = 4`），必须 ≥ 1。
2. **padding 的怪点**：`pad = vws - 边长 % vws`——**整除时 pad 恰等于 vws，不是 0**！于是整除的边长也会被垫出一个全空窗行/列，这些空窗不贡献 token，但会在 cu 账上留下重复边界，最后由 `unique_consecutive` 抹平。
3. 每帧把 `arange(t·llm_h·llm_w)` 摆成 `(t, llm_h, llm_w)` 网格，右/下垫 `-100`，按 `(窗行, 窗列)` 行主序切窗；**窗内 token 保持 (窗内行, 窗内列) 行主序**。窗口逐帧、逐窗行、逐窗列展开成一条序列。
4. `window_index` = 各窗依次取出的非 `-100` 网格索引，**加上跨图偏移**（前面所有图的 `t·llm_h·llm_w` 之和）后拼接；它是一个 `0..T-1` 的排列。
5. `cu_window_seqlens` 从 0 开始：每过完一个窗口就累加 `(该窗非 pad 的 llm token 数) × merge²`；注意它账的是 **patch token 级**（varlen attention 在 merge 前的 patch 序列上做），所以整条账比 `window_index` 长得快。最后 `unique_consecutive` 去相邻重复，得 int32。
6. `reverse_indices = argsort(window_index)`——排列的 argsort 就是逆排列，上游用它把窗口输出还原。

两个输出的长度**天然不同**：`window_index`/`reverse_indices` 是 LLM token 级（长度 `Σ t·llm_h·llm_w`），`cu_window_seqlens` 是 patch 级边界，这不是 bug。

## 样例 1：单窗锚点

`grid=[[1,4,4]]`、`merge=2`、`patch=14`、`window=112` → `llm 2×2`、`vws=4`、pad 到 `4×4`、单窗：

```text
window_index      = [0, 1, 2, 3]
cu_window_seqlens = [0, 16]        # 4 个 llm token × merge² = 16 个 patch token
reverse_indices   = [0, 1, 2, 3]
```

## 样例 2：整除产生空窗（核心怪点）

`grid=[[1,16,16]]` → `llm 8×8`。`pad_h = pad_w = 4`（8 整除 4！），切出 **3×3 = 9 窗格，其中第 3 窗行/第 3 窗列共 5 个空窗**：

```text
window_index 前 8 项 = [0, 1, 2, 3, 8, 9, 10, 11]   # 窗(0,0) 内 (r,c) 行主序
len(window_index)    = 64                            # 空窗贡献 0 个 token
cu_window_seqlens    = [0, 64, 128, 192, 256]        # 4 个满窗，重复边界已去重
```

若把 pad 改成「整除时 0」——数学上**输出完全相同**（见 notes 的等价变异记录）；真正可观察的是去掉 `unique_consecutive` 这一步， cu 会残留 `[0,64,128,192,256,256,256,256,256,256]` 的重复尾巴。

## 样例 3：视频帧与不整除

`grid=[[2,10,6]]`、`merge=2` → 每帧 `llm 5×3`：窗行 `r0-3`（12 token）+ `r4`（3 token），帧 1 后接帧 2：

```text
window_index      = [0..29]          # 恰好恒等（窗行全覆盖、列不跨窗）
cu_window_seqlens = [0, 48, 60, 108, 120]   # 12×4 与 3×4 逐帧累计
```

## 判题

8 个 `check_*`：单窗锚点、整除空窗与去重、跨图偏移、视频帧不整除、merge unit 的缩放账、随机网格与独立 oracle 对拍、逆排列性质（`window_index[reverse] == arange`）、非连续视图与非法输入。oracle 是纯 Python 五重循环（帧 → 窗行 → 窗列 → 窗内行 → 窗内列），不使用任何 torch reshape/permute；开发期另用本地 transformers 5.20.2 的 `get_vision_window_index` 做了 36 组随机网格的三方交叉验证（oracle = transformers = solution）。

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-qwen25vl-window-index --mode submit
```
