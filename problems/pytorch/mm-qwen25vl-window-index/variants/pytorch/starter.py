import torch


def solve(grid_thw, spatial_merge_size, patch_size, window_size):
    """Qwen2.5-VL 视觉窗口重排索引与 cu_window_seqlens / Window reorder index
    and cumulative window seqlens for Qwen2.5-VL's windowed vision attention.

    中文契约：
    - grid_thw：CPU int64 的 [G,3]，每行是 (t, h, w)：帧数与该帧在 **patch 级**的
      高/宽。G=1..4，t=1..4，h、w=1..64 且都必须被 spatial_merge_size 整除
      （上游 spatial merge 投影的要求）。
    - spatial_merge_size / patch_size / window_size：Python int（拒绝 bool），
      都必须 ≥ 1。vit_merger_window_size = window_size // spatial_merge_size //
      patch_size（LLM token 网格上的窗口边长；官方配置 112//2//14 = 4），
      结果必须 ≥ 1，否则抛 ValueError。
    - 全部逐行转写 transformers `vision_utils.get_vision_window_index`
      （Qwen2.5-VL 窗口注意力层把 token 重排成窗口连续、用 varlen attention
      分窗计算，再按逆排列还原）：
      1. 每张图/每段视频：llm_grid_h = h // spatial_merge_size、
         llm_grid_w = w // spatial_merge_size；token 网格索引
         index = arange(t * llm_grid_h * llm_grid_w).reshape(t, llm_h, llm_w)。
      2. **pad = vws - 边长 % vws（整除时 pad = vws，不是 0！）**——这是上游
         行为点：整除的边长也会被垫出一个**全空窗行/列**，随后靠
         unique_consecutive 去掉空窗产生的重复 cu 边界。pad 加在网格右/下方。
      3. 垫 -100 后 reshape(t, num_h, vws, num_w, vws).permute(0, 1, 3, 2, 4)
         → 每帧按 (窗行, 窗列) 行主序展开窗口；窗口内的 token 保持
         (窗内行, 窗内列) 行主序。seqlens 是每个窗口内非 -100 的 llm token 数。
      4. window_index = 各窗按序取出的非 -100 网格索引，**加上跨图偏移**
         （前面所有图的 t*llm_h*llm_w 之和）后拼接。
      5. cu_window_seqlens：从 0 开始，每追加一个窗口就把
         (该窗口非 pad 的 llm token 数) × spatial_merge_unit（= merge²，
         一个 llm token 对应 merge² 个 patch token）累加到账上；再对整条
         序列做 **unique_consecutive**（只去相邻重复），得 int32。
      6. reverse_indices = argsort(window_index)（window_index 是 0..T-1 的
         排列，argsort 给出逆排列，上游用它把窗口输出还原回原始顺序）。
    - 注意两个长度不同是**正常且正确**的：window_index / reverse_indices 是
      LLM token 级（长度 Σ t*llm_h*llm_w），cu_window_seqlens 是 patch token
      级（重排后的 varlen attention 在 merge 前的 patch 序列上做）。
    - ValueError：grid_thw 不是 int64 二维张量、G<1 或列数≠3、任一 t/h/w < 1、
      h 或 w 不被 spatial_merge_size 整除、三个标量参数非 int 或 < 1、
      vit_merger_window_size < 1。
    - 接受非连续输入，不修改任何输入。
    - 返回三元组，顺序固定：
      window_index：int64 [T]，LLM token 级重排索引；
      cu_window_seqlens：int32 [W+1]，首项 0、严格递增（空窗已去重）；
      reverse_indices：int64 [T]，逆排列。
    - torch 的 F.pad / reshape / permute / cumsum / unique_consecutive /
      argsort 均可用；禁止 scipy。

    English contract:
    - grid_thw is a CPU int64 tensor [G,3]; each row is (t, h, w): frame count
      and that frame's height/width at the PATCH level. G=1..4, t=1..4,
      h,w=1..64 and both must be divisible by spatial_merge_size (the upstream
      spatial-merge projection requirement).
    - spatial_merge_size / patch_size / window_size are Python ints (bool
      rejected), each >= 1. vit_merger_window_size = window_size //
      spatial_merge_size // patch_size (the window edge on the LLM-token
      grid; the official config gives 112//2//14 = 4) must be >= 1, else
      raise ValueError.
    - Everything below transcribes transformers' `vision_utils.
      get_vision_window_index` (Qwen2.5-VL's windowed attention reorders
      tokens into window-contiguous order, runs varlen attention per window,
      then restores via the inverse permutation):
      1. Per image/video: llm_grid_h = h // merge, llm_grid_w = w // merge;
         index = arange(t * llm_h * llm_w).reshape(t, llm_h, llm_w).
      2. **pad = vws - size % vws (EQUALS vws when divisible, NOT 0!)** — the
         upstream behaviour point: a divisible edge still gains one ALL-EMPTY
         window row/column, whose duplicate cu boundaries are later removed
         by unique_consecutive. Padding goes on the right/bottom.
      3. Pad with -100, reshape(t, num_h, vws, num_w, vws), permute(0,1,3,2,4)
         — windows unfold per frame in (window-row, window-col) row-major
         order; inside a window tokens keep (in-row, in-col) row-major order.
         seqlens counts the non-(-100) llm tokens per window.
      4. window_index = the non-(-100) grid indices taken window by window,
         each PLUS the cross-image offset (sum of previous images'
         t*llm_h*llm_w), concatenated.
      5. cu_window_seqlens: start from 0; after each window add
         (its non-pad llm-token count) × spatial_merge_unit (= merge², one
         llm token corresponds to merge² patch tokens); finally
         **unique_consecutive** over the whole sequence (adjacent duplicates
         only), result int32.
      6. reverse_indices = argsort(window_index) — window_index is a
         permutation of 0..T-1, so argsort is its inverse, which upstream
         uses to restore the original order.
    - The two length scales are BOTH correct: window_index / reverse_indices
      are LLM-token level (length Σ t*llm_h*llm_w) while cu_window_seqlens is
      patch-token level (the varlen attention runs on the pre-merge patch
      sequence).
    - Raise ValueError when grid_thw is not a 2-D int64 tensor, G < 1 or its
      column count is not 3, any t/h/w < 1, h or w not divisible by merge,
      any scalar argument is not an int or < 1, or vit_merger_window_size < 1.
    - Accept noncontiguous inputs and never mutate any input.
    - Return a triple in this fixed order: window_index (int64 [T]),
      cu_window_seqlens (int32 [W+1], first entry 0, strictly increasing,
      empty windows deduplicated), reverse_indices (int64 [T]).
    - F.pad / reshape / permute / cumsum / unique_consecutive / argsort are
      all allowed; no scipy.
    """
    raise NotImplementedError
