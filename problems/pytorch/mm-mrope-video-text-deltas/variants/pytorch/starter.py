def solve(tokens, attention_mask, image_grid_thw, video_grid_thw, second_per_grid_ts,
          spatial_merge_size, tokens_per_second, image_token_id, video_token_id,
          vision_start_token_id):
    """Qwen2.5-VL 完整 M-RoPE 索引账（视频时间轴与多模态交错）/
    Full Qwen2.5-VL M-RoPE index accounting (video temporal axis and interleaving).

    中文契约：
    - tokens：list[int]，长度 1..32，单条序列（batch 维省略）。元素是 token id
      （int，拒绝 bool）；image_token_id / video_token_id / vision_start_token_id
      三者两两不等，且可以不出现在 tokens 里。
    - attention_mask：list[int]，长度与 tokens 相同，元素只能 0 或 1，至少一个 1。
      官方语义：0 是 padding。有效序列 eff = [tokens[i] for i where mask==1]，
      后续全部账目都在 eff 上做；mask 不要求 0 全在尾部。
    - image_grid_thw / video_grid_thw：None 或 list[[t,h,w]]，t/h/w 为正 int
      （拒绝 bool）。h、w 必须能被 spatial_merge_size 整除且商 >= 1
      （官方用 // 地板除不校验，本题显式校验，见 notes）。
    - second_per_grid_ts：None 或 list[float]（长度必须等于视频段数），
      每个元素 >= 0.0。为 None 时官方默认 1.0。
    - spatial_merge_size：正 int（拒绝 bool）；tokens_per_second：正 int（拒绝 bool）。
    - 计数账：vision_start_indices 是 eff 中等于 vision_start_token_id 的下标；
      image_nums = 满足 eff[i+1]==image_token_id 的 vision_start 下标 i 的个数
      （i+1 越界即 vision_start 是 eff 最后一位 → ValueError，官方此处 IndexError，
      本题显式化）。video_nums 同理。image_nums 必须等于 image_grid_thw 的段数、
      video_nums 必须等于 video_grid_thw 的段数，否则 ValueError
      （官方 IndexError，本题显式化）。裸的 image/video token（前面不是
      vision_start）不计入段数，但扫描定位 `index` 会找到它（官方 quirk，
      见 notes 与用例）。
    - 分段账（主路径，grid 两项都为 None/空时走 fallback，见下）：
      对每个视觉段（按 eff 中 image_token_id / video_token_id 从 st 起首次出现
      位置较小者先处理；remain 计数控制类型优先级，与官方一致）：
      * ed = 该段首个视觉 token 在 eff 中的下标；t,h,w 取对应 grid；
        second_per_grid_t：图像恒 0，视频取 second_per_grid_ts[k]（None 时 1.0）。
      * llm_grid = (t, h//spatial_merge_size, w//spatial_merge_size)。
      * 文本段（st 到 ed-1）：三轴同为 arange(text_len) + st_idx，
        st_idx = 此前所有已生成段的三轴全局最大值 + 1（无前段时 0）。
      * 视觉段：t 轴 = floor(i * second_per_grid_t * tokens_per_second)
        （i=0..llm_grid_t-1，先乘 second_per_grid_t 再乘 tokens_per_second，
        float 乘法后 int 截断，复刻官方 time_tensor.long()）；
        h 轴 / w 轴按 (t, h, w) C 序 flatten；
        三轴都加 (text_len + st_idx)。
      * st 更新为 ed + llm_grid_t*llm_grid_h*llm_grid_w。
      循环结束后若 st < len(eff)，剩余部分作为最后一段文本（st_idx 同上）。
    - position_ids：[3][L] 的 int 嵌套列表（t/h/w 三行）。mask==1 的位置按
      eff 顺序填入分段账；mask==0 的位置填 1（官方 ones 初始化后从不覆盖，
      是 1 不是 0）。两个 grid 都为 None/空时走官方纯文本 fallback：
      三轴同为 (cumsum(mask)-1)，mask==0 处同样填 1。
    - mrope_position_delta：单个 int。主路径 = 分段账三轴全局最大值 + 1 - L
      （L 是含 padding 的全长，官方 quirk：用 total_input_ids 的长度）；
      fallback = 三轴有效位置最大值 + 1 - L。
    - ValueError：tokens 非法（空/超长/含 bool 或非 int）、mask 长度不符/元素非
      0 或 1/全 0、grid 结构非法（非 None 却不是 list[[t,h,w]]、t/h/w 非 int、
      <=0、h/w 不被 merge 整除、商为 0）、second_per_grid_ts 长度不等于视频段数
      或含负数或非数值、spatial_merge_size/tokens_per_second 非法（bool 冒充、
      <1）、三个 token id 相等、vision_start 是 eff 最后一位、段数与 grid 数不符。
    - 输入不被修改。返回 dict：{"position_ids": [[...],[...],[...]],
      "mrope_position_delta": int}。

    English contract mirrors the Chinese one: tokens is a list[int] (len 1..32,
    bool rejected); attention_mask is a 0/1 list of the same length with at least
    one 1; grids are None or list[[t,h,w]] of positive ints whose h/w are
    divisible by spatial_merge_size with quotient >= 1; second_per_grid_ts is
    None or a list of non-negative floats with length equal to the number of
    video segments (None means official default 1.0). Vision-start counting only
    counts tokens directly following vision_start; bare vision tokens are not
    counted but are found by the scan (official quirk). Segments: text segments
    get arange + (global max so far + 1) on all three axes; vision segments get
    t-axis = floor(i * second_per_grid_t * tokens_per_second) (images always
    second_per_grid_t = 0), C-order h/w axes, all offset by text_len + st_idx;
    st advances to ed + llm_grid_t*llm_grid_h*llm_grid_w. position_ids is a
    [3][L] int nested list; masked-out positions carry the value 1 (official
    ones-initialised buffer is never overwritten). mrope_position_delta = global
    max + 1 - L where L is the FULL padded length (official quirk). When both
    grids are empty the official pure-text fallback applies: axes are
    cumsum(mask)-1 with masked-out positions filled by 1, delta = max + 1 - L.
    Raise ValueError for every malformed input listed above. Inputs are not
    modified. Returns {"position_ids": [[...],[...],[...]], "mrope_position_delta": int}.

    禁止 torch；纯 Python 实现。
    No torch; pure Python only.
    """
    raise NotImplementedError
