def solve(tokens, attention_mask, image_grid_thw, video_grid_thw, second_per_grid_ts,
          spatial_merge_size, tokens_per_second, image_token_id, video_token_id,
          vision_start_token_id):
    """Qwen2.5-VL 完整 M-RoPE 索引账（见 starter 契约）。"""
    # ---------- 校验 ----------
    if not isinstance(tokens, list) or not (1 <= len(tokens) <= 32):
        raise ValueError("tokens must be a list[int] of length 1..32")
    for v in tokens:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("tokens entries must be ints, not bool")
    if not isinstance(attention_mask, list) or len(attention_mask) != len(tokens):
        raise ValueError("attention_mask must match tokens length")
    for m in attention_mask:
        if isinstance(m, bool) or not isinstance(m, int) or m not in (0, 1):
            raise ValueError("attention_mask entries must be 0/1 ints")
    if sum(attention_mask) < 1:
        raise ValueError("attention_mask must have at least one 1")
    for name, tid in (("image_token_id", image_token_id), ("video_token_id", video_token_id),
                      ("vision_start_token_id", vision_start_token_id)):
        if isinstance(tid, bool) or not isinstance(tid, int):
            raise ValueError(f"{name} must be an int")
    if len({image_token_id, video_token_id, vision_start_token_id}) != 3:
        raise ValueError("the three token ids must be pairwise distinct")
    if isinstance(spatial_merge_size, bool) or not isinstance(spatial_merge_size, int) or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be an int >= 1")
    if isinstance(tokens_per_second, bool) or not isinstance(tokens_per_second, int) or tokens_per_second < 1:
        raise ValueError("tokens_per_second must be an int >= 1")

    def _check_grid(g):
        if g is None:
            return []
        if not isinstance(g, list):
            raise ValueError("grid must be None or a list of [t,h,w]")
        out = []
        for row in g:
            if (not isinstance(row, list) or len(row) != 3):
                raise ValueError("grid rows must be [t,h,w]")
            for x in row:
                if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                    raise ValueError("grid entries must be positive ints")
            t, h, w = row
            if h % spatial_merge_size or w % spatial_merge_size:
                raise ValueError("grid h/w must be divisible by spatial_merge_size")
            if h // spatial_merge_size < 1 or w // spatial_merge_size < 1:
                raise ValueError("grid h/w must exceed spatial_merge_size")
            out.append((t, h, w))
        return out

    images = _check_grid(image_grid_thw)
    videos = _check_grid(video_grid_thw)
    if second_per_grid_ts is not None:
        if not isinstance(second_per_grid_ts, list):
            raise ValueError("second_per_grid_ts must be None or list[float]")
        if len(second_per_grid_ts) != len(videos):
            raise ValueError("second_per_grid_ts length must equal the number of videos")
        for s in second_per_grid_ts:
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not (s >= 0):
                raise ValueError("second_per_grid_ts entries must be floats >= 0")

    eff = [tokens[i] for i in range(len(tokens)) if attention_mask[i] == 1]
    L = len(tokens)

    # ---------- 纯文本 fallback ----------
    if not images and not videos:
        pos = []
        run = 0
        for m in attention_mask:
            if m == 1:
                pos.append(run)
                run += 1
            else:
                pos.append(1)  # 官方 masked_fill_(mask==0, 1)
        axis = pos[:]
        max_pos = max(pos)
        delta = max_pos + 1 - L
        return {"position_ids": [axis[:], axis[:], axis[:]], "mrope_position_delta": delta}

    # ---------- vision 计数（只认 vision_start 后紧跟的 token） ----------
    image_nums = 0
    video_nums = 0
    for i, v in enumerate(eff):
        if v == vision_start_token_id:
            if i + 1 >= len(eff):
                raise ValueError("vision_start token cannot be the last effective token")
            nxt = eff[i + 1]
            if nxt == image_token_id:
                image_nums += 1
            elif nxt == video_token_id:
                video_nums += 1
    if image_nums != len(images):
        raise ValueError("image segment count does not match image_grid_thw")
    if video_nums != len(videos):
        raise ValueError("video segment count does not match video_grid_thw")

    # ---------- 分段扫描 ----------
    segments = []  # 每段是 (3, n) 的三个 list
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index, video_index = 0, 0
    n_eff = len(eff)

    for _ in range(image_nums + video_nums):
        if remain_images > 0 and image_token_id in eff[st:]:
            ed_image = eff.index(image_token_id, st)
        else:
            ed_image = n_eff + 1
        if remain_videos > 0 and video_token_id in eff[st:]:
            ed_video = eff.index(video_token_id, st)
        else:
            ed_video = n_eff + 1
        if ed_image < ed_video:
            t, h, w = images[image_index]
            second_per_grid_t = 0
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = videos[video_index]
            second_per_grid_t = 1.0 if second_per_grid_ts is None else float(second_per_grid_ts[video_index])
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        llm_grid_t = t
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        text_len = ed - st

        prev = [v for seg in segments for axis in seg for v in axis]
        st_idx = (max(prev) + 1) if prev else 0

        segments.append([[st_idx + i for i in range(text_len)] for _ in range(3)])

        t_axis = [int(i * second_per_grid_t * tokens_per_second) for i in range(llm_grid_t)]
        flat_t = [v for v in t_axis for _ in range(llm_grid_h * llm_grid_w)]
        flat_h = [hh for _ in range(llm_grid_t) for hh in range(llm_grid_h) for _ in range(llm_grid_w)]
        flat_w = [ww for _ in range(llm_grid_t) for _ in range(llm_grid_h) for ww in range(llm_grid_w)]
        off = text_len + st_idx
        segments.append([ [v + off for v in flat_t],
                          [v + off for v in flat_h],
                          [v + off for v in flat_w] ])
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < n_eff:
        prev = [v for seg in segments for axis in seg for v in axis]
        st_idx = (max(prev) + 1) if prev else 0
        text_len = n_eff - st
        segments.append([[st_idx + i for i in range(text_len)] for _ in range(3)])

    # 重组：segments 顺序即 eff 顺序，每段是 (3, n)
    flat = [[], [], []]
    for seg in segments:
        for ax in range(3):
            flat[ax].extend(seg[ax])
    all_vals = flat[0] + flat[1] + flat[2]
    gmax = max(all_vals)
    delta = gmax + 1 - L

    position_ids = [[1] * L for _ in range(3)]
    idxs = [i for i in range(L) if attention_mask[i] == 1]
    for ax in range(3):
        for k, i in enumerate(idxs):
            position_ids[ax][i] = flat[ax][k]
    return {"position_ids": position_ids, "mrope_position_delta": delta}
