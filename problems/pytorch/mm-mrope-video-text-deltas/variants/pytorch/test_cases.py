"""独立 oracle + 锚点用例：mm-mrope-video-text-deltas (order 685).

oracle 与 solution 完全独立实现：官方分段逻辑改写为「先解析成
text/vis 段列表、再逐段生成位置」的两遍结构，避免与 solution 的
单循环共享任何中间变量。token 协议：视觉段必须以 vision_start(102)
开头，紧跟 image_token(100) 或 video_token(200)。
"""

import math


def _oracle(tokens, attention_mask, image_grid_thw, video_grid_thw, second_per_grid_ts,
            spatial_merge_size, tokens_per_second, image_token_id, video_token_id,
            vision_start_token_id):
    L = len(tokens)
    eff = [t for t, m in zip(tokens, attention_mask) if m == 1]
    n = len(eff)
    images = list(image_grid_thw or [])
    videos = list(video_grid_thw or [])
    spg = second_per_grid_ts

    if not images and not videos:
        axis = []
        run = 0
        for m in attention_mask:
            axis.append(run if m == 1 else 1)
            run += 1 if m == 1 else 0
        return {"position_ids": [list(axis), list(axis), list(axis)],
                "mrope_position_delta": max(axis) + 1 - L}

    ni = nv = 0
    for i, v in enumerate(eff):
        if v == vision_start_token_id and i + 1 < n:
            if eff[i + 1] == image_token_id:
                ni += 1
            elif eff[i + 1] == video_token_id:
                nv += 1

    segs = []
    st = 0
    ri, rv = ni, nv
    ii = vi = 0
    for _ in range(ni + nv):
        ei = eff.index(image_token_id, st) if (ri > 0 and image_token_id in eff[st:]) else n + 1
        ev = eff.index(video_token_id, st) if (rv > 0 and video_token_id in eff[st:]) else n + 1
        if ei < ev:
            grid, sec, ed = images[ii], 0, ei
            ii += 1; ri -= 1
        else:
            grid, sec, ed = videos[vi], (1.0 if spg is None else float(spg[vi])), ev
            vi += 1; rv -= 1
        t, h, w = grid
        gh, gw = h // spatial_merge_size, w // spatial_merge_size
        if ed > st:
            segs.append(("text", ed - st))
        segs.append(("vis", t, gh, gw, sec))
        st = ed + t * gh * gw
    if st < n:
        segs.append(("text", n - st))

    axes = [[], [], []]
    for seg in segs:
        if seg[0] == "text":
            m = max((max(a) for a in axes if a), default=-1) + 1
            for ax in range(3):
                axes[ax].extend(range(m, m + seg[1]))
        else:
            _, t, gh, gw, sec = seg
            m = max((max(a) for a in axes if a), default=-1) + 1
            off = m
            for ti in range(t):
                tv = math.floor(ti * sec * tokens_per_second)
                for hh in range(gh):
                    for ww in range(gw):
                        axes[0].append(tv + off)
                        axes[1].append(hh + off)
                        axes[2].append(ww + off)
    gmax = max(max(a) for a in axes)
    idxs = [i for i in range(L) if attention_mask[i] == 1]
    pos = [[1] * L for _ in range(3)]
    for ax in range(3):
        for k, i in enumerate(idxs):
            pos[ax][i] = axes[ax][k]
    return {"position_ids": pos, "mrope_position_delta": gmax + 1 - L}


def _compare(solve, tokens, mask, images, videos, spg, merge, tps, iid, vid, sid):
    exp = _oracle(tokens, mask, images, videos, spg, merge, tps, iid, vid, sid)
    got = solve(tokens, mask, images, videos, spg, merge, tps, iid, vid, sid)
    assert exp == got, f"\nexp={exp}\ngot={got}"
    return got


def check_pure_text_full_mask(solve):
    # 无 grid → 官方 fallback：三轴 cumsum-1；delta = (L-1)+1-L = 0
    led = _compare(solve, [7, 8, 9, 10], [1, 1, 1, 1], None, None, None, 2, 25, 100, 200, 102)
    assert led["position_ids"] == [[0, 1, 2, 3]] * 3
    assert led["mrope_position_delta"] == 0


def check_pure_text_padding_mask(solve):
    # fallback + 尾部 padding：有效位置 cumsum-1，padding 填 1（不是 0）
    led = _compare(solve, [7, 8, 9, 10, 11], [1, 1, 1, 0, 0], None, None, None, 2, 25, 100, 200, 102)
    assert led["position_ids"] == [[0, 1, 2, 1, 1]] * 3
    # 官方 quirk：delta 用含 padding 的全长 L → 2+1-5 = -2
    assert led["mrope_position_delta"] == -2


def check_hole_mask_fallback(solve):
    # mask 中间有洞（官方对任意 mask 有效）：洞位填 1
    led = _compare(solve, [7, 8, 9, 10], [1, 0, 1, 1], None, None, None, 2, 25, 100, 200, 102)
    assert led["position_ids"] == [[0, 1, 1, 2]] * 3
    assert led["mrope_position_delta"] == 2 + 1 - 4


def check_single_image(solve):
    # T start [V×4] T T：t=1,h=2,w=2,merge=1 → 4 token
    # 文本前 [0,1]（start 也算文本）；视觉 off=2；尾文本 st_idx = max(2,3)+1 = 4
    toks = [7, 102, 100, 100, 100, 100, 8, 9]
    led = _compare(solve, toks, [1] * 8, [[1, 2, 2]], None, None, 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 2, 2, 2, 2, 4, 5]
    assert led["position_ids"][1] == [0, 1, 2, 2, 3, 3, 4, 5]
    assert led["position_ids"][2] == [0, 1, 2, 3, 2, 3, 4, 5]
    assert led["mrope_position_delta"] == 5 + 1 - 8


def check_image_temporal_axis_constant(solve):
    # 图像 second_per_grid_t 恒 0：即使 grid_t=2，t 轴也是常数（官方账）
    toks = [102] + [100] * 8
    led = _compare(solve, toks, [1] * 9, [[2, 2, 2]], None, None, 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0] + [1] * 8
    assert led["position_ids"][1] == [0, 1, 1, 2, 2, 1, 1, 2, 2]
    assert led["position_ids"][2] == [0, 1, 2, 1, 2, 1, 2, 1, 2]
    assert led["mrope_position_delta"] == 2 + 1 - 9


def check_video_time_steps_floor(solve):
    # 视频 t=3, second_per_grid_t=0.5, tps=25 → 步 12.5 → floor: 0,12,25（再 +off 1）
    toks = [102, 200, 200, 200, 9]
    led = _compare(solve, toks, [1] * 5, None, [[3, 1, 1]], [0.5], 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 13, 26, 27]
    # gh=gw=1：h/w 轴三个视频 token 全是 off=1，尾文本 = max(26)+1 = 27
    assert led["position_ids"][1] == [0, 1, 1, 1, 27]
    assert led["position_ids"][2] == [0, 1, 1, 1, 27]
    assert led["mrope_position_delta"] == 27 + 1 - 5


def check_video_default_second(solve):
    # second_per_grid_ts=None → 默认 1.0：步长 25
    toks = [102, 200, 200, 200, 9]
    led = _compare(solve, toks, [1] * 5, None, [[3, 1, 1]], None, 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 26, 51, 52]


def check_interleave_img_vid(solve):
    # T start img(4) T start vid(1) T T T：分段偏移链 st_idx = 前段全局 max + 1
    toks = [7, 102, 100, 100, 100, 100, 8, 102, 200, 9, 10, 11]
    led = _compare(solve, toks, [1] * 12, [[1, 2, 2]], [[1, 1, 1]], [2.0], 1, 25, 100, 200, 102)
    t_ax = led["position_ids"][0]
    assert t_ax[:2] == [0, 1]              # 前文本（含 start）
    assert t_ax[2:6] == [2, 2, 2, 2]       # 图像 t 轴恒 off=2
    assert t_ax[6:8] == [4, 5]             # 中间文本两 token（8 与 start），st_idx=img max(3)+1
    assert t_ax[8] == 6                    # 视频 t 轴 = floor(0*2*25)+off 6
    assert t_ax[9:] == [7, 8, 9]           # 尾文本 = 视频全局 max(6)+1 起
    assert led["position_ids"][1][8] == 6 and led["position_ids"][2][8] == 6
    assert led["mrope_position_delta"] == 9 + 1 - 12


def check_merge_division(solve):
    # merge=2：grid (1,4,6) → llm_grid (1,2,3)，6 个 token
    toks = [102] + [100] * 6 + [9]
    led = _compare(solve, toks, [1] * 8, [[1, 4, 6]], None, None, 2, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 1, 1, 1, 1, 1, 4]
    assert led["position_ids"][1] == [0, 1, 1, 1, 2, 2, 2, 4]
    assert led["position_ids"][2] == [0, 1, 2, 3, 1, 2, 3, 4]
    assert led["mrope_position_delta"] == 4 + 1 - 8


def check_bare_image_token_quirk(solve):
    # 官方 quirk：裸 image token（前无 vision_start）不计数，但 index 定位会找到它
    # eff = [img_bare, start, img, 8]；image_nums=1；ed=0 → grid 挂在 bare 位置，
    # [start, img, 8] 全被当尾文本
    toks = [100, 102, 100, 8]
    led = _compare(solve, toks, [1] * 4, [[1, 1, 1]], None, None, 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 2, 3]
    assert led["position_ids"][1] == [0, 1, 2, 3]
    assert led["position_ids"][2] == [0, 1, 2, 3]


def check_two_images_offset_chain(solve):
    # 双图交错：T start img(2) T start img(2) T；验证 st 推进与文本偏移
    toks = [7, 102, 100, 100, 8, 102, 100, 100, 9]
    led = _compare(solve, toks, [1] * 9, [[1, 1, 2], [1, 1, 2]], None, None, 1, 25, 100, 200, 102)
    # img2 的 ed 从 st=4 起找首个 100 → 6（start 在 5）→ 中间文本是 idx4,5 两个
    assert led["position_ids"][0] == [0, 1, 2, 2, 4, 5, 6, 6, 8]
    assert led["position_ids"][1] == [0, 1, 2, 2, 4, 5, 6, 6, 8]
    assert led["position_ids"][2] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert led["mrope_position_delta"] == 8 + 1 - 9


def check_video_floor_half_odd(solve):
    # t=4, s=0.5, tps=25 → 步 0,12.5,25,37.5 → int 截断 0,12,25,37
    # （Python round(37.5)=38 的 half-to-even 上取整会得到 39——钉 floor 口径）
    toks = [102] + [200] * 4 + [9]
    led = _compare(solve, toks, [1] * 6, None, [[4, 1, 1]], [0.5], 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 13, 26, 38, 39]
    assert led["position_ids"][1] == [0, 1, 1, 1, 1, 39]


def check_main_path_padding(solve):
    # 主路径 + 尾部 padding：position_ids padding 位填 1（官方 ones 初始化）；
    # delta 用含 padding 的全长 L（官方 total_input_ids quirk）
    toks = [7, 102, 100, 100, 100, 100, 8, 9, 0, 0]
    mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    led = _compare(solve, toks, mask, [[1, 2, 2]], None, None, 1, 25, 100, 200, 102)
    assert led["position_ids"][0] == [0, 1, 2, 2, 2, 2, 4, 5, 1, 1]
    assert led["mrope_position_delta"] == 5 + 1 - 10


def check_input_not_mutated(solve):
    toks = [7, 102, 100, 100, 8]
    mask = [1, 1, 1, 1, 1]
    imgs = [[1, 1, 2]]
    toks_copy, mask_copy = list(toks), list(mask)
    _compare(solve, toks, mask, imgs, None, None, 1, 25, 100, 200, 102)
    assert toks == toks_copy and mask == mask_copy and imgs == [[1, 1, 2]]


def check_invalid_inputs(solve):
    def run(tokens, mask, images=None, videos=None, spg=None, merge=2, tps=25,
            iid=100, vid=200, sid=102):
        solve(tokens, mask, images, videos, spg, merge, tps, iid, vid, sid)

    def expect_err(*a, **kw):
        try:
            run(*a, **kw)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    expect_err([], [1])                                   # 空 tokens
    expect_err([7, 8, True], [1, 1, 1])                   # bool token
    expect_err([7, 8], [1])                               # mask 长度
    expect_err([7, 8], [1, 2])                            # mask 非 0/1
    expect_err([7, 8], [0, 0])                            # 全 0
    expect_err([7, 8], [1, 1], merge=0)                   # merge<1
    expect_err([7, 8], [1, 1], tps=0)                     # tps<1
    expect_err([7, 8], [1, 1], iid=102, vid=100, sid=102)  # id 相等
    expect_err([7, 102, 100, 8], [1] * 4, images=[[0, 1, 1]])      # t<=0
    expect_err([7, 102, 100, 8], [1] * 4, images=[[1, 3, 1]], merge=2)  # 不整除
    expect_err([7, 102, 100, 8], [1] * 4, images=[[1, 2, 1]], merge=2)  # w 商 0
    expect_err([7, 102, 200, 8], [1] * 4, videos=[[1, 2, 2]], spg=[1.0, 2.0])  # spg 长度
    expect_err([7, 102, 200, 8], [1] * 4, videos=[[1, 2, 2]], spg=[-0.5])      # 负 second
    expect_err([7, 102, 8], [1, 1, 1], images=[[1, 2, 2]])   # 计数不符
    expect_err([7, 100, 102], [1, 1, 1], images=[[1, 2, 2]])  # start 是 eff 最后位
