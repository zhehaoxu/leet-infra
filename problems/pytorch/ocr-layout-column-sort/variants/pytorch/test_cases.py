"""order 691: PP-Structure recovery `sorted_layout_boxes` 双栏阅读顺序账。

oracle 用独立结构（显式扫描队列 + 桶状态机重写）实现同一官方规则，
与 solve 对拍；再加手算锚点钉死 quirk。
"""
import random

W = 1000


def _oracle(bboxes, w):
    n = len(bboxes)
    if n == 1:
        return {"order": [0], "layouts": ["single"]}

    # 独立实现稳定排序：插入手写比较，键相同保留原始先后
    scan = []
    for idx in range(n):
        key = (bboxes[idx][1], bboxes[idx][0])
        pos = len(scan)
        for j, jdx in enumerate(scan):
            if (bboxes[jdx][1], bboxes[jdx][0]) > key:
                pos = j
                break
        scan.insert(pos, idx)

    # 状态机：pending 先攒列内框，遇到切分事件一次性提交
    out = []
    pending_left = []
    pending_right = []
    queue = list(scan)
    pos = 0
    while queue:
        is_last = pos == n - 1
        idx = queue.pop(0)
        x0, y0, x1, y1 = bboxes[idx]
        if is_last:
            px0, py0, px1, py1 = bboxes[scan[pos - 1]]
            if y0 > py1 and x0 < w / 2 and x1 > w / 2:
                out += [(k, "double") for k in pending_left]
                out += [(k, "double") for k in pending_right]
                out.append((idx, "single"))
            else:
                dropped = True
                if x1 > w / 2:
                    pending_right.append(idx)
                    dropped = False
                elif x0 < w / 2:
                    pending_left.append(idx)
                    dropped = False
                if not dropped:
                    out += [(k, "double") for k in pending_left]
                    out += [(k, "double") for k in pending_right]
                # dropped: 连桶带框全部丢弃
            pending_left = []
            pending_right = []
        else:
            if x0 < w / 4 and x1 < 3 * w / 4:
                pending_left.append(idx)
            elif x0 > w / 4 and x1 > w / 2:
                pending_right.append(idx)
            else:
                out += [(k, "double") for k in pending_left]
                out += [(k, "double") for k in pending_right]
                out.append((idx, "single"))
                pending_left = []
                pending_right = []
        pos += 1

    return {"order": [k for k, _ in out], "layouts": [l for _, l in out]}


def _compare(solve, bboxes, w=W):
    led = solve([list(b) for b in bboxes], w)
    exp = _oracle([list(b) for b in bboxes], w)
    assert led == exp, f"solve={led}\noracle={exp}\nbboxes={bboxes}"
    return led


def check_two_column_interleave(solve):
    # 两栏交替：左桶全部先于右桶输出
    bboxes = [[50, 100, 300, 200], [600, 100, 900, 200],
              [50, 250, 300, 350], [600, 250, 900, 350]]
    led = _compare(solve, bboxes)
    assert led["order"] == [0, 2, 1, 3]
    assert led["layouts"] == ["double"] * 4


def check_full_width_middle_flushes(solve):
    # 中部全宽块 → 先 flush 左右桶，本框 single
    bboxes = [[50, 100, 300, 200], [600, 100, 900, 200],
              [100, 400, 900, 500], [50, 600, 300, 700]]
    led = _compare(solve, bboxes)
    assert led["order"] == [0, 1, 2, 3]
    assert led["layouts"] == ["double", "double", "single", "double"]


def check_last_spanning_flush_before_append(solve):
    # 末框横跨中线且有纵向间隙：先 flush 两桶再追加 single（append 在 flush 之后）
    bboxes = [[50, 100, 300, 200], [600, 100, 900, 200], [100, 500, 900, 600]]
    led = _compare(solve, bboxes)
    assert led["order"] == [0, 1, 2]
    assert led["layouts"] == ["double", "double", "single"]


def check_last_column_box_appended_in_group(solve):
    # 末框不横跨：入左桶后 flush —— 末框在左桶组**内部**（J 锚点）
    bboxes = [[600, 100, 900, 200], [50, 100, 300, 200], [50, 500, 300, 600]]
    led = _compare(solve, bboxes)
    assert led["order"] == [1, 2, 0]
    assert led["layouts"] == ["double", "double", "double"]


def check_quarter_boundary_strict(solve):
    # x0 == w/4 两侧条件均不满足（严格不等号）→ 全部走 single / 末框分支
    bboxes = [[250, 100, 450, 200], [250, 300, 450, 400]]
    led = _compare(solve, bboxes)
    assert led["layouts"] == ["single", "double"]
    # x1 == 3w/4 → left 条件失败 → single
    led2 = _compare(solve, [[100, 100, 750, 200]])
    assert led2["order"] == [0] and led2["layouts"] == ["single"]


def check_degenerate_drop_discards_buckets(solve):
    # 末框 x0 == x1 == w/2 零宽框被静默丢弃，且**已攒的左桶一并丢弃**（F 锚点）
    led = _compare(solve, [[50, 100, 300, 200], [500, 300, 500, 400]])
    assert led["order"] == [] and led["layouts"] == []
    # 左右桶都有内容时同样全丢
    led2 = _compare(solve, [[50, 100, 300, 200], [600, 100, 900, 200],
                            [500, 500, 500, 600]])
    assert led2["order"] == [] and led2["layouts"] == []


def check_single_box_passthrough(solve):
    # 单框：不排序直接 single
    led = _compare(solve, [[600, 100, 900, 200]])
    assert led["order"] == [0] and led["layouts"] == ["single"]


def check_empty(solve):
    led = _compare(solve, [])
    assert led == {"order": [], "layouts": []}


def check_stable_sort_ties(solve):
    # (y0, x0) 相同 → 稳定排序保持原始顺序（H2 锚点）
    led = _compare(solve, [[50, 100, 300, 200], [50, 100, 300, 201]])
    assert led["order"] == [0, 1]
    # 仅 y0 相同、x0 不同 → x0 小者在前（H 锚点）
    led2 = _compare(solve, [[600, 100, 900, 200], [50, 100, 300, 200]])
    assert led2["order"] == [1, 0]


def check_last_gap_touching_not_spanning_branch(solve):
    # 末框 y0 == 前框 y1（相邻不重叠）：严格 > 不成立 → 走普通列分支，
    # 不进"横跨 single"分支（钉死 last-gap 严格不等号）
    bboxes = [[50, 100, 300, 300], [100, 300, 900, 400]]
    led = _compare(solve, bboxes)
    assert led["order"] == [0, 1]
    assert led["layouts"] == ["double", "double"]
    # y0 == prev y1 且 x0 >= w/2（右半窄框触线）→ 也不进横跨分支
    led2 = _compare(solve, [[50, 100, 300, 300], [500, 300, 900, 400]])
    assert led2["order"] == [0, 1]
    assert led2["layouts"] == ["double", "double"]


def check_sort_key_ignores_x1_y1(solve):
    # 排序键只用 (y0, x0)，与 x1/y1 无关：y0 相同时 x0 主导
    bboxes = [[600, 100, 900, 900], [50, 150, 300, 120]]
    led = _compare(solve, bboxes)
    assert led["order"][0] == 1  # (150, 50) < (100, 600)


def check_random_against_oracle(solve):
    rng = random.Random(691)
    for _ in range(300):
        n = rng.randint(0, 8)
        w = rng.choice([400, 800, 1000, 1200])
        bboxes = []
        for _i in range(n):
            x0 = rng.randint(0, w)
            x1 = rng.randint(x0, w)
            y0 = rng.randint(0, 800)
            y1 = rng.randint(y0, y0 + 300)
            bboxes.append([x0, y0, x1, y1])
        led = solve([list(b) for b in bboxes], w)
        exp = _oracle([list(b) for b in bboxes], w)
        assert led == exp, f"solve={led}\noracle={exp}\nbboxes={bboxes} w={w}"
        # 输出是原下标的排列（退化丢弃除外），且无重复
        assert len(set(led["order"])) == len(led["order"])
        assert all(l in ("single", "double") for l in led["layouts"])
