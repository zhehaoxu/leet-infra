"""order 692: olmOCR `_merge_image_elements` union-find 容差合并账。

oracle 用 BFS 连通分量独立实现同一官方规则，与 solve 对拍；再加手算锚点。
"""
import random


def _oracle(images, tolerance=0.5):
    n = len(images)
    if n == 0:
        return []

    # 独立实现：邻接表 + BFS 连通分量（不用 union-find）
    def sep(b1, b2):
        h = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        v = max(0.0, max(b1[2], b2[2]) - min(b1[4], b2[4]))
        return h, v

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            h, v = sep(images[i], images[j])
            if h <= tolerance and v <= tolerance:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        comp = []
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(sorted(comp))

    # 输出顺序 = min index 升序；BFS 以 start 升序发现分量，天然有序
    out = []
    for comp in components:
        name = images[comp[0]][0]
        x0 = min(images[k][1] for k in comp)
        y0 = min(images[k][2] for k in comp)
        x1 = max(images[k][3] for k in comp)
        y1 = max(images[k][4] for k in comp)
        for k in comp[1:]:
            name += f"+{images[k][0]}"
        out.append({"name": name, "bbox": [x0, y0, x1, y1]})
    return out


def _compare(solve, images, tolerance=0.5):
    led = solve([list(im) for im in images], tolerance)
    exp = _oracle([list(im) for im in images], tolerance)
    assert led == exp, f"solve={led}\noracle={exp}\nimages={images} tol={tolerance}"
    return led


def check_overlap_merges(solve):
    led = _compare(solve, [["a", 0, 0, 1, 1], ["b", 0.8, 0.8, 2, 2],
                           ["c", 10, 10, 11, 11]])
    assert led == [{"name": "a+b", "bbox": [0, 0, 2, 2]},
                   {"name": "c", "bbox": [10, 10, 11, 11]}]


def check_tolerance_closed_interval(solve):
    # 触碰 h_dist=0 → 合并
    led = _compare(solve, [["a", 0, 0, 1, 1], ["b", 1, 0, 2, 1]])
    assert len(led) == 1 and led[0]["name"] == "a+b"
    # 恰好 gap=0.5 → 合并（闭区间 <=）
    led2 = _compare(solve, [["a", 0, 0, 1, 1], ["b", 1.5, 0, 2, 1]])
    assert len(led2) == 1
    # gap=0.6 → 不合并
    led3 = _compare(solve, [["a", 0, 0, 1, 1], ["b", 1.6, 0, 2, 1]])
    assert len(led3) == 2 and led3[0]["name"] == "a"
    # 垂直方向对称
    led4 = _compare(solve, [["a", 0, 0, 1, 1], ["b", 0, 1.5, 1, 2.5]])
    assert len(led4) == 1 and led4[0]["bbox"] == [0, 0, 1, 2.5]


def check_transitive_chain(solve):
    # A~B、B~C、A 与 C 自身 gap 0.8 > 0.5 → 传递闭包三者同分量
    led = _compare(solve, [["a", 0, 0, 1, 1], ["b", 0.9, 0, 2, 1],
                           ["c", 1.8, 0, 3, 1]])
    assert led == [{"name": "a+b+c", "bbox": [0, 0, 3, 1]}]


def check_name_order_by_index(solve):
    # 名字按原始下标升序拼接，与谁先出现/union 方向无关
    led = _compare(solve, [["z", 0.8, 0.8, 2, 2], ["a", 0, 0, 1, 1]])
    assert led[0]["name"] == "z+a" and led[0]["bbox"] == [0, 0, 2, 2]


def check_output_order_by_min_index(solve):
    # 分量 {0,1} 的 min 是 0，分量 {2} 的 min 是 2 → 大分量在前
    led = _compare(solve, [["p", 0.8, 0.8, 2, 2], ["q", 0, 0, 1, 1],
                           ["r", 10, 10, 11, 11]])
    assert [d["name"] for d in led] == ["p+q", "r"]
    # 交错下标：分量 {0,2} 与 {1,3} → {0,2} 在前
    led2 = _compare(solve, [["a", 0, 0, 1, 1], ["m", 10, 10, 11, 11],
                            ["b", 0.5, 0, 1, 1], ["n", 10.5, 10, 11, 11]])
    assert [d["name"] for d in led2] == ["a+b", "m+n"]


def check_containment_and_custom_tolerance(solve):
    # 包含关系 dist=0 → 合并
    led = _compare(solve, [["a", 0, 0, 10, 10], ["b", 3, 3, 4, 4]])
    assert led[0]["bbox"] == [0, 0, 10, 10]
    # tolerance=1.0：gap 恰 1.0 合并，1.1 不合并
    led2 = _compare(solve, [["a", 0, 0, 1, 1], ["b", 2, 0, 3, 1]], 1.0)
    assert len(led2) == 1
    led3 = _compare(solve, [["a", 0, 0, 1, 1], ["b", 2.1, 0, 3, 1]], 1.0)
    assert len(led3) == 2


def check_empty_and_single(solve):
    assert _compare(solve, []) == []
    led = _compare(solve, [["only", 0, 0, 1, 1]])
    assert led == [{"name": "only", "bbox": [0, 0, 1, 1]}]


def check_hv_both_required(solve):
    # h 远离但 v 相邻 → 不合并（两方向必须同时 <= tolerance）
    led = _compare(solve, [["a", 0, 0, 1, 1], ["b", 5, 0.5, 6, 1.5]])
    assert len(led) == 2
    # h 相邻但 v 远离 → 不合并
    led2 = _compare(solve, [["a", 0, 0, 1, 1], ["b", 0.5, 5, 1.5, 6]])
    assert len(led2) == 2


def check_all_in_one_component(solve):
    imgs = [["g0", 0, 0, 1, 1], ["g1", 0.4, 0.4, 1.4, 1.4],
            ["g2", 0.9, 0.9, 1.9, 1.9], ["g3", 1.6, 1.6, 2.6, 2.6]]
    led = _compare(solve, imgs)
    assert led == [{"name": "g0+g1+g2+g3", "bbox": [0, 0, 2.6, 2.6]}]


def check_random_against_oracle(solve):
    rng = random.Random(692)
    for _ in range(300):
        n = rng.randint(0, 10)
        tol = rng.choice([0.0, 0.25, 0.5, 1.0])
        imgs = []
        for k in range(n):
            x0 = round(rng.uniform(0, 20), 1)
            y0 = round(rng.uniform(0, 20), 1)
            x1 = round(x0 + rng.uniform(0.1, 3), 1)
            y1 = round(y0 + rng.uniform(0.1, 3), 1)
            imgs.append([f"im{k}", x0, y0, x1, y1])
        led = solve([list(im) for im in imgs], tol)
        exp = _oracle([list(im) for im in imgs], tol)
        assert led == exp, f"solve={led}\noracle={exp}\nimgs={imgs} tol={tol}"
        # 输出顺序：各分量 min index 升序；名字分量数 = 成员数
        assert [d["name"].split("+")[0] for d in led] == \
               [d["name"].split("+")[0] for d in sorted(
                   led, key=lambda d: int(d["name"].split("+")[0][2:]))] \
               if led else True
