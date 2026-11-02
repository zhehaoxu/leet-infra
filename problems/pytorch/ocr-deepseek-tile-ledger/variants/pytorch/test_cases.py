"""order 719: DeepSeek-OCR 五档动态分辨率与 tile/token 账。

oracle 用独立写法（显式列表构造 ratio 集合 + 手写最近匹配）对拍同一
官方语义；锚点全部手算（官方 vllm process/image_process.py + config.py）。
"""
import math
import random


# ---------------- oracle（独立实现） ----------------

def _oracle(width, height, base_size=1024, image_size=640, cropping=True,
            min_crops=2, max_crops=6, patch_size=16, downsample_ratio=4):
    assert width > 0 and height > 0 and base_size > 0 and image_size > 0
    if not cropping or (width <= image_size and height <= image_size):
        nw, nh = 1, 1
        tile_boxes = [[0, 0, image_size, image_size]]
    else:
        # 显式枚举去重（不用 set）
        seen = {}
        for n in range(min_crops, max_crops + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if min_crops <= i * j <= max_crops:
                        seen[(i, j)] = i * j
        ratios = sorted(seen.items(), key=lambda kv: kv[1])
        aspect = width / height
        bd, best = float("inf"), (1, 1)
        area = width * height
        for (i, j), _ in ratios:
            diff = abs(aspect - i / j)
            if diff < bd:
                bd, best = diff, (i, j)
            elif diff == bd and area > 0.5 * image_size ** 2 * i * j:
                best = (i, j)
        nw, nh = best
        tile_boxes = []
        for row in range(nh):
            for col in range(nw):
                tile_boxes.append([col * image_size, row * image_size,
                                   (col + 1) * image_size,
                                   (row + 1) * image_size])
    q = math.ceil(image_size // patch_size / downsample_ratio)
    qb = math.ceil(base_size // patch_size / downsample_ratio)
    gtok = qb * (qb + 1) + 1
    ltok = (q * nw + 1) * (q * nh) if nw > 1 or nh > 1 else 0
    return {"crop_ratio": [nw, nh], "tiles": nw * nh,
            "tile_boxes": tile_boxes, "global_tokens": gtok,
            "local_tokens": ltok, "num_image_tokens": gtok + ltok}


def _compare(solve, w, h, **kw):
    led = solve(w, h, **kw)
    exp = _oracle(w, h, **kw)
    assert led == exp, f"solve={led}\noracle={exp}\nw={w} h={h} kw={kw}"
    return led


def check_gundam_typical(solve):
    """2000x1500 (4:3) -> (3,2) 6 tiles；tokens 273+620=893（手算）。"""
    led = _compare(solve, 2000, 1500)
    assert led["crop_ratio"] == [3, 2] and led["tiles"] == 6
    assert led["tile_boxes"] == [[0, 0, 640, 640], [640, 0, 1280, 640],
                                 [1280, 0, 1920, 640], [0, 640, 640, 1280],
                                 [640, 640, 1280, 1280],
                                 [1280, 640, 1920, 1280]]
    assert led["global_tokens"] == 273
    assert led["local_tokens"] == (10 * 3 + 1) * (10 * 2) == 620
    assert led["num_image_tokens"] == 893


def check_square(solve):
    """2000x2000 -> (2,2)（(1,2)/(2,1) 平分被 (2,2) diff 0 覆盖）。"""
    led = _compare(solve, 2000, 2000)
    assert led["crop_ratio"] == [2, 2]
    assert led["local_tokens"] == 21 * 20 == 420
    assert led["num_image_tokens"] == 693


def check_wide_strip(solve):
    """3000x500 (aspect 6.0) -> (6,1) 单行 6 tiles。"""
    led = _compare(solve, 3000, 500)
    assert led["crop_ratio"] == [6, 1]
    assert led["local_tokens"] == 61 * 10 == 610


def check_small_gate(solve):
    """640x640 恰等（<=）不切；500x700 一维越界（and 门）即切 (2,3)。"""
    led = _compare(solve, 640, 640)
    assert led["crop_ratio"] == [1, 1] and led["local_tokens"] == 0
    assert led["num_image_tokens"] == 273
    led2 = _compare(solve, 500, 700)
    assert led2["crop_ratio"] == [2, 3]
    assert led2["local_tokens"] == (10 * 2 + 1) * (10 * 3) == 630
    assert led2["num_image_tokens"] == 903


def check_small_gate_lt(solve):
    """w=640 恰等 image_size：<= 不切（官方 <=）。"""
    led = _compare(solve, 640, 400)
    assert led["crop_ratio"] == [1, 1]


def check_tie_area_steal(solve):
    """1000x400 (aspect 2.5)：(2,1)/(3,1) diff 同 0.5，area=4e5 <
    0.5*640^2*3=614400 不抢 -> 最终 (2,1)。2500x1000 area=2.5e6 抢 ->
    (3,1)。两档分隔 area-thresh。"""
    led = _compare(solve, 1000, 400)
    assert led["crop_ratio"] == [2, 1]
    led2 = _compare(solve, 2500, 1000)
    assert led2["crop_ratio"] == [3, 1]


def check_native_modes(solve):
    """五档 native：Tiny/Small/Base/Large 序列账 73/111/273/421。"""
    assert _compare(solve, 800, 600, base_size=512, image_size=512,
                    cropping=False)["num_image_tokens"] == 73
    assert _compare(solve, 800, 600, base_size=640, image_size=640,
                    cropping=False)["num_image_tokens"] == 111
    assert _compare(solve, 800, 600, base_size=1024, image_size=1024,
                    cropping=False)["num_image_tokens"] == 273
    assert _compare(solve, 800, 600, base_size=1280, image_size=1280,
                    cropping=False)["num_image_tokens"] == 421


def check_crop_off(solve):
    """cropping=False：大图也 (1,1) 单 tile，仅全局账。"""
    led = _compare(solve, 3000, 3000, cropping=False)
    assert led["crop_ratio"] == [1, 1] and led["tiles"] == 1
    assert led["tile_boxes"] == [[0, 0, 640, 640]]
    assert led["local_tokens"] == 0 and led["num_image_tokens"] == 273


def check_ratio_bounds(solve):
    """min_crops=1 时 (1,1) 入集：800x800 方图 diff 0 -> 单 tile
    （nw=nh=1 不加局部账——官方 or 条件吞掉 1 tile 的 images_crop）。"""
    led = _compare(solve, 800, 800, min_crops=1)
    assert led["crop_ratio"] == [1, 1] and led["tiles"] == 1
    assert led["local_tokens"] == 0 and led["num_image_tokens"] == 273


def check_invalid_inputs(solve):
    bad = [lambda: solve(0, 500), lambda: solve(500, -1),
           lambda: solve(500, 500, base_size=0)]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected error")


def check_randomized(solve):
    rng = random.Random(719)
    for _ in range(200):
        w = rng.randint(50, 4000)
        h = rng.randint(50, 4000)
        kw = {}
        pick = rng.random()
        if pick < 0.2:
            kw = dict(base_size=rng.choice([512, 640, 1024, 1280]),
                      image_size=rng.choice([512, 640, 1024, 1280]),
                      cropping=False)
        else:
            kw = dict(min_crops=rng.randint(1, 3),
                      max_crops=rng.randint(4, 9))
            if rng.random() < 0.3:
                kw["image_size"] = rng.choice([512, 640])
                kw["base_size"] = rng.choice([1024, 1280])
        _compare(solve, w, h, **kw)


def check_no_pil(solve):
    """红线：不得 import PIL（AST 节点级）。"""
    import ast
    import os
    import sys
    mod = sys.modules.get(__name__)
    if mod is None or not getattr(mod, "__file__", None):
        return
    path = os.path.join(os.path.dirname(mod.__file__), "solution.py")
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "PIL":
                    raise AssertionError("banned import: PIL")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "PIL":
                raise AssertionError("banned import: PIL")
