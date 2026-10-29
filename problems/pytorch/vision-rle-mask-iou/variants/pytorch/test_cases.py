import torch

# 示例（实测回填）：2x3 的两块掩码，A 在左上/右下，B 在左上两格与右下两格
EXAMPLE_SIZE = (2, 3)
EXAMPLE_MASK = [[1, 0, 0],
                [0, 0, 1]]
EXAMPLE_COUNTS_B = [0, 1, 1, 2, 1, 1]
EXAMPLE_COUNTS_A = [0, 1, 4, 1]
EXAMPLE_INTERSECTION = 2
EXAMPLE_UNION = 4
EXAMPLE_IOU = 0.5


def _mask_t(mask, dtype=torch.bool):
    return torch.tensor(mask, dtype=dtype)


def _rle(mask, dtype=torch.int64):
    """Python 侧规范列优先 RLE：先逐列扫描，再压游程。"""
    h, w = len(mask), len(mask[0])
    flat = [int(mask[y][x]) for x in range(w) for y in range(h)]
    runs, cur, n = [], flat[0], 1
    for v in flat[1:]:
        if v == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = v, 1
    runs.append((cur, n))
    out = [0] if runs[0][0] == 1 else []
    out.extend(n for _, n in runs)
    return torch.tensor(out, dtype=dtype)


def _oracle(mask, counts, size):
    """稠密展开 + 线性合并的纯 Python oracle（与参考解的游程归并结构不同）。"""
    h, w = int(size[0]), int(size[1])
    flat_a = [int(mask[y][x]) for x in range(w) for y in range(h)]
    vals = [int(v) for v in counts.tolist()]
    flat_b, cur = [], 0
    for n in vals:
        flat_b.extend([cur] * n)
        cur ^= 1
    assert len(flat_b) == h * w, (len(flat_b), h * w)
    inter = sum(1 for a, b in zip(flat_a, flat_b) if a == 1 and b == 1)
    union = sum(1 for a, b in zip(flat_a, flat_b) if a == 1 or b == 1)
    iou = inter / union if union > 0 else 0.0
    return iou, inter, union, _rle(mask)


def check_example(solve):
    iou, inter, union, ca = solve(_mask_t(EXAMPLE_MASK),
                                  torch.tensor(EXAMPLE_COUNTS_B, dtype=torch.int64),
                                  EXAMPLE_SIZE)
    assert iou.dtype == torch.float64 and iou.dim() == 0, (iou.dtype, iou.shape)
    assert inter.dtype == torch.int64 and union.dtype == torch.int64
    assert abs(iou.item() - EXAMPLE_IOU) < 1e-12, iou.item()
    assert inter.item() == EXAMPLE_INTERSECTION and union.item() == EXAMPLE_UNION, \
        (inter.item(), union.item())
    assert torch.equal(ca, torch.tensor(EXAMPLE_COUNTS_A, dtype=torch.int64)), ca.tolist()


check_example.example = True


def check_oracle(solve):
    """多种尺寸/形状上全量对拍：iou、交集、并集与重编码结果。"""
    cases = [
        ([[1]], [[1]]),
        ([[0]], [[1]]),
        ([[1, 0, 1, 1, 0, 1, 0]], [[0, 1, 1, 0, 0, 1, 1]]),
        ([[1], [0], [1], [1], [0]], [[1], [1], [0], [1], [0]]),
        ([[(i * 3 + j * 5) % 7 < 3 for j in range(4)] for i in range(5)],
         [[(i + j * 2) % 3 == 0 for j in range(4)] for i in range(5)]),
        ([[(i + j) % 2 for j in range(6)] for i in range(6)],
         [[(i * 2 + j) % 5 < 2 for j in range(6)] for i in range(6)]),
        ([[0, 0, 0], [0, 0, 0]], [[1, 0, 1], [0, 1, 0]]),
        ([[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]),
    ]
    for mask, other in cases:
        h, w = len(mask), len(mask[0])
        counts = _rle(other)
        iou, inter, union, ca = solve(_mask_t(mask), counts, (h, w))
        e_iou, e_inter, e_union, e_ca = _oracle(mask, counts, (h, w))
        assert abs(iou.item() - e_iou) < 1e-12, (mask, iou.item(), e_iou)
        assert inter.item() == e_inter and union.item() == e_union, \
            (mask, inter.item(), e_inter, union.item(), e_union)
        assert torch.equal(ca, e_ca), (mask, ca.tolist(), e_ca.tolist())


def check_column_major(solve):
    """列优先判定：同一 mask 的行优先编码与列优先编码不同。"""
    mask = [[1, 1, 0], [0, 0, 0]]
    iou, inter, union, ca = solve(_mask_t(mask), _rle(mask), (2, 3))
    # 列优先扫描 [1,0,1,0,0,0] -> [0,1,1,1,3]；行优先会得到 [0,2,4]
    assert torch.equal(ca, torch.tensor([0, 1, 1, 1, 3], dtype=torch.int64)), ca.tolist()
    assert abs(iou.item() - 1.0) < 1e-12 and inter.item() == 2 and union.item() == 2, \
        (iou.item(), inter.item(), union.item())
    # 同一 mask 用「行优先」编码 [0,2,4] 喂进来，按列优先解码只与 A 交叠 1 格
    iou2, inter2, union2, _ = solve(_mask_t(mask), torch.tensor([0, 2, 4], dtype=torch.int64),
                                    (2, 3))
    assert (inter2.item(), union2.item()) == (1, 3), (inter2.item(), union2.item())
    assert abs(iou2.item() - 1.0 / 3.0) < 1e-12, iou2.item()


def check_encoding_canonical(solve):
    """规范编码：全背景、全前景、首项 0、末尾背景游程、多段交替。"""
    checks = [
        ([[0, 0], [0, 0]], [4]),
        ([[1, 1], [1, 1]], [0, 4]),
        ([[1], [0]], [0, 1, 1]),
        ([[1, 0], [1, 0]], [0, 2, 2]),
        ([[1, 0, 1, 0, 1, 1]], [0, 1, 1, 1, 1, 2]),
        ([[0, 1, 0, 1, 0, 1]], [1, 1, 1, 1, 1, 1]),
    ]
    for mask, exp in checks:
        h, w = len(mask), len(mask[0])
        ca = solve(_mask_t(mask), _rle(mask), (h, w))[3]
        assert torch.equal(ca, torch.tensor(exp, dtype=torch.int64)), (mask, ca.tolist(), exp)


def check_self_iou(solve):
    """自比较恒等：IoU=1（空掩码除外），交集=并集=面积。"""
    masks = [
        [[1, 0, 0], [0, 1, 0]],
        [[1, 1], [0, 1]],
        [[1, 0, 1, 1]],
        [[0, 0], [0, 0]],
    ]
    for mask in masks:
        h, w = len(mask), len(mask[0])
        enc = _rle(mask)
        iou, inter, union, _ = solve(_mask_t(mask), enc, (h, w))
        area = sum(int(v) for row in mask for v in row)
        assert inter.item() == area and union.item() == area, (mask, inter.item(), union.item())
        expect = 1.0 if area > 0 else 0.0
        assert abs(iou.item() - expect) < 1e-12, (mask, iou.item(), expect)


def check_set_relations(solve):
    """包含关系与互斥关系：IoU 分别等于 area_a/area_b 与 0。"""
    # A ⊂ B
    a = [[1, 0], [0, 0]]
    b = [[1, 1], [0, 0]]
    iou, inter, union, _ = solve(_mask_t(a), _rle(b), (2, 2))
    assert (inter.item(), union.item()) == (1, 2) and abs(iou.item() - 0.5) < 1e-12
    # 互斥
    c = [[0, 0], [0, 1]]
    iou2, inter2, union2, _ = solve(_mask_t(a), _rle(c), (2, 2))
    assert (inter2.item(), union2.item()) == (0, 2) and abs(iou2.item()) < 1e-12
    # 全背景 vs 全前景
    iou3, inter3, union3, _ = solve(_mask_t([[0, 0], [0, 0]]),
                                   torch.tensor([0, 4], dtype=torch.int64), (2, 2))
    assert (inter3.item(), union3.item()) == (0, 4) and abs(iou3.item()) < 1e-12
    # 两张全背景：union=0 -> IoU 约定为 0
    iou4, inter4, union4, _ = solve(_mask_t([[0, 0], [0, 0]]),
                                   torch.tensor([4], dtype=torch.int64), (2, 2))
    assert (inter4.item(), union4.item()) == (0, 0) and iou4.item() == 0.0


def check_dtype_and_immutability(solve):
    mask = _mask_t(EXAMPLE_MASK, dtype=torch.uint8)
    counts32 = torch.tensor(EXAMPLE_COUNTS_B, dtype=torch.int32)
    mask0, counts0 = mask.clone(), counts32.clone()
    iou, inter, union, ca = solve(mask, counts32, EXAMPLE_SIZE)
    assert iou.dtype == torch.float64 and iou.dim() == 0
    assert inter.dtype == torch.int64 and union.dtype == torch.int64
    assert ca.dtype == torch.int64 and ca.dim() == 1
    assert abs(iou.item() - EXAMPLE_IOU) < 1e-12
    assert torch.equal(mask, mask0) and torch.equal(counts32, counts0)
    # bool 与 uint8 输入同结果
    iou_b = solve(_mask_t(EXAMPLE_MASK), torch.tensor(EXAMPLE_COUNTS_B), EXAMPLE_SIZE)[0]
    assert torch.allclose(iou, iou_b, atol=0.0, rtol=0.0)
    # int32 counts 与 int64 counts 同结果
    ca64 = solve(_mask_t(EXAMPLE_MASK), torch.tensor(EXAMPLE_COUNTS_B), EXAMPLE_SIZE)[3]
    assert torch.equal(ca, ca64)
    # torch.Size 也可作为 size
    assert abs(solve(_mask_t(EXAMPLE_MASK), torch.tensor(EXAMPLE_COUNTS_B),
                     torch.Size([2, 3]))[0].item() - EXAMPLE_IOU) < 1e-12


def check_errors(solve):
    mask = _mask_t(EXAMPLE_MASK)
    counts = torch.tensor(EXAMPLE_COUNTS_B, dtype=torch.int64)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(EXAMPLE_MASK, counts, EXAMPLE_SIZE), "mask 非张量")
    bad(lambda: solve(mask.unsqueeze(0), counts, EXAMPLE_SIZE), "mask 非二维")
    bad(lambda: solve(mask.to(torch.float32), counts, EXAMPLE_SIZE), "mask dtype")
    bad(lambda: solve(torch.tensor([[0, 2], [1, 0]], dtype=torch.uint8), counts, (2, 2)),
        "mask 非 0/1")
    bad(lambda: solve(mask, EXAMPLE_COUNTS_B, EXAMPLE_SIZE), "counts 非张量")
    bad(lambda: solve(mask, counts.unsqueeze(0), EXAMPLE_SIZE), "counts 非一维")
    bad(lambda: solve(mask, torch.zeros(0, dtype=torch.int64), EXAMPLE_SIZE), "counts 空")
    bad(lambda: solve(mask, counts.to(torch.float32), EXAMPLE_SIZE), "counts dtype")
    bad(lambda: solve(mask, torch.tensor([-1, 3, 4], dtype=torch.int64), EXAMPLE_SIZE),
        "counts 负值")
    bad(lambda: solve(mask, torch.tensor([0, 1, 0, 5], dtype=torch.int64), EXAMPLE_SIZE),
        "内部零长游程")
    bad(lambda: solve(mask, torch.tensor([1, 0, 5], dtype=torch.int64), EXAMPLE_SIZE),
        "非首项零长游程")
    bad(lambda: solve(mask, torch.tensor([0, 1, 4], dtype=torch.int64), EXAMPLE_SIZE),
        "元素和不足")
    bad(lambda: solve(mask, torch.tensor([0, 1, 5, 1], dtype=torch.int64), EXAMPLE_SIZE),
        "元素和超出")
    bad(lambda: solve(mask, counts, (2,)), "size 长度")
    bad(lambda: solve(mask, counts, (3, 2)), "size 与 mask 不一致")
    bad(lambda: solve(mask, counts, (0, 6)), "size 非正")
    bad(lambda: solve(mask, counts, (2.0, 3.0)), "size 非整数")
    bad(lambda: solve(mask, counts, None), "size 为 None")
