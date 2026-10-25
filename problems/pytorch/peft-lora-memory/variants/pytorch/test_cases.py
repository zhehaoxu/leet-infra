def _ref(layer_dims, r):
    """独立清点：用列表推导分别求和（结构异于参考解的逐层累加），交叉核对。"""
    full = sum(d_in * d_out for d_in, d_out in layer_dims)
    lora = sum(r * (d_in + d_out) for d_in, d_out in layer_dims)
    return full, lora, lora / full


def check_example(solve):
    full, lora, ratio = solve([(1024, 1024), (1024, 4096), (4096, 1024)], 8)
    assert full == 9437184, full
    assert lora == 98304, lora
    assert abs(ratio - 98304 / 9437184) < 1e-9, ratio


check_example.example = True


def check_matches_ref(solve):
    for dims, r in [([(64, 64)], 4), ([(512, 2048), (2048, 512)], 16),
                    ([(768, 768), (768, 3072), (3072, 768)], 8),
                    (([(128, 256), (256, 128), (128, 128)], 1))]:
        assert solve(dims, r) == _ref(dims, r), (dims, r)


def check_returns_ints_and_float(solve):
    full, lora, ratio = solve([(1024, 1024)], 8)
    assert isinstance(full, int) and isinstance(lora, int), (type(full), type(lora))
    assert isinstance(ratio, float), type(ratio)


def check_ratio_is_sum_not_mean(solve):
    # 全局 ratio 必须是「两侧求和之比」，不是逐层 ratio 的算术平均
    dims = [(512, 512), (512, 4096)]
    r = 8
    full, lora, ratio = solve(dims, r)
    per_layer_mean = sum((r * (a + b)) / (a * b) for a, b in dims) / len(dims)
    assert abs(ratio - lora / full) < 1e-12
    assert abs(ratio - per_layer_mean) > 1e-6, "ratio 应是求和之比而非逐层均值"


def check_r_equals_min_no_shrink_single_square(solve):
    # 单个方阵、r=d 时 lora=2*d^2 反而 > full=d^2（LoRA 不省），验证公式而非硬编码“更小”
    d = 32
    full, lora, ratio = solve([(d, d)], d)
    assert full == d * d
    assert lora == d * (d + d)
    assert abs(ratio - 2.0) < 1e-9, ratio
