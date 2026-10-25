"""参考答案：sign/低秩压缩后的聚合载荷 -> 环 all-reduce 每卡字节，含 fp32 回退与 max_bits 切换点。"""


def _ceil_div(a, b):
    return (a + b - 1) // b


def _sign_payload(shapes, bits, group):
    """符号位按位打包 + 每组一个 fp32 scale；空矩阵跳过。"""
    total = 0
    for rows, cols in shapes:
        numels = int(rows) * int(cols)
        if numels == 0:
            continue
        total += _ceil_div(numels * int(bits), 8)
        total += _ceil_div(numels, int(group)) * 4 if int(group) >= 1 else 4
    return total


def solve(shapes, world, method="sign", bytes_per_element=4, bits_per_element=1,
          scale_group_size=0, rank=1, error_feedback=True):
    total_elements = 0
    for rows, cols in shapes:
        total_elements += int(rows) * int(cols)
    full_payload = total_elements * int(bytes_per_element)

    if method == "full":
        payload = full_payload
    elif method == "sign":
        payload = _sign_payload(shapes, bits_per_element, scale_group_size)
    else:                                    # lowrank（PowerSGD）
        payload = 0
        for rows, cols in shapes:
            rows, cols = int(rows), int(cols)
            numels = rows * cols
            if numels == 0:
                continue
            if int(rank) >= min(rows, cols):  # 秩吃满 -> 直接发原矩阵更省
                payload += numels * int(bytes_per_element)
            else:
                payload += (rows + cols) * int(rank) * int(bytes_per_element)

    compressed = min(payload, full_payload)
    fallback = payload > full_payload

    # 环系数沿用 distributed-zero-communication：一次 all-reduce 每 rank 搬 2*c*S
    c = (float(world) - 1.0) / float(world)
    per_gpu_comm_bytes = 2.0 * c * compressed
    full_comm_bytes = 2.0 * c * full_payload
    saved_ratio = 1.0 - compressed / float(full_payload) if full_payload > 0 else 0.0
    ef_state_bytes = total_elements * 4 if error_feedback else 0

    max_bits = 0
    if method == "sign" and full_payload > 0:        # payload(b) 对 b 单调不减 -> 扫描最大可行位
        for b in range(1, int(bytes_per_element) * 8 + 1):
            if _sign_payload(shapes, b, scale_group_size) <= full_payload:
                max_bits = b

    return (int(full_payload), int(compressed), float(per_gpu_comm_bytes),
            float(full_comm_bytes), float(saved_ratio), int(ef_state_bytes),
            bool(fallback), int(max_bits))
