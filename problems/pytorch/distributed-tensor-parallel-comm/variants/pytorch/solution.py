def solve(batch, seq, hidden, world, n_layers=1, bytes_per_elem=2):
    W = float(world)
    act_bytes = float(batch) * float(seq) * float(hidden) * float(bytes_per_elem)
    c = (W - 1.0) / W if W > 0 else 0.0
    per_allreduce = 2.0 * c * act_bytes       # 一次 all-reduce 每 GPU 收发
    per_layer = 4.0 * per_allreduce           # 每层 4 次 all-reduce
    total = float(n_layers) * per_layer
    return (float(act_bytes), float(per_layer), float(total))
