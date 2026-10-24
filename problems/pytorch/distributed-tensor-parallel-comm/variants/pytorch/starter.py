def solve(batch, seq, hidden, world, n_layers=1, bytes_per_elem=2):
    # TODO: 返回 (act_bytes, per_layer_comm, total_comm)；每层 4 次 all-reduce，环系数 (W-1)/W
    raise NotImplementedError
