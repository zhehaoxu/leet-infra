# KV Cache 显存估算参考解
def solve(layers, heads, head_dim, seq, batch, dtype_bytes=2, kv_heads=None):
    kvh = heads if kv_heads is None else kv_heads
    nbytes = 2 * layers * kvh * head_dim * seq * batch * dtype_bytes
    return int(nbytes)
