# KV cache 显存 = 2(K,V) * layers * kv_heads * head_dim * seq * batch * dtype_bytes
def solve(layers, heads, head_dim, seq, batch, dtype_bytes=2, kv_heads=None):
    # TODO: kv_heads 为 None 时按 MHA 取 heads；否则用给定值（MQA=1, GQA 分组）
    raise NotImplementedError
