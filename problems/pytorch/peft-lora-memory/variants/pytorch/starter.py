def solve(layer_dims, r):
    """LoRA 可训练参数账：layer_dims 是 [(d_in, d_out), ...]，r 为秩。
    返回 (full, lora, ratio)：full=sum(d_in*d_out), lora=sum(r*(d_in+d_out)), ratio=lora/full。
    full/lora 为 int，ratio 为 float。"""
    # TODO
    raise NotImplementedError
