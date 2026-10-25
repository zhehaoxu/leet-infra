# LoRA 可训练参数量与显存账参考解
def solve(layer_dims, r):
    full = 0
    lora = 0
    for d_in, d_out in layer_dims:
        full += d_in * d_out
        lora += r * (d_in + d_out)
    return int(full), int(lora), lora / full
