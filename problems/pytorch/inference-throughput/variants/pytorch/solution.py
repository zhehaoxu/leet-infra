# LLM 推理 prefill/decode 吞吐账参考解
def solve(n_params, batch, prompt_len, gen_len, effective_flops):
    fpt = 2 * int(n_params)                          # 每 token 前向 FLOPs（2N 近似）
    prefill_flops = int(batch) * int(prompt_len) * fpt
    decode_flops = int(batch) * int(gen_len) * fpt
    eff = float(effective_flops)
    ttft = prefill_flops / eff
    total = (prefill_flops + decode_flops) / eff
    return (int(prefill_flops), int(decode_flops), ttft, total)
