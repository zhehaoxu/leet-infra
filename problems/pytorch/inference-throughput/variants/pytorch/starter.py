# prefill/decode FLOPs 与 TTFT / 总耗时
def solve(n_params, batch, prompt_len, gen_len, effective_flops):
    # TODO: fpt=2*N; prefill=batch*prompt_len*fpt; decode=batch*gen_len*fpt; 再除 effective_flops
    raise NotImplementedError
