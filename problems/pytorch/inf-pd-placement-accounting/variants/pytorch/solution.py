def solve(prompt_lens, gen_lens, kv_bytes_per_token, caps, bws, tps, prefill_tps):
    n = len(prompt_lens)
    if len(gen_lens) != n:
        raise ValueError("prompt_lens 与 gen_lens 长度必须一致")
    if len(caps) != len(bws) or len(caps) != len(tps):
        raise ValueError("caps / bws / tps 描述的是同一组 decode 副本，长度必须一致")
    if n and any(p < 1 for p in prompt_lens):
        raise ValueError("prompt_lens 必须全部 >= 1")
    if n and any(g < 0 for g in gen_lens):
        raise ValueError("gen_lens 必须全部 >= 0")
    if kv_bytes_per_token < 1:
        raise ValueError("kv_bytes_per_token 必须 >= 1")
    if prefill_tps < 1:
        raise ValueError("prefill_tps 必须 >= 1")
    if any(c < 0 for c in caps):
        raise ValueError("caps 必须全部 >= 0")
    if any(b < 1 for b in bws):
        raise ValueError("bws 必须全部 >= 1")
    if any(t < 1 for t in tps):
        raise ValueError("tps 必须全部 >= 1")

    d_count = len(caps)
    placement = [-1] * n
    ttft = [-1] * n
    wire = [0] * d_count
    resident = [0] * d_count
    link_free = [0] * d_count
    rejected = 0
    makespan = 0
    cum_prompt = 0

    for i in range(n):
        cum_prompt += prompt_lens[i]
        prefill_done = -(-cum_prompt // prefill_tps)
        kv = (prompt_lens[i] + gen_lens[i]) * kv_bytes_per_token

        best_finish = None
        best_d = -1
        best_wire = 0
        for d in range(d_count):
            if resident[d] + kv > caps[d]:
                continue
            ring = -(-(kv * (tps[d] - 1)) // tps[d])
            moved = kv + ring
            start = prefill_done if prefill_done > link_free[d] else link_free[d]
            finish = start + -(-moved // bws[d])
            if best_finish is None or finish < best_finish:
                best_finish = finish
                best_d = d
                best_wire = moved

        if best_d < 0:
            rejected += 1
            continue

        placement[i] = best_d
        ttft[i] = best_finish
        wire[best_d] += best_wire
        resident[best_d] += kv
        link_free[best_d] = best_finish
        if best_finish > makespan:
            makespan = best_finish

    return placement, ttft, wire, resident, rejected, makespan
