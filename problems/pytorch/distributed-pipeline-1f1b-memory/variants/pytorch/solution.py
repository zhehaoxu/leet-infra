def solve(n_layers, n_stages, n_microbatches, act_bytes):
    lps = n_layers // n_stages
    per_mb = float(lps) * float(act_bytes)
    m = int(n_microbatches)
    p = int(n_stages)
    gpipe_peak = per_mb * m                 # 每段都存 m 份
    offfb_worst = per_mb * min(m, p)        # stage 0：min(m, p-0)
    offfb_best = per_mb * min(m, 1)         # stage p-1：min(m, 1)
    return (float(gpipe_peak), float(offfb_worst), float(offfb_best))
