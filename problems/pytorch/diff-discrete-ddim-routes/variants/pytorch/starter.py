def solve(logits, tokens, step_index, num_inference_steps, use_loo=False):
    """离散 DDIM（均匀腐蚀）单步的三路路由质量。

    logits 形如 (B, L, K)：每个位置的 K 类词表 logits；
    tokens 形如 (B, L) 的整数张量：当前 block 的 token，取值 [0, K)。

    时间账（alpha 越大越接近干净端）：N = num_inference_steps，
      alpha_t = step_index / N， alpha_s = (step_index + 1) / N
      a = alpha_t / alpha_s   （从 s 存活到 t 的概率）
      b = alpha_s

    先取贪心去噪结果 x0 = argmax(logits, -1)。use_loo=True 时先做 leave-one-out 修正：
    当 0 < alpha_t < 1，把「已观测 token」位置的 logit 减去
      delta = log1p(K * alpha_t / (1 - alpha_t))
    再取 argmax（alpha_t = 0 时不修正）。

    记 same = (tokens == x0)，三路质量为
      clean = b * (1 - a) / K + a * b * same
      stay  = a * (1 - b) / K
      noise = (1 - a) * (1 - b) / K
    再按三者之和归一化（注意这三元组本身并不总是和为 1：
    same 为假时和为 (1 - a*b)/K，所以这个除法不是摆设）。

    返回 (route_probs, pred_tokens)：
      route_probs 为 float32、形状 (B, L, 3)，最后一维依次是 clean / stay / noise；
      pred_tokens 为 int64、形状 (B, L)，即（可能经 LOO 修正后的）argmax。
    """
    # TODO
    raise NotImplementedError
