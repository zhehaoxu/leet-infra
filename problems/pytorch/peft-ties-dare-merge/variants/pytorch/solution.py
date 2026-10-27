import torch


def solve(delta1, delta2, dare_p, mask1, mask2, keep_n):
    """DARE + TIES 两步合并。
    delta1,delta2,mask1,mask2: 同形 (R,C)；mask 取值 0/1（1=保留）；
    dare_p: 丢弃概率 [0,1)；keep_n: TIES trim 保留个数 (1..R*C)。
    delta_t' = mask_t * delta_t / (1-dare_p)；
    dare = (delta1' + delta2') / 2；
    ties: trim(|·| 降序, 并列取小下标) -> elect sign(sign a + sign b) ->
          disjoint mean（分母=符号一致的任务数，gamma=0 处为 0）。
    返回 (dare, ties) 两个与输入同形的 float32 张量。"""
    scale = 1.0 / (1.0 - float(dare_p))
    d1 = delta1.to(torch.float64) * mask1.to(torch.float64) * scale
    d2 = delta2.to(torch.float64) * mask2.to(torch.float64) * scale
    dare = (d1 + d2) / 2.0

    flat1, flat2 = d1.flatten(), d2.flatten()
    n = flat1.numel()

    def trim(v):
        order = sorted(range(n), key=lambda i: (-abs(float(v[i])), i))
        keep = set(order[:int(keep_n)])
        return [float(v[i]) if i in keep else 0.0 for i in range(n)]

    t1, t2 = trim(flat1), trim(flat2)
    out = []
    for a, b in zip(t1, t2):
        sa = 1 if a > 0 else (-1 if a < 0 else 0)
        sb = 1 if b > 0 else (-1 if b < 0 else 0)
        gamma = sa + sb
        gamma = 1 if gamma > 0 else (-1 if gamma < 0 else 0)
        if gamma == 0:
            out.append(0.0)
        else:
            members = [x for x in (a, b) if (1 if x > 0 else (-1 if x < 0 else 0)) == gamma]
            out.append(sum(members) / len(members))
    ties = torch.tensor(out, dtype=torch.float64).reshape(dare.shape)
    return (dare.to(torch.float32), ties.to(torch.float32))
