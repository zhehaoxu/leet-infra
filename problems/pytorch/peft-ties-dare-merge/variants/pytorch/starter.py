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
    # TODO
    raise NotImplementedError
