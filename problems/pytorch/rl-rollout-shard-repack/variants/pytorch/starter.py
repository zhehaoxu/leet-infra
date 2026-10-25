import torch


def solve(seq_lens, num_ranks, micro_budget):
    """RL rollout 数据的跨 rank 重分片 + padding-free 微批打包 + pad 开销账（纯整数）。

    入参：
      seq_lens     int64 (N,)，每条 rollout 序列的 token 数（≥1，按「生成分片」的下标顺序给出，
                   第 i 条的初始归属为 rank floor(i*num_ranks/N)）
      num_ranks    int ≥1，训练侧数据并行 rank 数 W
      micro_budget int ≥1，单个 padding-free 微批的 token 预算 M
    返回五元组（全部 int64）：
      dest_rank      (N,) 重分片后每条序列所属 rank
      micro_batch    (N,) 每条序列所属微批的全局编号（rank 升序、批内升序，从 0 连续编号）
      tokens_per_rank(W,) 每个 rank 的真实 token 总数（padding-free 代价；全零表示该 rank 空）
      pad_saved      (W,) 序列级 padding 的额外开销 = rank 内条数 × rank 内最长条 − 真实 token
      moved          (N,) 0/1，该序列的重分片目标是否 ≠ 初始归属 rank
    口径：均衡顺序 = 长度升序、等长按原下标升序；贪心取「当前 token 最少」的 rank（并列取最小 rank）；
      rank 内按均衡顺序做贪心装箱（放不下则另起一包；单条 ≥ 预算时独占一包，允许超预算）；
      N=0 时 dest_rank/micro_batch/moved 为空、tokens_per_rank/pad_saved 为长度 W 的全零。"""
    # TODO
    raise NotImplementedError
