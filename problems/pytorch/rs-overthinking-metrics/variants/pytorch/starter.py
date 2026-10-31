
def solve(correct, tokens, bin_edges):
    """Overthinking 指标账（题面钉死口径，全确定性）。

    参数
    ----
    correct   : list[bool]，N 条样本是否答对
    tokens    : list[int]，N 条样本的推理 token 数（>= 0）
    bin_edges : list[int]，严格递增、长度 K+1 >= 2；共 K 个分箱，
                箱 j（0 <= j < K-1）为 [edges[j], edges[j+1])，
                末箱为 [edges[K-1], +inf)（右端含）。

    返回
    ----
    dict：
      n            int     样本总数
      accuracy     float   总准确率
      total_tokens int
      bins         list[dict]  每箱 {'count', 'accuracy', 'mean_tokens'}；
                           accuracy/mean_tokens 空箱为 None
      len_correct  float|None  答对样本的平均 token 数（无答对为 None）
      len_wrong    float|None  答错样本的平均 token 数（无答错为 None）
      efficiency_ratio float|None  len_wrong / len_correct（口径缺失为 None）
      best_bin     int|None    准确率最高箱号（空箱不计；并列取最小箱号）
    """
    raise NotImplementedError
