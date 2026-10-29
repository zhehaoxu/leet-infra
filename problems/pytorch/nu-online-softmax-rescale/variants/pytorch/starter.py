import torch


def solve(x, shard_size):
    """每行按 shard_size 切成连续的段（最后一段可以短），先各算段内 (m_s, l_s)，再重标定合并成整行 softmax。

    这是 sequence-parallel / 分片 logits 的真实形态：每个 rank 只交出自己段的 (max, sumexp)，
    合并方不需要原始 logits 就能把整条 softmax 算对。不许调用 torch.softmax / torch.logsumexp /
    torch.nn.functional.log_softmax，也不许先把整行拼回去一次算完（那样就没有重标定过程）。

    段内统计（对第 r 行的第 s 段 seg）：
        m_s = max(seg)                        # 段内最大值
        l_s = Σ exp(seg - m_s)                # 相对段内最大值的分母，段里至少有一个元素取到 exp(0)=1
    合并（M = max_s m_s）：
        L = Σ_s l_s * exp(m_s - M)            # 关键：把每段的分母重标定到同一个参考最大值上
        probs[j] = exp(x[j] - M) / L
        lse = M + log(L)
    必须始终对「已经减掉某个最大值」的量取 exp，参数永远 <= 0；出现 exp(900) 这种上溢写法就是错的。

    入参：
    - x: float32 或 float64 的二维张量 (rows, cols)，cols >= 1；内部用 float64 计算，
      输出转回输入的 dtype。
    - shard_size: >= 1 的整数。shard_size >= cols 时整行就是一段。
    返回 (probs, lse)：probs 与 x 同形状同 dtype，lse 形状 (rows,)。

    边界与特殊行（逐行判断）：
    - 行里出现 NaN → 该行 probs 全为 NaN，lse 为 NaN。
    - 行里有 k 个 +inf（k >= 1）→ 这 k 个位置的 probs 各为 1/k，其余位置为 0.0，lse = +inf。
      （torch.softmax 在这种行上给 NaN，本题故意约定成均分，因为工程上更需要「不炸」。）
    - 整行都是 -inf → 该行 probs 全为 NaN，lse = -inf。
    - 行内最大值有限但超出 float32 范围（例如 1e39）→ 仍是普通行，lse 必须保持有限；
      不要中途把段最大值转成 float32，那会把 1e39 折成 +inf 从而走错支路。
    - 某一段全是 -inf 但行内另有有限值 → 该段贡献 0，不影响结果。
    - rows == 0 → 返回形状正确的空结果，不报错。

    不同 shard_size 会改变累加顺序，结果只在最后一位上不同；判题容差是 float64 下 1e-9
    （相对输入 dtype 为 float32 时 1e-5），所以不要指望跨分块方案逐位复现。
    - x 不是二维、cols == 0、shard_size < 1、dtype 非浮点、非 Tensor 都抛 ValueError。
    """
    # TODO
    raise NotImplementedError
