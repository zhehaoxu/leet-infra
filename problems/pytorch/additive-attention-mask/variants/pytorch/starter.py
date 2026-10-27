import torch


def solve(q_len, kv_len, seq_lens=None, kv_lens=None, kv_offset=0,
          causal=True, window=None, dtype=torch.float32):
    """构造 4-D 加性注意力 mask：(B,1,q_len,kv_len)，可看处 0.0，遮蔽处 finfo.min。

    这是 KV-cache 解码场景的加性 mask：i 是查询窗内 0-based 位置，
    j 是 KV 缓冲内的绝对位置，kv_offset 是已经缓存在 KV 缓冲左侧的
    旧 token 数（因此"当前查询行"对应 KV 缓冲内的对角线在列 j = i + kv_offset）。

    入参：
    - q_len / kv_len: 非负 int（负数或非整数抛 ValueError）；为 0 时返回
      尾维为空的 (B,1,0,kv_len) / (B,1,q_len,0) 张量。
    - seq_lens: None 或非负 int 序列（每个元素 <= q_len），第 b 项表示该 batch
      里有效的查询行数；i >= seq_lens[b] 的 pad 查询行整行不可看（整行 finfo.min）。
    - kv_lens: None 或非负 int 序列（每个元素 <= kv_len），第 b 项表示该 batch
      里有效的 KV 长度；j >= kv_lens[b] 的 pad KV 列不可看。
    - kv_offset: 非负 int（默认 0），语义见上。
    - causal: True 时叠加因果约束 j <= i + kv_offset；False 时不叠加。
    - window: None 或 >=1 的 int；给定时刻叠加局部带约束
      i - (j - kv_offset) < window（相对当前查询行往回最多 window-1 个 KV 位置，
      对角线恒可见）。window <= 0 抛 ValueError。注意 window 与 causal 独立：
      causal=False 时仅施加带约束（未来位置不受限）。
    - dtype: torch.float32 或 torch.float64（其他取值抛 ValueError）。
      遮蔽值取 torch.finfo(dtype).min。

    可见性判定（对每个 (b,i,j)，下列已启用条件同时满足才可看）：
    (a) causal=True：j <= i + kv_offset；
    (b) window 给定：i - (j - kv_offset) < window；
    (c) kv_lens 给定：j < kv_lens[b]；
    (d) seq_lens 给定：i < seq_lens[b]。

    batch 广播语义：
    - B = 各给定 lens 序列长度的最大值；都为 None 时 B=1。
    - 每个 lens 序列长度须为 B 或 1（长度 1 沿 batch 广播，例如
      seq_lens 长度 1 而 kv_lens 长度 B 时 seq 约束逐 batch 复制）；
      两者都给且长度都 >1 时必须相等，否则抛 ValueError。

    为什么遮蔽值用 finfo.min 而不是 -inf：加性 mask 会直接加到 logits 上。
    若某行全部被遮（如 pad 查询行、kv_lens[b]=0），用 -inf 会得到整行 -inf，
    softmax 输出 NaN；用 finfo.min 则该行 logits 加完仍为有限大负数，
    softmax 输出近似均匀分布（1/L 摊分），训练/解码端更稳。这也是
    HF transformers / flash-attn 加性 mask 的通用口径。

    返回：形状 (B,1,q_len,kv_len)、给定 dtype 的连续张量；可看处恰为 0.0，
    不可看处恰为 torch.finfo(dtype).min。不使用任何随机数。
    """
    # TODO
    raise NotImplementedError
