import torch


def solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT):
    """单层 LSTM 沿时间反向传播（BPTT）：返回 (dxs, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0)。

    前向（门顺序与 PyTorch 的权重排布一致：i, f, g, o，即 input/forget/cell/output）：
    gates = x_t @ W_ihᵀ + h_{t−1} @ W_hhᵀ + b_ih + b_hh
    i = σ(gates[0:J]), f = σ(gates[J:2J]), g = tanh(gates[2J:3J]), o = σ(gates[3J:4J])
    c_t = f ⊙ c_{t−1} + i ⊙ g,  h_t = o ⊙ tanh(c_t)
    （把 forget 与 cell 两段切错位置，前向看着几乎正常、反向却全错——这是本题最常见的坑。）

    入参：
    - xs: (T, N, E) 逐步输入；h0/c0: (N, Hid)；W_ih: (4·Hid, E)；W_hh: (4·Hid, Hid)；
      b_ih/b_hh: (4·Hid,)；dhT/dcT: (N, Hid) 是末状态的来自上游的梯度。
    - batch 轴 N 可为 1；维度不符（如 W_ih 行数是 4·Hid、xs 末轴是 E）都抛 ValueError。

    口径：
    - 时间轴从 T−1 往回走，dh/dc 要**同时**累加「上游给的 dH_{t+1}」与本步产生的
      分量（截断 BPTT 时上游梯度可能为 0，仍要把本步的分量传下去）。
    - 门的导数：di = dct·g、dg = dct·i、df = dct·c_{t−1}、do = dht·tanh(c_t)，其中
      dct = dc_{传回} + dht ⊙ o ⊙ (1 − tanh(c_t)²)（**那个 o 因子来自 h_t = o ⊙ tanh(c_t)**，
      漏掉它在 T=1 就错）；折进 dgates 时 sigmoid 门要乘 y(1−y)、tanh 门乘 (1−y²)，
      dgates 的四段必须按 (i, f, g, o) 排，与 W_ih 的行块一一对应。
    - 权重梯度是**跨时间累加**：dW_ih = Σ_t dgates_tᵀ @ x_t、dW_hh = Σ_t dgates_tᵀ @ h_{t−1}。
    - 偏置梯度：gates 里 b_ih 与 b_hh 都是全长 (4·Hid,) 且各自独立相加，因此
      db_ih = db_hh = Σ_t dgates_t（两段完全相同的 4·Hid 向量），不要按门切成两半。
    - dxs 形状 (T, N, E)，每个时间步各一份；内部 float64，返回 dtype 与 xs 等各输入一致。
    """
    # TODO
    raise NotImplementedError
