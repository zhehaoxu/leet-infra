import torch


def solve(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    input_ln_weight: torch.Tensor,
    post_attention_ln_weight: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    rms_eps: float = 1e-6,
) -> torch.Tensor:
    """Pre-Norm Transformer 解码层：x + Attn(RMSNorm(x)) → + MLP(RMSNorm(·))。

    hidden_states : (B, S, hidden)
    cos, sin      : (B, S, head_dim)  已算好的 RoPE
    input_ln_weight, post_attention_ln_weight : (hidden,)  两个 RMSNorm 的 gain
    q/k/v_weight+bias, o_weight : GQA 注意力（q/k/v 带 bias，o 无 bias）
    gate/up/down_weight : SwiGLU MLP（无 bias）
    head_dim=cos.shape[-1]；rms_eps 默认 1e-6。
    严格 pre-norm + 加法残差；softmax fp32；因果 -inf 掩码；缩放 head_dim**-0.5。
    返回该层输出 (B, S, hidden)。
    """
    # TODO: norm→attn→残差→norm→mlp→残差
    raise NotImplementedError
