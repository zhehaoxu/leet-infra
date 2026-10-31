import torch


def solve(cache, attentions, key_or_value, level, symmetric, outliers_ratio,
          last_n_attentions, target_quantization_error, n_bits_min, n_bits_max,
          q_norm=None, method="uniform", quantile_upper=None, quantile_center=None):
    """QAQ：先按误差目标反解每单元的位宽，再用该位宽做仿射量化（确定性、无随机数）。

    cache        : float32/float64 (n_batch, seq_len, n_layer, n_head, head_dim)，即上游 permute 之后的布局
    attentions   : key 分支传 None；value 分支传 (n_layer, n_batch, n_head, q_len, seq_len)，
                   上游取最后 last_n_attentions 行（q 轴）后按 quantize_dims 取 amax
    key_or_value : "key" 或 "value"，决定 max_error 的两条公式
    level        : "token" / "layer" / "head" -> quantize_dims = (-3,-2,-1) / (-2,-1) / (-1,)
    symmetric    : True 时均值取 0、尺度取 2*max|x|；False 时取均值与 max-min
    outliers_ratio : [0,1) 的 outlier 占比（含最低与最高两端）
    last_n_attentions : int >= 1；>1 时末 last_n_attentions-1 个 token 的位宽无条件改成 n_bits_max
    target_quantization_error : > 0 的目标误差
    n_bits_min / n_bits_max : 0 <= n_bits_min <= n_bits_max <= 16
    q_norm       : key 分支必须给出的正数（query 的 2-范数）
    method       : "uniform"（minmax + round + clamp）或 "normal"（std + 查分位表）
    quantile_upper / quantile_center : method="normal" 必填，长度 >= n_bits_max+1 的序列/tuple，
                   第 n 项是长度 2**n 的一维张量：上界表 = norm.ppf(i/2**n + 1/2**n)（i=0..2**n-1，
                   末项 +inf）、中心表 = norm.ppf(i/2**n + 0.5/2**n)

    口径（内部 float64，返回按 cache 的 dtype）：
      1. outlier 掩码：把 quantize_dims 展平，lower = kthvalue(k=ceil(ratio/2*M))、
         upper = kthvalue(k=floor((1-ratio/2)*M))，掩码 = (cache <= lower) | (cache > upper)
         —— 低端含等号、高端不含，且 ratio=0 时直接全 False。
      2. max_error：key 为标量 sqrt(12/q_norm * log(L**3/(L-1) * target**2 + 1))；
         value 为 sqrt(12/L) * target / amax(最后 last_n 行 attentions 沿 quantize_dims)。
      3. 位宽：scale_value = 2*amax(|非 outlier|)（symmetric）或 amax-amin；然后
         n_bits = ceil(log2(scale_value/(2*max_error) + 1))，再 clamp 到 [n_bits_min, n_bits_max]；
         最后把末 last_n-1 个 token 的位宽强制改成 n_bits_max。
      4. 量化：按位宽分组，每组做 minmax/std 归一化 -> round（半偶）-> clamp 到
         [-(2**(b-1)), 2**(b-1)-1] -> 反归一化（normal 方法改用 searchsorted 落桶再取桶中心）；
         掩码位置保留原值。

    返回 dict，键固定为 5 个：
      outlier_mask   : bool (n_batch, seq_len, n_layer, n_head, head_dim)
      max_error      : float，形状同 n_bits（key 分支上游是个标量，这里广播到每个量化单元）
      n_bits_clamped : int64，ceil+clamp 之后、末几行覆盖之前
      n_bits         : int64，末几行覆盖之后（真正用于量化）
      quant_dequant  : float，量化再反量化的 cache，outlier 位置等于原值

    非法输入抛 ValueError。
    """
    # TODO
    raise NotImplementedError
