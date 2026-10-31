def solve(latents, codebooks):
    """RVQ（Residual Vector Quantization，EnCodec/SoundStream/Mimi/DAC 系）的量化账。

    输入
    ----
    latents   : [N, D] 待量化向量
    codebooks : [K, C, D] 逐级码本（K 级、每级 C 个条目）

    记账（题面钉死，逐样本独立）：
      residual = latent
      对每级 k = 0..K-1：
        idx = argmin_c ||residual - codebooks[k][c]||^2（并列取最小下标）
        codes[k][n] = idx
        residual -= codebooks[k][idx]      # 残差逐级递减
      reconstructed = 全级被选码本条目之和
      residual_sq[k][n] = 第 k 级量化后残差的 L2^2（k=0..K-1）

    返回
    ----
    dict：
      codes          list[list[int]]  [K][N] 每级每样本的码索引
      reconstructed  list[[D]]        [N][D] 码本条目之和
      residual_sq    list[list[float]] [K][N] 每级量化后残差 L2^2
    """
    raise NotImplementedError
