import torch


def solve(weight, num_bits, group_size=-1, apot_nums=1):
    """ShiftAddLLM：先把权重矩阵贪心分解成位平面，再把每平面的标度压成少数几个有符号 2 的幂（确定性、无梯度、无随机数）。

    weight     : float32 或 float64 的 (R, C) 张量，待量化的权重矩阵（非连续张量按行主序读取）
    num_bits   : int >= 1，位平面个数（上游的 qbits）
    group_size : int，<= 0（含 -1）时退化成"整行一组"，即 g = C；> 0 时必须整除 R*C
    apot_nums  : int >= 1，每个标度用几个有符号 2 的幂相加来近似（上游的 apot_nums）

    口径（内部一律 float64，返回时按输入的 dtype 给出；n = R*C/g 为组数）：
      1. 分组：把权重**按行主序展平**后切成 n 个长度为 g 的组。g 不整除 C 时组会跨过行边界，
         这不是错误，而是上游 `w_.view([-1, group_size])` 的行为。
      2. 贪心位平面：r_0 = 该组的元素；对 i = 0..num_bits-1，
         b_i = sign(r_i)（取值 +1/-1/0），alpha_i = mean(|r_i|)（组内所有元素取均值），
         r_{i+1} = r_i - b_i * alpha_i。
         残差一旦落到 0，sign(0) = 0 会让它后面的所有平面都停在 0，不会跳回 ±1。
      3. ApOT 分解（对每个 alpha 独立做，round 是**半偶舍入**，不是 floor）：
         t_0 = 2**round(log2|alpha|) * sign(alpha)；
         对 k = 1..apot_nums-1，先取残差 d_k = alpha - (t_0 + ... + t_{k-1})，
         再取 t_k = 2**round(log2|d_k|) * sign(d_k)。
         alpha 或残差为 0 时 log(0) = -inf、2**(-inf) = 0、sign(0) = 0，该项是 0 而不是某个 ±2 的幂。
         alpha_apot = t_0 + ... + t_{apot_nums-1}。
      4. 重建：recon 的每个元素 = sum_i planes[该元素所在组][该元素][i] * alpha_apot[该元素所在组][i]，
         再按行主序还原成 (R, C)；mse = mean((weight - recon)**2)，对全部 R*C 个元素取均值。

    返回 dict，键固定为 7 个：
      group_shape : tuple[int]  (n, g)
      planes      : float (n, g, num_bits) 元素属于 {-1, 0, +1}，第 2 轴是位平面下标
      alpha       : float (n, num_bits)    贪心位平面的原始标度
      alpha_apot  : float (n, num_bits)    ApOT 分解之后的标度（重建用的是它，不是 alpha）
      apot_terms  : float (n, num_bits, apot_nums)  每个标度的各项，第 2 轴按提交顺序
      recon       : float (R, C)           用 alpha_apot 重建出的权重
      mse         : float 0-dim            weight 与 recon 的 MSE

    非法输入抛 ValueError：weight 不是 2 维 / 元素数为 0 / dtype 不是 float32 或 float64、
    num_bits < 1、apot_nums < 1、三者中任一不是 int（bool 也算非法）、
    group_size > 0 但不整除 R*C。
    """
    # TODO
    raise NotImplementedError
