import torch


def solve(params, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01, rho=1.0):
    """返回一个继承 `torch.optim.Optimizer` 的 Sophia-II 实例，`step(hessians=...)` 就地更新 params。

    Sophia-II（Liu et al., arXiv:2305.14342，Algorithm 3 的对角 Hessian 变体）：

      m_t = β1·m_{t−1} + (1−β1)·g_t
      H_t = β2·H_{t−k} + (1−β2)·Ĥ_t        （只有估算步才更新；否则 H 冻结仍可用）
      θ_t = θ_{t−1}·(1 − lr·λ) − lr·clip( m_t / max{H_t, ε}, −ρ, ρ )

    两个判题口径（唯一的 API 载体就是本 docstring）：
      * 分母是 H 的**一次方** max{H, ε}，不是 √H（那是 Adam 的口径）；Ĥ 整体翻倍 ⇒ 未夹紧
        时的位移精确减半。
      * clip 是逐元素夹到 [−ρ, ρ]，限幅由它承担；无权重衰减时每步 |Δp| ≤ lr·ρ 恒成立。
      * Hutchinson 估计 Ĥ **作显式入参**喂入：`step(closure=None, hessians=[...])`，hessians
        与 params 同长同序（无梯度的参数位置也占位）；`hessians=None` 表示本步不做 Hessian
        估算 ⇒ H 冻结，但 m 与 p 照常更新（论文 k 步复用的确定性表达）。
      * 状态初始化全零（m、H、step）；H 也从 0 开始做 EMA，因此前几步分母主要靠 ε 兜底、
        更新被 ρ 夹住——这是预期行为，不是 bug。
      * 解耦权重衰减只作用于 p（先于一切矩估计）；`grad is None` 的参数整体跳过。
      * 返回 Optimizer 实例本身（closure 返回值原样透传）。
    """
    raise NotImplementedError
