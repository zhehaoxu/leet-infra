# Notes

- 来源：MonteCLoRA（arXiv:2411.04358），官方 LCS2-IIITD/MonteCLoRA `peft/src/peft/tuners/tuners_utils_m.py` 的 `MonteCLoRASampler`（commit 0cd4a77）。
- 判题域：重参数化链的确定性核心（z_mvn/z_wishart/z_dirichlet 全部调用方给定）+ 专家权重归一 + 因子组合 + 三项 KL + 平方和"熵"；Wishart Bartlett 采样与异步预生成线程、MVN 分布对象不在域内。
- 模式（8）：std_to_sigma / wishart_reparam / multivariate_reparam / dirichlet_reparam / expert_weights / combine_factor / kl_terms / full_step。
- 官方坑位：专家权重是 sigmoid 后归一（不是 softmax）；"熵"是权重平方和（不是香农熵）；wishart_kl 只用 std@stdᵀ 的对角；dirichlet 是 log-normal 近似（chol(sigma)@z + mu）；clip 下限 1e-6。
- 锚值 float64；logdet 的 LU 误差 ~2.7e-12，multivariate_kl 解析对比容差放 1e-9，其余 1e-12。
- mut 17：exp 遗漏/去 diag/clip 下限/对角翻转/chol→cov/mean→sum/einsum 下标/scaler/mu 中心化/(1-2/M)/sigmoid/归一/组合 scaler/加权→均值/dirichlet 公式/熵公式/损失归一，全部 CAUGHT。
- 教训：mut anchor 不能撞 docstring（scaler 变异第一次替换了文档）；官方域内 cov 恒对角使 chol 转置真等价，需在接口合法范围内构造非对角 SPD cov 才能抓 einsum 下标变异。
