# 解题笔记 — add_weighted_adapter

## 官方账

- HF PEFT `peft/tuners/lora/model.py`（LoraModel.add_weighted_adapter + _check_add_weighted_adapter + _generalized_task_arithmetic_weighted_adapter + _svd_generalized_task_arithmetic_weighted_adapter）；离线加权 adapter 合并。
- 判题域：rank 规则、scaling 归一、cat/linear/svd 三条合并路径、bias 合并。Embedding/Conv 分支、TIES/DARE/ties_svd/dare 族不在域内（`peft-ties-dare-merge` 已占论文级算子）；modules_to_save/target_parameters/DoRA/KaSA 的前置 ValueError 分支不入题。

## 踩坑

- **占位锚值即 FAIL 真值**（本树第 N 次重演）：四个锚点全部先跑 solution 回填真实值，写猜值必挂。
- **clamp 测试的分位数可能为负**：q=0.5 的中位数落在负数区时，`clamp(-hi, hi)` 边界反转（min>max），torch 会把所有元素截成 max——变异体和期望解**同错同对**，变异测试假通过。改用 q=0.9 并显式断言 `hi > 0`。
- **linear 分支 √ 判别力**：测试里 `|w·s|` 若全为 1（如 0.5×2、−2×0.5），去 sqrt 变异无差（√1=|1|）——权重/缩放组合必须设计出 |w·s|≠1（0.5×2=1 与 −2×1=2）才能杀死 `sqrt→abs` 变异。
- **svd 的 S 吸收进 B**：`U@diag(S)` 与「S 乘到 Vh 上」的重构积完全相同（结合律），重构断言杀不死这个变异——用 `B` 列范数 == |S|（U 列正交）钉死 kohya 约定。
- **Vh 截断切片越界是静默的**：`Vh[:rank+1, :]` 当 rank=min_dim 时返回原张量不报错——截断断言必须在 rank 严格小于 min_dim 的用例上做（new_rank=2 < 6）。
- **单源强制 linear 的返回值断言**：不只断 new_rank，combination_type 字段也在 dict 里一起断（否则 `cat→linear` 改写变异漏网——单源 cat 的 new_rank 恰好等于线性规则值）。
- svd 分支组合用**原始权重**（scaling 已在 delta 里）；linear/cat 分支才显式乘 scaling——三处口径不一致是本题最大考点。

## mut 15 变异清单

resolve：单源强制→cat、linear 等秩检查移除、cat sum→max、svd max→min、rank_pattern max→min、scaling ÷→×；cat：B 误缩放、B 拼接 dim=1→0（形状错）；linear：A 侧去符号、B 侧 sqrt→abs；svd：delta 丢 scaling、S 吸收进 B 改为丢弃、Vh 截断 +1、clamp 对称→单边；bias：丢 scaling。15/15 全 CAUGHT。
