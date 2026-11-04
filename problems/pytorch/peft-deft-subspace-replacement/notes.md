# 解题笔记 — DeFT

## 官方账

- HF PEFT `peft/tuners/layer.py`（DeFT 层）+ `decomposition.py`（relu/qr 两种投影）+ 文档；解耦子空间替换微调（PaRa 派生）。
- 判题域：秩解析、投影分解、merge/unmerge 因子、delta 恒等式、activation 分支前向。训练循环与数据并行不在域内。

## 踩坑

- **dict 解构得键名字符串**：`Q_P, right = solve(mode="project", ...)` 把返回 dict 解构成两个键名字符串，后续矩阵乘直接 TypeError——必须 `out["Q_P"]` 键访问。
- **branch 返回完整输出**：不是纯增量，而是 `base_product - (base_product @ right) @ Q_P.T`（移除项）`+ F.linear(x, R * scaling) @ Q_P.T`（注入项）；恒等式是 `out == base_product + x @ delta.T`，写成 `out == x @ delta.T` 会全挂。
- **identity init 的正确断言**：`R_init = right.T @ W / scaling` 使初始 delta 恰为零，因此 `out0 == base_product`（而非 delta 为任意零张量）；para 模式忽略 R、只有移除项（delta 非零）。
- **fp32 管线容差**：投影/合并/分支全程 `to(torch.float32)`，float64 test 下恒等式差 ~1e-6，容差 1e-5（与 OSF/CorDA 同因）；dtype 对拍需 `to(torch.float32)`。
- **占位锚值即 FAIL 真值**：锚值必须先跑通 solution 确认回填，不能先写猜测值（否则恒假断言伪装成通过）。
- **relu 投影非正交**：`P_proj = P @ relu(P).T` 不是幂等正交投影（qr 才是），不能对 relu 分支断言正交性/幂等性，只断言 `right == relu(P)`。

## mut 14 变异清单

resolve：min→max（夹取方向反）、alpha/r→alpha*r；project：relu→sigmoid（right 因子）、qr(P)→qr(P.T)（正交基错）；merge_factor：W→W.T（形状/语义）、right.T@W→right@W；delta：R−factor→factor−R（符号）、para 去负号、R*scaling→R+scaling；init_R：/scaling→*scaling；branch：去移除项、注入项加→减、Q_P.T→Q_P（转置错位）、移除项 Q_P/right 交换（correction 错位）。14/14 全 CAUGHT。
