# Paraformer CIF 预测器 fire 发射账
#
# FunASR `funasr/models/paraformer/cif_predictor.py` 的 `cif()`（Paper:
# CIF 1905.11235）——Paraformer 用 Continuous Integrate-and-Fire 机制从
# encoder 隐态里按权重"积攒"出声学嵌入：每帧带一个 [0,1) 的发射权重
# alpha，累积越过阈值 1.0 即发射一个 token 嵌入，跨阈的余量留给下一
# 个 token。纯账本：alphas 与 hidden 全入参，不跑任何网络。
#
# 实现 solve(alphas, hidden, threshold=1.0, tail_threshold=0.45,
#            use_tail=True)：
#
#   输入
#     alphas         list[float]         每帧发射权重（官方 sigmoid 输出，
#                                        合法域 [0, 1)）
#     hidden         list[list[float]]   (T, D) encoder 隐态
#     threshold      float               fire 阈值，官方默认 1.0
#     tail_threshold float               尾部补发权重，官方默认 0.45
#     use_tail       bool                True 走官方推理路径（尾部补发）
#
#   输出 dict：
#     embeds          list[list[float]]  发射的声学嵌入，发射顺序排列
#     fire_positions  list[int]          每次发射发生在哪一帧（fire 判定帧）
#     token_num       int                use_tail=True: floor(总权重和)
#                                        use_tail=False: round(总权重和)
#                                        （官方两处口径）
#
# 状态机（照抄官方 cif()，单序列视角）：
#   1. integrate 从 0 起；每帧 t：
#      a. distribution_completion = 1 - integrate（当前 token 的缺口）
#      b. integrate += alpha_t
#      c. **fire 前值快照**（官方 list_fires.append 在减 1 之前）
#      d. fire = integrate >= threshold（**闭区间**，恰达阈值也发射）
#      e. fire 时 integrate -= 1；cur = 缺口（补齐当前 token），
#         否则 cur = alpha_t；remainds = alpha_t - cur（跨阈余量）
#      f. frame += cur * hidden_t；fire 时发射 frame（即当前 token 的
#         累计加权和），并将 frame 重置为 remainds * hidden_t
#   2. 发射嵌入 = fire 帧的 frame 快照；fire_positions 按 fire 判定帧升序。
#   3. 尾部余量（累积 < threshold）默认丢弃；use_tail=True 时官方
#      tail_process_fn 在 alphas 末尾**追加一帧 alpha = tail_threshold**，
#      并给 hidden 追加一行**零向量**——tail 帧恰对该零行（官方不变量
#      len(hidden) == len(alphas)），补缺口的权重乘零向量，即末 token
#      的嵌入只由真实帧累计而成。
#   4. use_tail=True 时单序列发射数恒等于 floor(总权重和)（可自证）；
#      use_tail=False 时发射数由快照决定，可与 round(总和) 不同。
#
# 示例：alphas=[0.5,0.5,0.5,0.5], hidden=[[1,0],[0,1],[2,2],[3,3]]
#   → fire at 1、3；embeds=[[0.5,0.5],[2.5,2.5]]（第 2 枚 = 0.5h2+0.5h3）
#   alphas=[0.6,0.6,0.6], hidden 3 帧（use_tail=False）
#   → fire 仅 at 1；embeds=[[0.6,0.4]]（余 0.2 丢弃）；token_num=round(1.8)=2
#     ——发射数 1 ≠ token_num 2，正是无尾部补发时的口径差
#   同输入（use_tail=True）：tail 帧对零行，fire at 1、3；
#   embeds=[[0.6,0.4],[1.2,1.4]]（第 2 枚 = 0.2h1+0.6h2，零行不贡献）
