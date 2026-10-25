# 详解：选择性激活重算

## 出处
Korthikanti et al. 2022，*Reducing Activation Recomputation in Large Transformer Models*（arXiv:2205.05198）。论文提出 sequence parallelism 与 **selective activation recomputation**：只重算「占显存小、重算便宜」的逐元素算子（Dropout / LayerNorm 等），保留 $SQK$、$S^2V$ 这类昂贵 GEMM 的输出，从而在同样显存下少重算。本题把它抽象成「显存预算 + 保底保留数下的最小重算代价」背包模型——模型本身是题面自定的组合优化，出处只提供动机与「选择性 = 部分保留部分重算」的核心思想。

## 核心与易错点
1. **翻译成背包**：最小化「重算代价」= 最大化「保留代价」，因为 $\sum_{i\notin K}c_i = \sum_i c_i - \sum_{i\in K}c_i$ 是常数差。于是变成：在 $\sum_{i\in K}\texttt{mem}_i\le B$、$|K|\ge$ `min_keep` 下最大化 $\sum_{i\in K}\texttt{cost}_i$。
2. **双约束 DP**：既要显存不超预算（重量维度），又要保留数下限（数量维度）。所以 `dp[j][m]` 同时按「保留块数 $j$」和「占用显存 $m$」两层展开；最后对 $j\ge$`min_keep` 取最优。只按显存做的普通背包会漏掉 `min_keep`。
3. **`min_keep` 会真正约束**：当最高 `cost` 的块 `mem` 很大、吃掉大半预算时，贪心「只留贵的」会让 $|K|<\texttt{min\_keep}$，被迫塞进便宜的小块——`check_min_keep_binds` 专门考这点。
4. **0/1 更新顺序**：每个块最多保留一次，$j$ 与 $m$ 都要**倒序**遍历，避免同一块在本轮被重复计入。
5. **不可行判定**：把 `mem` 最小的 `min_keep` 个相加已 $>B$（或 `min_keep>N`）时无解，抛 `ValueError`——用 DP 结果全为初始负值来检测，别忘 `min_keep=0` 恒可行（空集即可）。

## 对拍口径
参考解是 $(N\times N\times \texttt{budget})$ 背包 DP；判题 `_oracle` 对 $N\le 12$ 的用例**暴力枚举所有组合**（`itertools.combinations` 按保留数分层），完全独立地取最小重算代价交叉验证。另设「预算够大→重算 0」「`min_keep` 挤占」「`min_keep>N`/预算过小→ValueError」等断言。

## 复杂度
DP \(O(N^2\cdot \texttt{budget})\)（$N\le50$、budget$\le2000$）；暴力 oracle \(O(2^N)\)，仅用于小 $N$ 对拍。
