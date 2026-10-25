# 详解：每位置多候选的批量验证与残差采样

## 出处

- 单候选教科书口径：Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*（arXiv:2211.17192）与 Chen et al.（arXiv:2302.01318）——每位置只验证一个 draft token，拒绝后从 \(\mathrm{norm}((p-q)_+)\) 收尾。本库 `inf-spec-decoding-rejection` 实现的正是这一口径。
- 多候选来源：Miao et al., *SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference*（arXiv:2305.09781）——token tree 的每一层是并列多节点，等价于"每位置 \(C\) 个候选先到先得"的验证视图。本库 `spec-specinfer-token-tree-verify`、`spec-medusa-tree-verify` 验证的是整棵树；本题把树拉平成**定宽 \(C\)** 的每位置候选批，是两者之间的最小泛化。

## 完整推导与下标约定

**下标约定**：\(P\) 有 \(g+1\) 行（第 \(j<g\) 行对应位置 \(j\)，第 \(g\) 行是 bonus 行）；\(Q\) 只有 \(g\) 行；`cand`、`u` 均为 \((g,C)\)；`picks` 长度恒为 \(g\)（逐位置），`out` 长度 = 接受位置数 + 1。

**为什么"先到先得"是自洽的——残差单调收缩引理**：候选 \(i\) 被拒蕴含 \(\mathrm{ratio}<1\)，即 \(p[x_i]<q[x_i]\)（ratio 达到 1 时 \(u\le1\) 必收，不存在被拒的可能；\(q[x_i]=0,p[x_i]>0\) 的必收候选同理）。于是残差更新 \((p-q)_+\) 把它的质量清零。更强的不变量：一旦某 token 在残差里质量为 0，后续任何一次 \((p'-q)_+\) 更新都给它 \(0\)（因为 \((0-q)_+=0\)）。所以**位置内残差的支撑只收缩不扩张**，被拒 token 在同位置内"永不复活"，`cand` 重复出现它也绝不会再被接受——这正是先到先得语义不产生歧义的原因。

**多候选的收益（直觉参照，非逐位等式）**：设单候选对原始 \((P[j],Q[j])\) 的接受率为 \(\alpha=\sum_v\min(p_v,q_v)=1-\sum_v(p_v-q_v)_+\)。若把 \(C\) 个候选各自独立对原始分布验证，"至少收一个"的概率是 \(1-(1-\alpha)^C\)。本题的串行残差复用与该式并不逐位相等（被拒候选会搬走残差质量、改变后续分母），但方向一致：候选越多、越接近 target 支撑的候选越多，位置产出 token 的概率越高；且重复候选几乎无额外收益（被拒一次即出局）。

**两步更新的顺序为何不可交换**：拒绝时先 \(p\leftarrow\mathrm{norm}(\max(p-q,0))\)、再（可选）\(q[x]\leftarrow0,q\leftarrow\mathrm{norm}(q)\)。残差必须用**剔除前**的 \(q\) 计算——starter docstring 明确 no_replace"只影响同位置内后续候选的判定分母，不改变 p 的更新"。若先剔 \(q\) 再算残差，\((p-q)_+\) 的分母口径就变了，与判别用例直接冲突。

**CDF 反查的"严格大于"**：本题 `cdf(d,w)` 取最小 \(k\) 使前缀和**严格大于** \(w\)，与 `inf-spec-decoding-rejection` 的"\(\ge u\)"口径**刻意相反**。差异在踩线处可见：均匀分布行上 \(w=0.5\) 时，"\(\ge\)"给 \(k=0\)，"严格大于"给 \(k=1\)。这是两道题互为对照的判别点之一。兜底：全部前缀和 \(\le w\)（浮点尾差）时取 \(V-1\)。

**trial 计数与截断**：每个被尝试的候选记一次 trial（含被拒的）；位置 \(j\) 耗尽后，\(j+1..g-1\) 的所有候选既不产生 token 也不产生 trial。因此 \(0\le\text{trials}\le g\cdot C\)，全收时 \(=g\cdot C\)。

**空输入**：\(g=0\) 时没有验证发生，`Q` 形状 \((0,V)\)，直接退化为普通解码；\(C=0\)（\(g>0\)）等价于位置 0 当场耗尽——`picks=[-1]`、`out` 长度 1（残差即 \(P[0]\) 原样，因为没有任何拒绝发生）、`trials=0`。

## 核心与易错点

1. **两步更新顺序**：先残差、后剔 \(q\)；顺序写反会被 `no_replace` 判别用例抓住。
2. **归一化兜底**：残差总和为 0 时 \(p\) 保持原样；\(q\) 剔除后和为 0 时不归一化。两个兜底都不能省。
3. **\(q[x]=0\) 的候选**：\(p[x]>0\) 记 ratio 1（必收）；\(p[x]=0\) 记 ratio 0（只有 \(u=0\) 收）。不要除零。
4. **取等接受**：\(u=\text{ratio}\) 算接受，判别用例专门踩这条线。
5. **局部状态不跨位置**：\(p,q\) 的所有修改在进入 \(j+1\) 时作废，下一位置从 \(P[j+1],Q[j+1]\) 全新开始。
6. **bonus 行只在全收时用**：有任何位置耗尽，\(P[g]\) 就完全不参与；收尾 token 改从当前残差取。
7. **重复候选**：同一 id 重复出现时，未拒过的各次独立计 trial；拒过的一次出局、永久归零。
8. **长度语义**：有耗尽 ⇔ \(\text{len(out)}=\text{len(picks)}\) 且 `picks` 末元素为 −1；无耗尽 ⇔ \(\text{len(out)}=\text{len(picks)}+1\)。

## 与邻近题的区别

- **inf-spec-decoding-rejection**：\(C=1\) 的教科书口径；本题是其对 \(C\ge1\) 的泛化。两题 CDF 反查方向相反（那边前缀和 \(\ge u\)，本题严格大于），\(C=1\) 判别用例专门验证逐位塌回。
- **spec-medusa-tree-verify / spec-specinfer-token-tree-verify**：那两题验证整棵 token tree（树形注意力/树掩码、按路径取最优）；本题无树结构，每位置定宽 \(C\) 个并列候选、先到先得，是树验证的"逐层拉平"简化版。
- **spec-typical-acceptance**：接受准则换成 typical 采样式判据，与 \(p/q\) ratio 无关。

## 对拍口径与复杂度

- 独立重算采用与 `solve` 结构不同的纯 python 实现（显式前缀和循环 + 逆序 CDF 反查），`out`/`picks` 用 `torch.equal`、`trials` 用整数相等；260 组随机参数覆盖 \(g\in\{0..4\}\)、\(C\in\{0..4\}\)、重复候选、\(q\) 含零、\(u\) 踩线等角落。
- 判别口径三件套：\(C=1\) 塌回教科书式拒绝采样（与 `inf-spec-decoding-rejection` 互证）、\(u\) 恰踩 ratio 取等接受、`no_replace` 只动同位置后续分母。
- 复杂度：时间 \(O(g\cdot C\cdot V)\)（每候选一次 \(O(V)\) 残差更新与一次 \(O(V)\) 反查），额外空间 \(O(V)\)。\(V\) 是词表大小，实际系统中 \(g\cdot C\ll V\)，瓶颈与单候选推测解码相同，仍是 target 的一次 \((g+1)\) 位置前向。
