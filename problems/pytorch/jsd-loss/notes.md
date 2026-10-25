# 详解：Jensen-Shannon 散度损失

## 出处
Lin, *Divergence Measures Based on the Shannon Entropy*（IEEE Trans. Inf. Theory 1991）；Menéndez et al. 1997 引入带权 \(\beta\) 的一般形式。定义 \(\mathrm{JSD}_\beta(P\|Q)=\beta\mathrm{KL}(P\|M)+(1-\beta)\mathrm{KL}(Q\|M)\)，\(M=\beta P+(1-\beta)Q\)。性质：对称（\(\beta=0.5\)）、非负、以 \(\ln 2\) 为上界。现代用法见 Hinton 蒸馏、伪标签一致性（FixMatch/ReCoNet）等。

## 核心与易错点
1. **必须经过混合分布 M**：JSD 不是两个 KL(P‖Q) 的平均，而是各自对 **同一个** \(M\) 的 KL 加权和。写成 \(\frac12\mathrm{KL}(P\|M)+\frac12\mathrm{KL}(Q\|M)\)（\(\beta=0.5\)）。
2. **\(0\log 0=0\) 且分母安全**：用掩膜只累加 \(P_c>0\)（或 \(Q_c>0\)）的项。由于 \(M\ge\beta P\)，凡 \(P_c>0\) 必有 \(M_c>0\)，不会除零。
3. **上界 \(\ln 2\)**：不相交支撑（`[1,0]` vs `[0,1]`）时恰达 \(\ln 2\)，用来做数值锚点。
4. **对称性**：\(\beta=0.5\) 时 `solve(P,Q)==solve(Q,P)`，判题用它交叉验证。

## 对拍口径
参考解 `_kl(A,B)` 掩膜累加、`solve=beta*_kl(P,M)+(1-beta)*_kl(Q,M)`，逐行后对 N 行取均值（复用 kl-divergence 同款 _kl，M 传入）。判题 `_oracle` 纯 Python 由 P、Q 逐元素构造 M 再按定义累加，结构不同、数值等价。另设「不相交=ln2 上界」「P==Q→0」「对称性」「example 数值」断言。

## 复杂度
\(O(N C)\)。
