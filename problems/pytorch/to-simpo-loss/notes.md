# 详解：SimPO 长度归一化偏好损失

## 出处
Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*（arXiv:2405.14734）。奖励 = 平均对数概率 \(r=\frac1{|y|}\log\pi_\theta(y|x)\)（无参考模型、长度归一），损失 \(-\log\sigma(\beta(r_w-r_l)-\gamma)\)，\(\gamma\) 为目标边际（论文里等价写作 \(-\beta\gamma\) 并入，本题按题面式实现）。

## 核心与易错点
1. **先除长度再作差**：\(r_w=\text{logp}_w/\text{len}_w\)，是「平均对数概率」之差，不是序列对数概率直接相减。长度归一是 SimPO 的灵魂。
2. **margin 里减 γ**：\(\beta(r_w-r_l)-\gamma\)，γ 越大要求优劣差距越大才降损失；忘记减 γ 会退化成朴素 sigmoid-DPO 形。
3. **logsigmoid 数值稳定**：用 `F.logsigmoid` 或对 `-m` 求 `log(1+e^{m})`，避免大数 exp 溢出。
4. 损失取 batch 平均、再取负。

## 对拍口径
参考解 `(-F.logsigmoid(beta*(lw/nw - ll/nl) - gamma)).mean()`。判题 `_oracle` 逐元素用 `math.log1p(exp(...))` 的稳定式 `logsigmoid(m)=-log1p(exp(-m))` 重算（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「γ=0 且 r 直接用未归一 logp 时与朴素式一致」「长度归一影响」断言。

## 复杂度
逐元素 \(O(N)\)。
