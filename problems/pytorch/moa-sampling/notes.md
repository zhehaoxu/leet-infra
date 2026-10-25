# 详解：多路分布混合采样（MoA / Mixture-of-Samplers）

## 出处
- **llama.cpp 的部分（已回源核对，逐条说明）**：本题 ① 的两个子口径直接取自 llama.cpp，我在本次抓取到的 `src/llama-sampler.cpp` 原文里确认过——
  - `llama_sampler_temp_impl` 在 `temp <= 0.0f` 时**不做 softmax**，而是找出最大 logit、把它之外的全部写成 `-INFINITY`，等价于题面里的「one-hot(argmax)」；并列时保留第一个遇到的严格更大者，也就是最小下标。
  - `llama_sampler_top_p_apply` 的判据是「降序累积和 `cum_sum >= p` 时把截止点取在 `i + 1`」，即**首次达到 \(p\) 的那个元素仍然保留**，并有 `min_keep` 兜底至少留几个——与合集 `top-p-sampling` 的口径一致，本题每个专家核的截断判据就是它。
- **不属于 llama.cpp 主线的部分**：把多条采样链并联、过采样取并集、在并集上加权混合这一步，是 llama.cpp 生态里的**装配技巧**（社区称之为 MoA / mixture-of-adaptive-samplers），不是常驻的库内 stage：当前 master 的 `common/sampling.cpp` 里 `common_sampler_type_to_str` 只返回 `dry / top_k / typ_p / top_p / top_n_sigma / min_p / temperature / xtc / infill / penalties / adaptive_p`，**没有 moa**，`src/llama-sampler.cpp` 里也 grep 不到任何 `*moa*` 实现。所以本题按「实现技巧 + starter 的数学」描述与判分，不宣称上游有一行对一行的源码。
- **两个 MoA 不是一回事，别引错**：题集的立项条目把本题写成「llama.cpp moa / 2406.04692」。核对结果：arXiv:2406.04692 是 Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities*（2024，ICLR 2025），它的 MoA 是**分层多个 LLM 互相阅读对方的文本输出、再由 aggregator 续写**，作用在**文本域**、需要额外的前向与拼接，与本题「同一个位置上的 \(E\) 条概率分布加权求和」数学完全不同，故**不作为题面依据**，只在名称上同字。而「Mixture-of-**Adaptive**-samplers」这篇（本题过程口径真正对应的学术来源）的 arXiv 编号在本次受限网络里未能核对上——**待回源**。另记：立项材料里一度出现过的 2401.10096 经查是 *Long time regularity for 3d gravity waves with vorticity*（Ginsberg, 2024），一道数学物理论文，与本题无关，**不可引用**。

## 完整推导与下标约定
记专家数 \(E\)、词表 \(V\)，下标 \(e\) 指专家、\(t\) 指 token 下标（\(0\le t<V\)），一律 **0-based**。

1. **单专家分布**。
   \[
   p^{(e)}_t=\begin{cases}\mathbb{1}\!\left[t=\arg\max_{t'} \text{logits}_{e,t'}\right], & \tau_e\le 0\\[3pt]
   \dfrac{\exp(\text{logits}_{e,t}/\tau_e)}{\sum_{t'}\exp(\text{logits}_{e,t'}/\tau_e)}, & \tau_e>0\end{cases}
   \]
   令 \(\pi_e\) 为 \(p^{(e)}\) 的降序排列（同值按下标升序），\(c^{(e)}_j=\sum_{i\le j}\pi_e[i]\)。核 \(K_e=\{\pi_e[0],\dots,\pi_e[j_e-1]\}\)，其中 \(j_e=\min\{j: c^{(e)}_j\ge p_e\}\)，且 \(j_e\ge 1\)（\(p_e=1\) 时核就是全词表）。\(q_e\) 是 \(p^{(e)}\) 在 \(K_e\) 上除以 \(c^{(e)}_{j_e-1}\)、其余位置写 \(0\) 后**放回原下标顺序**的分布。
2. **过采样与联合候选集**。专家 \(e\) 的第 \(j\) 枪取
   \[
   s_{e,j}=\min\Big\{t:\ \sum_{t'\le t} q_e(t')\ \ge\ u^{\text{exp}}_{e,j}\Big\},
   \]
   注意求和是**沿原下标 \(0,1,\dots,V-1\) 的 CDF**，不是按概率大小；核外位置质量为 \(0\)，CDF 在那里不动，所以 \(s_{e,j}\in K_e\)。
   \(\mathcal U=\operatorname{sort}\big(\{s_{e,j}: 1\le e\le E,\ 1\le j\le m\}\big)\)，\(|\mathcal U|\le E\cdot m\)（示例里 \(E m=4\) 枪只得 \(3\) 个候选，因为专家 0 的两枪都落在 token 0）。
3. **在 \(\mathcal U\) 上再归一化后混合**。\(D_e=\sum_{t\in\mathcal U} q_e(t)\)，\(r_e(t)=q_e(t)/D_e\)（\(t\in\mathcal U\)），\(D_e=0\) 的专家被整个跳过。
   \[
   p_{\text{mix}}(t)=\sum_{e:\,D_e>0}\frac{w_e}{\sum_{e'}w_{e'}}\,r_e(t),\qquad
   \text{filtered}(t)=\begin{cases}p_{\text{mix}}(t)/\sum_{t'\in\mathcal U}p_{\text{mix}}(t'), & t\in\mathcal U\\ 0, & \text{否则}\end{cases}
   \]
   顺序很要紧：**先按 \(\mathcal U\) 归一化各专家、再加权**，而不是「先加权、后限制到 \(\mathcal U\)」。用示例数据：\(D_0=0.755271\)（因为 token 1 进了专家 0 的核、却没进 \(\mathcal U\)，专家 0 有 \(1-0.755271=0.244729\) 的质量被排除），\(r_0=[0.880797,0.119203,0.0]\)；\(D_1=0.817862\)，\(r_1=[0.0,0.777300,0.222700]\)；\(\text{filtered}\big|_{\mathcal U}=[0.616558,0.316632,0.066810]\)。
4. **末次采样**与逆变换口径同上：\(\text{chosen}=\min\{t:\sum_{t'\le t}\text{filtered}(t')\ge u\}\)，找不到就回退 \(V-1\)。示例里 CDF 为 \([0.616558,0.616558,0.933190,1.0,1.0]\)，\(u=0.5\) 首项即命中 → \(0\)。

## 核心与易错点
1. **混合发生在概率域、且发生在截断之后**。每条链的截断（温度 + top-\(p\)）先各自完成，混合的是**截断并归一化后的** \(q_e\)；写成「\(E\) 条 logits 加权平均再 softmax」是另一件事（logit 域平均 = 概率域几何平均型，永远拿不到题面数值）。
2. **「先在 \(\mathcal U\) 上再归一化」不可省，也不可挪到最后**。专家在 \(\mathcal U\) 上的质量 \(D_e\) 各不相同（示例里 \(0.755271\) 对 \(0.817862\)），少了这一步就等于让「核更宽的专家」白占权重。
3. **最后那次归一化在 \(m\ge1\) 时几乎不动数值**：每个 \(r_e\) 都已经在 \(\mathcal U\) 上积分为 \(1\)，\(\sum_e w_e/\sum w=1\)，故 \(p_{\text{mix}}\) 的总质量恒为 \(1\)。我用 400 组随机配置（含 \(\tau=0\) 的贪心专家、\(E\le3\)、\(V\le9\)）实测该总质量落在 \([0.9999998808,\,1.0000001192]\)——纯粹是 float32 舍入，所以它是**防御性**的（真正的触发条件是某专家在 \(\mathcal U\) 上质量为 \(0\)，而 \(m\ge1\) 时每个专家至少有一枪落在 \(\mathcal U\) 内、故 \(D_e>0\) 恒成立）。防御性不等于可省：省掉后一旦上游改动（例如允许 \(m=0\) 或负权重）就会静默不归一。
4. **权重为 \(0\) 的专家仍然进 \(\mathcal U\)**——过采样发生在加权之前。实测：`weights=[0,1]`、两路各抽 \(1\) 枪时 \(\mathcal U=\{0,3\}\)，最终 `filtered=[0.017986, 0, 0, 0.982014]`，即「有效专家在 \(\mathcal U\) 上再归一化」，而 token 0 之所以拿到 \(0.017986\) 正是因为它被那个**零权重**专家拽进了候选集。
5. **\(\tau_e\le0\) 的专家是「一票定死」**：one-hot 的 CDF 在 argmax 之前是 \(0\)、之后是 \(1\)，\(m\) 枪全落在同一个 token（示例 \(\mathcal U=\{1\}\)、`filtered` 就是 one-hot）。所以「贪心路 + 发散路」混合时，贪心路的 \(m\) 枪其实只贡献 \(1\) 个候选——它进 \(\mathcal U\) 的方式与随机路不同。
6. **逆变换一律用原下标顺序**，`filtered` 里 \(\mathcal U\) 外的 \(0\) 会让 CDF 出现平台，首个 \(\ge u\) 的下标必然跳过它们：示例里 \(u=0.7\) 落到 token 2、\(u=0.99\) 落到 token 3。用「排序后按名次取」的写法会把 token 2/3 搞反。
7. **不要用 `torch.multinomial`**：它自带随机状态、不保证与显式 \(u\) 的逆变换等价（也不保证 tie 行为），合集口径一律手算 CDF。
8. **判题不覆盖的退化**：\(m=0\) 时 \(\mathcal U=\varnothing\)，参考解返回全零 `filtered` 与回退位 \(V-1\)（实测），题面 \(m\ge1\) 已排除。

## 与邻近题的区别
- **vs `dynatemp-sampling`**：dynatemp 只有一个分布源，它用熵把归一化熵 \(C\) 插值进温度窗口，最后仍是一次 `softmax(logits/dyn)`——**只重标定温度，从不合成分布源**。本题 \(E\) 路并行、\(E\) 套温度与 \(E\) 个 top-\(p\)，被合成的是概率分布本身；不存在「一个全局温度」这个量。
- **vs `top-p-sampling` / `top-k-sampling` / `min-p-sampling`**：那些是**截断**算子（候选集只可能变小、剩余候选按同一条分布重归一）。本题的 top-\(p\) 只是每路的内部一步，整体上算子是**混合**：候选集是各路抽样的**并集**（既可能包含某路核外的 token，也可能凭空多出别的路的 token），末分布不是任何一条 \(q_e\) 的单调变形。
- **vs `adaptive-p-sampling` / `mirostat-v2-sampling`**：那两题是带状态的闭环控制器（EMA 目标 / 阈值 \(\mu\) 随步反馈），入参与返回都多一个状态。本题**无状态**，一次成型。
- **vs `penalty-sampling` / `logit-bias-sampling`**：它们都在 **logit 域**改造**单个**分布源（惩罚由历史计数导出 / 偏置由调用方静态给定）；本题在**概率域**合并**多个**分布源，全程不碰 logits 的加法。
- **vs `llama-sampling-pipeline`（同批在制）**：那题装配**一条**链里 bias→penalties→dry→top-k→top-p→min-p→xtc→temp 的**顺序**并逐步吐存活集合；本题是 **\(E\) 条链并联后合流**，只考「过采样取并集 + 在并集上加权混合 + 单次采样」这一段数学，不做管线装配。
- **vs MoE 类题（`moe-topk-gating` 等）**：那里的权重由路由网络算出、混合的是专家网络的**输出表示**（且通常 top-k 稀疏）；这里的权重是外部入参、混合的是**同一个位置上 \(E\) 条下一 token 分布**。名字像，层次不同。

## 对拍口径
参考解走张量路（`softmax` + `argsort(stable=True)` + `cumsum` 前缀扫描 + 逐专家掩码累加）。判题 `_oracle` 完全另起一路：纯 Python 列表，逐专家手算 `math.exp` softmax（用 `math.fsum` 求分母）、用 `sorted(range(v), key=lambda i: (-p[i], i))` 拿降序、**逐元素累加**找首次 \(\ge p_e\) 的前缀、逆变换用 `for i, x in enumerate(...)` 的累加比较，全程不做任何张量排序/索引。除样例与 25 组随机全量对拍（含 \(\tau\le0\) 贪心专家混入）外，另有不变量用例：\(|\mathcal U|\le E m\)、\(\mathcal U\) 外严格 \(0\)、`filtered` 和为 \(1\)、`chosen\in\mathcal U\)；以及三个退化断言：\(E=m=1\) ⇒ one-hot、\(w=0\) 专家只影响 \(\mathcal U\)、\(\tau=0\) 专家多枪同一位。统一 `allclose(atol=1e-5, rtol=1e-5)`。

## 复杂度
每条链的 softmax \(O(V)\)、top-\(p\) 排序 \(O(V\log V)\)，共 \(O(EV\log V)\)；过采样 \(E\cdot m\) 次逆变换各 \(O(V)\)；混合阶段对每个存活 token 累加 \(E\) 次，\(O(E\,|\mathcal U|)\le O(E^2 m)\)；末次采样 \(O(V)\)。空间 \(O(EV)\)。真实部署里 \(E\) 是「几路模型/几条链」，通常 \(2\sim4\)，故 \(E\) 的二次项可忽略，瓶颈仍是每路的 \(V\log V\) 排序。

## 实践提醒
过采样把「该给谁机会」这件事从**阈值**（留多少质量）换成了**抽样**（\(m\) 枪打到谁），代价是候选集依赖随机数、方差随 \(m\) 缩小，\(m=1\) 时混合结果近乎由各路各自的运气决定；调 \(m\) 与调 \(w\) 的效果并不等价——\(w_e=0\) 仍会改变候选集（见易错点 4），所以「关掉一路」在本题语义里是做不到的，只能把它的候选质量也压掉。权重不必归一（实现里显式除以 \(\sum w\)），但和必须为正。
