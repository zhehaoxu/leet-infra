# 详解：Epsilon Sampling 相对阈值截断

## 出处

- 算法出处：Hewitt, Manning, Liang, *Truncation Sampling as Language Model Desmoothing*，arXiv:**2210.15191**（已核实；Findings of EMNLP 2022），官方实验代码 `john-hewitt/truncation-sampling`。论文把截断采样解读为"去平滑"（desmoothing）：语言模型的 softmax 天然过平滑（长尾概率来自训练时的标签平滑效应），截断的意义是把这些概率质量**移走**——这也是本题 `renorm=False` 口径（保留项不动、质量归峰）的灵感来源。注意：论文原版的 \(\varepsilon\) 截断与 HF `EpsilonLogitsWarper` 的阈值口径（绝对阈值还是相对阈值）**待回源**，本题把口径钉成**相对阈值** \(\varepsilon\cdot p_{\max}\)——它让同一个 \(\varepsilon\) 在不同"温度/尖锐度"的分布下行为可比。
- 库侧实现：HuggingFace transformers 的 `EpsilonLogitsWarper`（`epsilon_cutoff`）；llama.cpp 无此算子（它走 min-p/xtc 一族）。本题的实现细节（取等方向、fill-the-hole 的并列裁决、逆变换兜底）按本合集采样题的公共约定执行。
- 一句话记忆：**epsilon 采样 = "绝对概率 + 相对锚" 的尾部剔除**。阈值不是固定数、也不是峰值的一个比例——是"峰值概率的 \(\varepsilon\) 倍"，兼得两者的可比性。

## 完整推导与下标约定

**为什么相对阈值**：绝对阈值 \(\varepsilon\) 在低温（尖锐）分布下几乎不砍、高温（平滑）分布下砍一大片——\(\varepsilon\) 的语义随分布漂移。锚到 \(p_{\max}\) 后，阈值与峰值**同步缩放**：\(p_{\max}\) 大（尖峰）阈值高、砍得多；\(p_{\max}\) 小（平坦，\(p_{\max}\to1/V\)）阈值低、砍得少。两个端点把行为钉死：\(\varepsilon=0\) 全保留（阈值 0、\(p\ge0\) 恒真）；\(\varepsilon=1\) 只剩 argmax（只有 \(p_{\max}\ge p_{\max}\) 的那一个）。

**取等方向**：判据 \(p_i\ge\text{thr}\)。写成 \(>\) 的实现当 \(p_{\max}\) 自身达标边界（\(\varepsilon=1\) 时 \(p_{\max}\ge p_{\max}\) 恰取等）会把自己也砍掉 → `filtered` 全零 → 归一化除零。取等保留让"argmax 恒存活"成为结构保证，分母恒正。

**两种质量处置的本质差别**：

- `renorm=True`（条件分布视角）：把截断当作"在保留集合上重新定义分布"——保留项等比放大，信息是"在这几个里选"。
- `renorm=False`（去平滑视角）：把截断当作"把模型输出里不该有的质量搬走"——保留项的概率**就是模型的真实信念**，不应被人工放大；被剔的质量有一个自然的去处：分布的最优点（argmax）。这保持了"其余 token 的相对序与绝对值不变"，只有峰值变高。论文口径下（待回源处）甚至可以不把质量补给任何 token（留下不满和为 1 的分布）；本题把去处钉成 argmax，使 `filtered` 恒为合法分布。

**并列裁决**：`renorm=False` 的窟窿补给 `argmax`；torch 的 `argmax` 并列时取**最小下标**——评测的 oracle 按同一裁决写，别自己发明"取最后一个"或"均分"。

**逆变换**：合集公共约定——原顺序 CDF、前缀和首次 \(\ge u\) 命中、兜底取最后下标。`renorm=False` 下 argmax 的概率被抬高，CDF 前段更陡，`u` 小时更容易命中峰——这正是去平滑的行为后果。

## 核心与易错点

1. **阈值锚在概率域的 \(p_{\max}\) 上**，不是 logit 域（那是 top-nσ）、也不是累计质量（那是 top-p）。三种"锚"混用是最常见的串题。
2. **`renorm=False` 下非 argmax 保留项逐位不动**：评测用 `atol=1e-7` 断言它等于原始 softmax 的对应分量。任何"顺手统一归一化"的实现都会让这三种分量全部偏移而挂掉。窟窿只流向 argmax（`scatter_add` 单点）。
3. **\(\varepsilon=1\) 的 one-hot**：只有 argmax 存活；`renorm=False` 口径下它拿到 \(p_{\max}+(1-p_{\max})=1\)，`renorm=True` 口径下 \(1/1=1\)——两种口径殊途同归。若取等方向写反，这里除零。
4. **\(\varepsilon=0\) 的恒等**：阈值 0，\(p_i\ge0\) 全真；`renorm=False` 的窟窿为 0、补峰是无害空操作。用原始 softmax 直接断言（`atol=1e-6`）。
5. **argmax 用原始概率判定**（`probs.argmax`），不是 `filtered.argmax`——两者在取等边界上可能不同（填充后峰值变大不改变 argmax，但实现上先算 argmax 再填更安全）。
6. **批维广播**：`p_max` 逐行、阈值逐行、argmax 逐行、CDF 逐行；`u` 的形状与批维匹配（标量 logits 配标量 `u`）。
7. **float32 细节**：`keep` 转 dtype 后相乘；`scatter_add_` 的 index 形状 (batch, 1)；兜底下标只在"浮点前缀和差点到不了 \(u\)"时用。
8. **不许随机**：`u` 是唯一随机源；内部任何 `torch.rand` 都会让对拍失败。

## 与邻近题的区别

- **vs `min-p-sampling`**：同样是"相对阈值 + 峰值锚"，方向与锚的性质不同——min-p 阈值 = \(p_{\max}\cdot\alpha\) 砍**尾部**、阈值随峰值**比例**缩放；epsilon 的 \(\varepsilon\) 也是峰值比例，但 \(\varepsilon\) 语义是"概率本身的下限"而非"与峰的比值上限"。数值上 min-p 保留 \(p_i\ge\alpha p_{\max}\)、epsilon 保留 \(p_i\ge\varepsilon p_{\max}\)——**判据竟然相同**！真正的差别在质量处置与定位：min-p 出自 llama.cpp（只 renorm）、epsilon 出自去平滑论文（renorm 可关、质量归峰）；min-p 的 \(\alpha\) 惯用小值（0.05–0.2）、epsilon 的 \(\varepsilon\) 直接在概率量纲上取值。两题的示例数值矩阵不同，可从"非 argmax 保留项是否逐位不动"一锤定音。
- **vs `top-p-sampling`**：top-p 的截断点由**累计**决定（头部再多也只留到 p 为止），epsilon 由**单点概率**决定（尾部全砍、头部全留）。尖峰分布下 top-p 早就塌到 1–2 个，epsilon 仍保留整批高概率项——两者对"头部"的态度相反。
- **vs `eta-sampling`（论文同门的熵版）**：eta 用熵 \(H\) 自适应阈值 \(\min(\varepsilon,\sqrt{\varepsilon}\,e^{-H})\)，本题的阈值与熵无关——纯统计、无熵项。
- **vs `dripping-sampling`**：方向相反的统计截断——dripping 砍**头部**（高于均值/加 σ），epsilon 砍**尾部**（低于峰值比例）。一题的存活集合恰好是另一题想压缩的对象。

## 对拍口径与复杂度

- 主口径：独立重算 `_oracle`——纯 Python 循环：逐项 exp 的 softmax、手算阈值、显式 \(\ge\) 筛选、两种质量处置（renorm 用除法、fill-the-hole 用单点累加）、while 前缀和逆变换。与参考解（torch 向量化 + scatter_add）在 40 组随机批 logits × {True,False} × {0.05,0.2,0.5,0.9} 全网格上对拍：filtered `allclose(1e-5)`、chosen `torch.equal`。
- 专项断言：\(\varepsilon\in\{0,1\}\) 两个端点（恒等 / one-hot）；`renorm=False` 的"非 argmax 分量不动"（`atol=1e-7`）；`renorm=False` 下 `filtered.sum()==1`（质量守恒）。
- 复杂度：每行 \(O(V)\)（softmax、阈值、筛选、归一化/补峰、CDF 各一遍），无排序；显存与输入同阶。
