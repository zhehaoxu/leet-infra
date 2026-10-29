# 详解：检索：IVF-PQ 非对称 ADC 打分

## 出处

- 方法源头：Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545。PQ 论文同时给出对称距离计算（SDC，查询也编码）与非对称距离计算（ADC，查询保持浮点、只解码库中码字）；本题实现的是 ADC。
- 工程对标：Faiss 官方文档中的 `IndexIVFPQ` / `IndexIVF::search`（粗量化 + 倒排链 + `nprobe`）与 PQ 码本的 LUT 打分方式。Faiss 文档没有规定本题的补齐、并列与 dtype 细节，那些以 starter docstring 为准。
- CPU、纯确定性：选 cell 只做距离排序，打分只做查表与求和，不调用随机数。

## 完整推导与下标约定

**粗选择。** 记粗中心矩阵 \(C\in\mathbb R^{C\times D}\)，查询 \(q\in\mathbb R^D\)。对每个 cell \(j\) 计算
\[
s_j=\|q-c_j\|_2^2=\sum_{t=0}^{D-1}(q_t-c_{j,t})^2 .
\]
把 \(j\) 按 \((s_j,\ j)\) 升序排，取前 \(n_{\text{probe}}\) 个。参考实现用 `torch.argsort(..., stable=True)`：稳定排序保留相等元素的原始下标顺序，而原始下标就是 cell 号，因此并列自然取小 cell 号。实测 `torch.argsort(torch.tensor([2.,1.,1.,0.,1.]), stable=True)` 返回 `[3, 1, 2, 4, 0]`——三个 1 保持了下标 1、2、4 的相对顺序。

**LUT。** 第 \(j\) 个子空间覆盖特征区间 \([j d,(j+1)d)\)，该子空间的第 \(t\) 个码字是 \(\text{codebooks}[j,t]\in\mathbb R^d\)。查询不量化，直接切段：
\[
\text{LUT}[j,t]=\bigl\|q_{j d:(j+1)d}-\text{codebooks}[j,t]\bigr\|_2^2
=\sum_{u=0}^{d-1}\bigl(q_{j d+u}-\text{codebooks}[j,t,u]\bigr)^2 .
\]
LUT 形状为 \((m,K)\)，每个查询一张。

**ADC 距离。** 文档 \(i\) 的码字行是 \(z_i=(z_{i,0},\dots,z_{i,m-1})\)，\(0\le z_{i,j}<K\)。PQ 的近似向量是各子空间码字的拼接，故
\[
\hat d(q,x_i)=\sum_{j=0}^{m-1}\bigl\|q_{j d:(j+1)d}-\text{codebooks}[j,z_{i,j}]\bigr\|_2^2
=\sum_{j=0}^{m-1}\text{LUT}[j,z_{i,j}].
\]
注意这是 **\(m\) 项之和**，不是平均，也不是把 \(m\) 个码字拼成完整向量后再算一次欧氏距离（两者在数学上相等，但后者要多做一次 \(D\) 维减法，LUT 的意义就是把这步摊到每个子空间一次）。

**排序。** 候选键是二元组 \((\hat d,\ \text{id})\)。参考实现先按 id 稳定排序、再按距离稳定排序：第二趟排序时相等距离保持第一趟的 id 升序，于是得到字典序。torch 没有直接的 lexsort，双稳定排序是最短路径。`ids` 全局唯一时这等价于「距离升序、id 升序」。

**补齐。** 设 \(N_q\) 为该查询在所选 cell 中的候选总数，真实槽位数是 \(t=\min(k,N_q)\)。`ids` 固定 \((Q,k)\)，前 \(t\) 格写真实 id，其余写 \(-1\)；`dists` 前 \(t\) 格写真实距离，其余写 \(+\infty\)。`k=0` 直接返回 \((Q,0)\)。这样形状不随候选数变化，空候选也能用统一口径表达。

**dtype。** 输入只接受 float32/float64，三者必须同 dtype。内部把查询、中心、码本都 cast 到 float64 做平方差与求和，最后把距离 cast 回查询 dtype；`ids` 恒 int64。

**复杂度。** 选 cell：\(O(QCD)\)；建 LUT：\(O(QmKd)=O(QKD)\)；扫候选：每个候选 \(O(m)\) 次查表加法，共 \(O(QNm)\)。空间上 LUT 是 \(O(QmK)\)，候选拼接是 \(O(Nm)\)。

## 核心与易错点

1. **子空间码本不能串用**：第 \(j\) 段查询只能查 `LUT[j]`，而 `LUT[j]` 只能由 `codebooks[j]` 生成。把 \(m\) 套码本当成一套大码本、或全用第 0 套，是最常见的形状对但数值错的 bug。
2. **ADC 是“查询不量化”**：不要把查询也编码成码字再算码字间距离——那是 SDC。本题查询保持浮点，LUT 用原始查询切段算。
3. **距离累加用 float64**：实测在一个 \(D=32,m=4,K=16,C=8\)、坐标放大 10 倍的随机用例上（距离范围 2370.5864–11989.4717），纯 float32 的 LUT + 求和与 float64 累加再 cast 的结果最大绝对差为 \(9.76562\times10^{-4}\)，均值 \(1.91154\times10^{-4}\)。这已经超过 float32 判题常用的 \(10^{-5}\) 绝对容差，所以参考实现坚持 float64 累加。
4. **nprobe 并列取小 cell 号**：靠稳定排序实现。测试里把 4 个中心都放在到 query 距离为 1 的位置，只在 cell 2/3 放候选；`nprobe=2` 必须选 cell 0/1 从而返回全填充，`nprobe=4` 才返回 `[20,30]`。若并列时错选大号 cell，这个用例会立刻失败。
5. **top-k 并列取小 id**：示例中 id 1 与 id 2 的 ADC 距离都恰为 100，输出必须是 `[0,1,2]` 而不是 `[0,2,1]`。双稳定排序保证这一点。
6. **padding 不是省略**：候选不足 \(k\) 时仍返回 \((Q,k)\)，尾部 `-1` / `+inf`；`k=0` 是 \((Q,0)\) 而不是报错。
7. **空 cell 要跳过**：`n_c=0` 的 cell 可以出现在倒排表里，`torch.cat` 前必须过滤，否则会拼接出空片段（本身无害，但若把它当成一个候选会破坏计数）。
8. **nprobe 边界**：`nprobe=0` 不探测；`nprobe>C` 等价于 \(C\)。实测示例 `nprobe=5` 与 `nprobe=2` 输出完全相同。
9. **全局 id 唯一是契约**：本题不去重；同一 id 多次出现时排序会出现重复 id，行为未定义。测试数据不会这样构造。
10. **C=0**：没有中心也没有倒排表，`argsort` 在长度 0 的维度上返回空，结果是全填充；不要因为 `min(nprobe,C)=0` 就走特殊分支报错。

## 与邻近题的区别

- **vs `retrieval-ivf-assign`**：那题只做「每个向量分配给最近粗中心」，输出每个向量的中心下标；本题先为**查询**选 \(n_{\text{probe}}\) 个 cell，再在链内用 PQ 码字做 ADC 打分，核心是 LUT 与 top-k。
- **vs `retrieval-pq-codes`**：那题是**编码**（给定原始向量，逐子空间 argmin 得到码字，输出 \((n,m)\) 整数）；本题码字是输入，只做**解码打分**，不训练也不编码。
- **vs `retrieval-hnsw-greedy`**：那题在图上逐跳贪心，候选来自邻居表；本题是倒排链的穷举扫描，没有图结构，也不存在局部最优。
- **vs `retrieval-nprobe-merge`（同批次）**：那题倒排表存完整向量，考点是探测顺序、跨 cell 去重取最小距离与候选数统计，不做 PQ；本题不做去重、不输出中间 cell 序列，考点是 LUT 的逐子空间求和与 ADC top-k。两题的 `nprobe` 选择规则相同，但下游打分机制完全不同。
- **vs `retrieval-kmeans-lloyd`**：那题训练粗中心；本题中心与码本都是给定输入。

## 对拍口径与复杂度

- **参考解结构**：float64 下广播算 LUT \((Q,m,K)\)，对每个查询按选中的 cell 拼接 id 与码字，用 `lut[qi][arange(m), codes]` 做二维 gather 后沿 \(m\) 求和，再双稳定排序、写前 \(t\) 格。
- **独立 oracle**：`_oracle` 用纯 Python 五重循环（查询 × cell × 文档 × 子空间 × 维）逐项累加平方差，显式 `sorted(range(C), key=(dist, cell))` 选 cell，`list.sort(key=(dist, id))` 排序；与参考解的「LUT gather + 双稳定排序」结构不同，专门抓子空间错位、LUT 索引错误与并列顺序错误。
- **用例**：示例（实测回填 `EXAMPLE_*`，含 nprobe=1 的填充支）；5 组随机 float32 × `nprobe∈{0,1,2,4,9}` × `k∈{0,1,3,10}`；3 组随机 float64 × `nprobe∈{0,1,3,4,7}` × `k∈{0,2,6}`；top-k 截断与 k 超大填充；四码字等距的 id 并列；nprobe 并列的 cell 选择；k=0、nprobe=0、全空倒排表、C=0、Q=0；10 组非法输入。
- **容差**：索引 `torch.equal`；float32 距离 `atol=1e-5, rtol=1e-6`；float64 `atol=1e-9, rtol=1e-12`；填充位要求恰为 `inf`。
- **实测记录**（`.venv-eval/bin/python` 内联脚本，参考实现直接调用）：
  - 示例 `nprobe=2` → `ids [[0,1,2]]`、`dists [[0.,100.,100.]]`；`nprobe=1` → `ids [[0,1,-1]]`、`dists [[0.,100.,inf]]`；`nprobe=0` → 全 `-1`/`inf`；`nprobe=5` 与 `nprobe=2` 相同。
  - `k=0` → `ids`/`dists` 形状 `(1,0)`；全空倒排表 → 全填充。
  - float32 累加 vs float64 累加的最大绝对差 \(9.76562\times10^{-4}\)（距离范围 2370.5864–11989.4717）。
  - `torch.argsort(torch.tensor([2.,1.,1.,0.,1.]), stable=True)` → `[3,1,2,4,0]`。
- **复杂度**：\(O(QCD+QKD+QNm)\) 时间；\(O(QmK+Nm)\) 临时空间。
