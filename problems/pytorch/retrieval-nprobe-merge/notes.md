# 详解：检索：nprobe 多探测与跨 cell 归并

## 出处

- 结构出处：Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545 中的 coarse quantizer + inverted list；Faiss 官方文档的 `IndexIVF` 与 `nprobe` 参数（`IndexIVF::search` 按粗中心距离选前 `nprobe` 条链）。Faiss 的多探测是「先粗排 cell、再扫描选中链」，本题把这套流程里的顺序、去重与统计单独抽出来。
- CPU、纯确定性：选 cell 只排序距离，归并只做 dict/sort 与最小值，无随机数。

## 完整推导与下标约定

**cell 打分与顺序。** 查询 \(q\)、粗中心 \(c_j\)：
\[
s_j=\|q-c_j\|_2^2,\qquad
\pi=\operatorname{argsort}\bigl(s_0,\dots,s_{C-1}\bigr)\ \text{（稳定，即并列取小 cell 号）}.
\]
前 \(P=\min(n_{\text{probe}},C)\) 个就是探测序列，按 \(\pi_0,\pi_1,\dots,\pi_{P-1}\) 依次扫描。注意这里必须**稳定**：若两个 cell 到 query 等距，原始下标顺序就是 cell 号顺序，稳定排序保证小号在前。实测中心 \(c=[(0,2),(1,0),(-1,0),(0,1)]\)、\(q=(0,0)\) 的平方距离是 \(4,1,1,1\)，`nprobe=4` 的序列为 \([1,2,3,0]\)，`nprobe=2` 为 \([1,2]\)——既不是 0 开头，也不是任意顺序。

**单条 posting 的距离。** cell \(c\) 的第 \(i\) 条 posting 存完整向量 \(v_{c,i}\)，其距离为
\[
d_{c,i}=\|q-v_{c,i}\|_2^2=\sum_{t=0}^{D-1}(q_t-v_{c,i,t})^2 .
\]
本题不做残差编码，也不做 PQ 近似；向量就是文档本身。

**并集去重取最小。** 记被探测 posting 的集合为
\[
\mathcal P=\{(c,i): c\in\pi_{0:P}\}.
\]
对每个全局 id \(g\)，它可能对应多条 posting，定义
\[
D(g)=\min\{d_{c,i}:\ \text{id}(c,i)=g\}.
\]
输出的是 \((g,D(g))\)，每个 id 一次。等价实现：把所有 posting 按 \((\text{id},\text{dist})\) 字典序排序，每个 id 的第一行就是最小距离。参考实现用两趟稳定排序（先距离、再 id）实现这一点：同一 id 内距离已经升序，首行即最小。纯 Python oracle 则用 `dict` 显式做 `min`。

**排序与截断。** 去重后的候选按 \((D(g),g)\) 升序排，取前 \(k\)。参考实现中，去重后的 id 已经升序，再对距离做一次稳定排序即可让并列距离保持 id 升序。真实槽位数 \(t=\min(k,\#\{g\})\)，尾部 `ids=-1`、`dists=+inf`。

**统计恒等式。** 设 \(N_{\text{post}}\) 为扫描的 posting 总数、\(N_{\text{uniq}}\) 为不同 id 数，则
\[
N_{\text{dup}}=N_{\text{post}}-N_{\text{uniq}},\qquad
N_{\text{ret}}=\min(k,N_{\text{uniq}}).
\]
`n_duplicates` 计的是「被删掉的重复出现次数」，不是「有重复的 id 个数」；`n_returned` 与 `n_duplicates` 无关，只由 \(k\) 与唯一候选数决定。

**dtype 与补齐。** 查询、中心、倒排向量必须同为 float32/float64；内部 cast 到 float64 累加平方差，输出距离 cast 回查询 dtype。`cells`、`stats`、`ids` 全程 int64。

**复杂度。** 选 cell：\(O(QCD)\)；扫描：每个 posting \(O(D)\)，共 \(O(QND)\)；去重排序：\(O(N\log N)\)。空间 \(O(QC+N)\)（不含输入）。

## 核心与易错点

1. **探测顺序不是 cell 号顺序**：示例里 \(q\) 到 cell 1/2 的距离是 9/1，`nprobe=2` 的序列是 `[0,2]`，cell 1 被跳过。测试专门构造了中心距离为 \(4,1,1,1\) 的用例，`nprobe=2` 必须得到 `[1,2]`。
2. **并列取小 cell 号**：靠稳定排序实现；`torch.argsort(..., stable=True)` 保留相等键的原始下标顺序。实测 `argsort([2,1,1,0,1], stable=True)` 返回 `[3,1,2,4,0]`。
3. **去重必须取最小，而不是第一次或最后一次出现**：示例中 id 3 先在 cell 0 以距离 0 出现、后在 cell 2 以距离 4 出现，第一次恰好也是最小；但测试另有一组同一 cell 内 id 1 的距离依次为 9、0、1，正确答案是 0——取第一次得 9、取最后一次得 1，都会失败。
4. **统计列的含义**：`n_postings` 含重复，`n_unique` 是不同 id 数，`n_duplicates` 是两者之差，`n_returned=min(k,n_unique)`。示例实测 `[4,3,1,3]`；`nprobe=1` 时 `[2,2,0,2]`；`nprobe=5`（clamp 到 \(C=3\)）时 cell 1 的 id 8 也被扫到，得到 `[5,4,1,3]`。
5. **nprobe 边界**：`nprobe=0` 时 `cells` 形状 \((Q,0)\)、四列统计全 0；`nprobe>C` 视为 \(C\)，`cells` 是全部 cell 的完整顺序；`nprobe=1` 只取最近且小号的 cell。三种情况都实测过。
6. **k=0 不改变中间量**：`ids`/`dists` 形状 \((Q,0)\)，但 `cells` 仍是 `[[0,2]]`（示例），`stats` 仍是 `[4,3,1,0]`——扫描与去重照做，只是不返回候选。
7. **浮点累加用 float64**：实测在 \(D=32\)、5 个 cell 各 150 条、坐标放大 10 倍的随机用例上（距离范围 2574.5815–11984.5059），纯 float32 距离与 float64 累加再 cast 的最大绝对差为 \(9.76562\times10^{-4}\)，均值 \(2.13542\times10^{-4}\)，超过 \(10^{-5}\) 的 float32 判题容差。
8. **padding 与空候选**：候选不足 \(k\) 时尾部 `-1`/`+inf`；全空倒排表仍是 \((Q,k)\) 的填充结果，不是空张量。
9. **C=0**：没有中心、没有链，`cells` 形状 \((Q,0)\)、`stats` 全 0、结果为全填充；不要因除零或 `argsort` 空维度报错。
10. **id 允许重复但必须非负**：重复是本题要处理的核心；负 id 与哨兵 `-1` 冲突，直接 `ValueError`。

## 与邻近题的区别

- **vs `retrieval-ivfpq-adc`（同批次）**：那题倒排表存 PQ 码字，核心是逐子空间 LUT 求和（ADC），不做去重、不输出中间 cell 序列；本题存完整向量，核心是探测顺序、跨 cell 去重取最小距离和候选数统计，距离本身是普通平方欧氏。
- **vs `retrieval-ivf-assign`**：那题只把每个向量分配给最近粗中心并返回下标；本题是**查询侧**选 \(n_{\text{probe}}\) 个 cell，再归并链内 posting，多了顺序、去重与统计。
- **vs `retrieval-hnsw-greedy`**：那题在图上逐跳贪心、可能卡局部最优；本题是倒排链的确定性扫描，扫完所有选中链，没有图与早停。
- **vs `retrieval-rrf-fusion`**：那题融合多路**名次**，不做距离、不做 cell；本题融合的是同一个查询在多个倒排链里的**距离**，并显式去重。
- **vs `retrieval-kmeans-lloyd`**：那题训练粗中心；本题中心是给定输入。

## 对拍口径与复杂度

- **参考解结构**：float64 下稳定排序选 cell；逐 cell 广播算距离；`torch.cat` 后两趟稳定排序（先距离、再 id）取每个 id 首行完成去重，再对唯一候选按距离稳定排序；统计用张量尺寸直接算。
- **独立 oracle**：`_oracle` 用纯 Python 逐 cell、逐 posting 算距离，`dict` 做 `min` 去重，`sorted(best.items(), key=(dist,id))` 排序；与参考解的「拼接 + 双稳定排序 + 布尔掩码」结构不同，专门抓探测顺序、去重方向和统计口径。
- **用例**：示例（实测 `EXAMPLE_*`，含 nprobe=1 支）；5 组随机 float32 × `nprobe∈{0,1,2,4,9}` × `k∈{0,1,3,10}`；3 组随机 float64 × `nprobe∈{0,1,3,4,8}` × `k∈{0,2,6}`；随机数据用有放回抽 id 制造跨 cell / 同 cell 重复；去重取最小（含「第一次/最后一次都不是最小」的构造）；cell 并列顺序；nprobe 边界；top-k 截断与超大 k；全空、k=0、C=0、Q=0；9 组非法输入。
- **容差**：`ids`/`cells`/`stats` 用 `torch.equal`；float32 距离 `atol=1e-5, rtol=1e-6`；float64 `atol=1e-9, rtol=1e-12`。
- **实测记录**（`.venv-eval/bin/python` 内联脚本，直接调用参考实现）：
  - 示例 `nprobe=2` → `ids [[3,7,5]]`、`dists [[0.,0.,1.]]`、`cells [[0,2]]`、`stats [[4,3,1,3]]`；`nprobe=1` → `ids [[3,7,-1]]`、`dists [[0.,0.,inf]]`、`cells [[0]]`、`stats [[2,2,0,2]]`。
  - 示例 `nprobe=5`（\(C=3\)）→ `cells [[0,2,1]]`、`stats [[5,4,1,3]]`；`nprobe=0` → `cells` 形状 `(1,0)`、`stats [[0,0,0,0]]`。
  - 并列中心距离 \(4,1,1,1\)：`nprobe=4` → `cells [[1,2,3,0]]`；`nprobe=2` → `[[1,2]]`。
  - 同 cell 重复 id 1、距离 9/0/1 → `ids [[1,-1]]`、`stats [[3,1,2,1]]`。
  - float32 vs float64 累加最大绝对差 \(9.76562\times10^{-4}\)（距离范围 2574.5815–11984.5059）。
- **复杂度**：\(O(QCD+QND+N\log N)\) 时间；\(O(QC+N)\) 额外空间。
