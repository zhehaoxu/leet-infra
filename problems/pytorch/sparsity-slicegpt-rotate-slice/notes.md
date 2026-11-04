# 解题笔记：SliceGPT 正交旋转-切边步

## 回源定位

- 源文件：microsoft/TransformerCompression（仓结构 src/slicegpt/，旧路径 slicegpt/ 404）`rotate.py` 528 行。与 sparsity 现有 16 题完全不同族（旋转降秩 vs 置零/重构/调度），无重叠。
- `pca_calc`（L501-528）：`H_batch = Σ_batch x.mT @ x`（每 batch 先掩码置零）→ damp=0.01·mean(diag) 只加对角 → eigh 升序 → argsort 降序同时重排 vals 与列向量。
- `rotate_and_slice_sequential`（L140-251）：四处旋转（attn 输入右乘 Q、attn 输出/MLP 输出左乘 Q.T、MLP 输入右乘 Q）+ 四处切边（列 3 处、行 2 处）+ 双 shortcut 链；`slice_attention_inputs` 切 shortcut 的**行**（L32），`get_mlp_input_dimension ≡ get_attention_output_dimension`（ForwardSlicingScheduler L194-196）。
- dtype 纪律：权重旋转 float64 后转回；shortcut 链 matmul 前 `Q.to(dtype)`（float32）——float32 输出粒度下 float64/float32 matmul 差异 < 2 ULP，判题容差不可分，dtype 边界变异不入清单（口径仍要求 float64）。
- 官方 pca 就地掩码（inps 被改写）；判题域纯函数式改为 clone。

## 数学要点

1. **旋转-切边 = 降秩**：W' = (W Q)[:, :k] 等价于在主成分坐标系里丢弃低能量坐标；重构性质 W' Q[:k].T ≈ W 只在"切前 k 主成分"时近似成立（信息损失集中在尾部特征值）。测试用 Q^T Q = I 与 H q = q·diag(vals) 的特征对性质做符号无关验证。
2. **damping 的双重作用**：防 eigh 数值退化；且它**只平移谱不换特征向量**——所以 eig_vals 的断言（vals = vals_nodamp + damp）既能钉住 damp 公式又能区分 damp-all（H+damp·11^T 换向量）与 damp-drop（谱不平移）两类变异。
3. **谱非退化是测试前提**：eigh 在退化特征值张成的子空间内任意取向都合法。`_sig` 的均匀纹波经掩码置零后会出现 ~5e-12 的近退化对（实测），换掩码乘子 0.37 后 min gap 1.8e-2——**掩码模式与数据模式要联合扫描**。
4. **H 的 batch 累积与拼接等价**：Σ_b Σ_token x xᵀ 与 cat(batch) 后单次累积只差浮点求和顺序（~1e-13），测试用 allclose 1e-7 验证。
5. **shortcut 链是"跨块基变换"**：attn_shortcut 把注意力输出坐标映到下一子块的切片坐标；两段 matmul 的 float32 cast 顺序（cast→slice == slice→cast）等价，但**先 slice 后 matmul 与先 matmul 后 slice 不等价**（矩阵乘了对角 0/1 的列选择——列选择与乘法可交换，其实也等价；真正不等价的是切错轴/不切）。

## 实现要点

- 校验顺序：ignore_masks 是 list 或 None → 三组信号（非空 list、3 维、隐维一致、掩码对齐 + 0/1）→ 三组 batch 数一致 → w_qkv list 非空且形状 (d,d) → w_o/w1/w2 形状 → o_bias → 三个切边维度（bool 排除、[1,d]）。
- 单元素 list 的 (B,S,3) 张量能过 3 维/隐维检查但会先触发 batch 数校验——**校验顺序决定测试用例的构造方式**（非 3 维用例必须用 2 维张量构造）。
- PCA 返回 (vals_desc, vecs_desc)；三组各调一次，输出把 q 与 vals 一并返回供性质断言。

## 测试要点

- **oracle 分层**：H 构造独立 numpy 复算（einsum 逐 token 外积 + damping），eigh 作共享原语（torch 与 numpy 的 LAPACK 符号约定可能不同，直接对拍 Q 矩阵不安全），记账管线镜像——突变 H 的任何一步都会传播到权重。
- 性质断言（符号无关）：Q^T Q=I、H q = q·diag(vals)、vals 降序且相邻 gap > 1e-3（谱非退化断言本身也是护栏）。
- damping 断言：vals == vals_nodamp + damp（对 _np_h 的 damping=False 变体）。
- 掩码语义：掩码运行 == 手工置零后无掩码运行（乘掩码等价置零）；掩码 vs 无掩码结果不同。
- batch 累积：两 batch 与 cat 成单 batch 的谱/向量一致。
- 切边维度 = d 的可逆性：W' Q.T ≈ W（重构）；同一信号三组 → q_in == q_mid == q_out。
- purity（掩码置零路径必须 clone）、bias None 路径、校验 battery（空 list/2 维/隐维不一致/batch 数/掩码对齐与 0-1/形状/dim bool 与越界）。

## 变异清单（44 条，全部 CAUGHT，无等价）

PCA 11：damp 丢弃/系数/加满/mean→min、掩码反转/丢弃/就地修改、x.mT 交换、batch 累积取末批、argsort 升序、向量不重排。旋转/切边 11：qkv 侧向/不切/切行、wo 侧向/不切、bias 不切/不旋转、w1 侧向/不切、w2 侧向/不切。shortcut 10：attn 不转置/不切行/不切 Q_mid 列/换 Q_out、mlp 不切行/换 Q_in/不乘 Q_out/不切列/末尾转置。校验 12：信号空 list/维度/隐维一致/batch 数/掩码 list/对齐/0-1、w_qkv 空/形状、w1w2 维度、o_bias、dim bool、消息。

经验：① **掩码置零与数据生成要联合设计**——掩码后谱可能退化（5e-12 gap），eigh 子空间内任意合法，oracle 对拍会随机失败；扫参数找 gap>1e-3 的组合并在断言里显式检查 gap（自守护栏）；② **共享原语（eigh）+ 独立构造（H）**的分层 oracle 是特征分解类题的判题解法——torch/numpy LAPACK 符号约定不同, 全独立 oracle 不可行, 但 H 构造/记账管线的独立性足以让全部机制变异现形；③ 校验顺序决定用例构造（单元素自洽 list 会跳过隐维检查直达 batch 数校验）；④ float32 输出粒度下 dtype 边界变异（旋转用 float32 matmul）差异 < 2 ULP 不可观测，不入变异清单、口径文字约束。
