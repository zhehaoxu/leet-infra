# torchaudio 高斯 Fréchet 距离账：五项形状校验与链式比较 quirk、mu 平方项与双迹之和、eigvals(Σx@Σy) 复主值平方根实部求和、F = a + b - 2c 装配

## 1. 背景

`torchaudio.functional.frechet_distance`（functional.py 2476-2511 行）是 FID（Fréchet Inception Distance）的内核：给定两个多元高斯的均值 μx/μy 与协方差 Σx/Σy，返回 `||μx-μy||² + Tr(Σx+Σy-2 sqrt(ΣxΣy))`。官方只有 35 行、四个无默认值的参数，却把「矩阵平方根」这一最容易实现错的步骤写成 `torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()`——对**复特征值先取主值平方根、再把实部相加**，而不是对称化矩阵平方根，也不是先截断实部。
本题把这条账用纯 Python 复刻并拆成七个可独立判题的 mode。全库 `frechet` / `FID` / `eigvals` / `sqrtm` 在 meta、description、notes 中零命中；855 / 857-862 已覆盖同一文件的频谱后处理、波形算子、NCCF、sinc 重采样、Griffin-Lim、响度门控与 MVDR 波束形成，函数集合不重叠。**回源纠错**：`todos.md` 曾把本函数记作「纯 DP / 离散 Fréchet」，实际它引用 Dowson & Landau (1982) 的高斯距离公式，与曲线 Fréchet、Levenshtein 编辑距离无关。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  frechet_distance            (line 2476, 35 lines, no defaults)
    guard 1..5                 shape / dimension validation
    a = (mu_x - mu_y).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
    return a + b - 2 * c
```

### 逐字执行环境

`official863.py` 用 `ast.get_source_segment` 从落盘的 2511 行官方源码中抽出 `frechet_distance` 的定义并 `exec` 进共享命名空间，官方函数体一字未改；所有张量固定 float64，因此 `eigvals` 返回 complex128。`ledger()` 逐行重算 a / b / A / eigvals / sqrt / c / F，用于探针与 fixtures；`sorted_eigvals()` 按 `(re, im)` 排序，规避 LAPACK 特征根顺序的实现定义。

## 2. 口径约定

### 输入与输出

- 四个必需参数：`mu_x`、`sigma_x`、`mu_y`、`sigma_y`；μ 是长度 N 的一维列表，Σ 是 N×N 二维列表，元素为 int / float / bool（bool 按 1 / 0 转换）。
- 判题域 N≤4，协方差与均值全部显式入参；不跑模型、不做音频 IO、无全局 RNG、无 scipy / librosa / soundfile / numpy / torch 依赖。
- 复特征值以 `[re, im]` 二元组表示；`eigvals` 返回按 `(re, im)` 字典序排序后的列表，实根虚部精确为 0.0。
- `charpoly` 返回 `det(lambda*I - A)` 的 monic 降幂系数 `[1, c1, ..., cN]`，其中 `A = SX @ SY`。
- 七个 mode：`frechet`（完整 F）、`mu_term`（a）、`trace_term`（b）、`sqrt_trace_term`（c）、`terms`（`[a,b,c]`）、`charpoly`、`eigvals`。

### 非法输入

- 五条形状校验按官方顺序抛出：`Input mu_x must be one-dimensional`、`Input sigma_x must be two-dimensional`、`Each of sigma_x's dimensions must match mu_x's size.`、`Inputs mu_x and mu_y must have the same shape`、`Inputs sigma_x and sigma_y must have the same shape`。
- 非矩形、非数值叶子、非列表入参、缺参数、未知 mode 各自显式报错；未知 mode 的错误消息必须列出全部七个 MODES。

## 3. mode 1：frechet（完整距离）

返回 `a + b - 2*c`，是唯一直接对拍官方函数返回值的 mode。主账样例（N=3，μx=[1,-2,0.5]、μy=[0,1,-0.5]）给 a=11.0、b=7.5、c=3.647581187475623、F=11.204837625048754。交换 `(μx,Σx)` 与 `(μy,Σy)` 结果不变：`A=ΣxΣy` 与 `ΣyΣx` 有相同谱，a 与 b 也对称。

## 4. mode 2-4：三项分项账

`mu_term` 是 `sum((x-y)**2)`，与 Σ 完全无关；`trace_term` 是 `tr(Σx)+tr(Σy)`，只看对角，改非对角元素不改变 b 却会通过 A 改变 c 与 F；`sqrt_trace_term` 是 `sum(cmath.sqrt(lambda_i).real)`，是本题最核心也最易写错的一项。

## 5. mode 5：terms（三元组）

返回 `[a, b, c]`，顺序固定。三个分项与 `frechet` 必须满足装配恒等式；该 mode 让判题可以分别钉死三项，而不必只靠一个最终标量。

## 6. mode 6：charpoly（特征多项式）

参考解用 Newton 恒等式从幂迹 `tr(A^k)` 递推初等对称多项式 `e_k`，再按 `coeff[k] = (-1)^k e_k` 组装 monic 降幂系数；`N=1/2/3` 用一次式、二次公式与 Cardano 闭式求根，`N>=4` 回退 Durand-Kerner 迭代。独立 oracle 走另一条路：Faddeev-LeVerrier 矩阵递推 `M_k = A M_{k-1} + c_{k-1} I`，与参考解无共享代码。

## 7. mode 7：eigvals（复谱）

返回 `A=Σx@Σy` 的全部特征值（按 `(re, im)` 排序）。LAPACK 的返回顺序是实现定义的，所以判题域比较排序后的谱；因为 `c` 只依赖谱的无序求和，排序不影响数值。实系数多项式的复根成共轭对出现，`AB` 与 `BA` 同谱。

## 8. 复特征值的主值平方根

官方 `.sqrt().real.sum()` 的顺序不可调换。以 Σx=[[1,2],[-2,1]]、Σy=I 为例，`A` 的谱是 `1±2j`；主值平方根实部为 `sqrt((|z|+Re z)/2) = sqrt((sqrt(5)+1)/2) ≈ 1.272`，故 c≈2.544、F≈-1.088。若先 clamp 实部再开方会得 c=2.0，是错的。负实根 -4 / -1 的主值平方根是纯虚数 `2j` / `1j`，实部为 0，故 c=0、F=-3。官方不校验正定性，F 可以为负。

## 9. 链式比较 quirk 与形状校验

第三条校验 `sigma_x.size(0) != sigma_x.size(1) != mu_x.size(0)` 是 Python 链式比较，等价于 `(s0 != s1) AND (s1 != n)`，**不是**三重互异。方阵 Σ 的第一段恒为 False，整条校验短路：3×3 的 Σ 配长度 2 的 μ 会被静默接受，因为 μ 只参与 `μx-μy`，从不与 Σ 做矩阵运算。只有非方阵且 `s1 != n` 才触发该报错；非方阵但 `s1 == n` 同样短路，最终在下游方阵检查处报错。参考解用逐字的链式写法复刻这个行为。

## 10. 官方 quirk 清单

- `Σx=Σy=Σ` 时 `F=||Δμ||²` 只在 Σ 谱非负时成立（此时 `sqrt(Σ²)=Σ`）；Σ=diag(-4,-1) 时 `sqrt(Σ²)=|Σ|`，`F(X,X)=-20` 而不是 0。
- 官方不做对称性、正定性或半正定校验，也不做 `sqrtm` 的对称化；`A` 非对称但相似于半正定矩阵时特征值仍全为正实数。
- 形状消息用 `torch.Size([...])` 文本；`mu_y` / `sigma_y` 没有独立的维度校验，它们的越界以「shape 不一致」暴露。
- 官方无默认参数、无 dtype 参数、无 batch 维；docstring 引用的 `:cite:dowson1982frechet` 是高斯距离而不是离散曲线距离。

## 11. 示例与落地

```text
solve('frechet',         mu_x=[3.0], sigma_x=[[4.0]],
                        mu_y=[1.0], sigma_y=[[1.0]])   -> 5.0
  a = (3-1)^2 = 4 ; b = 4 + 1 = 5 ; c = sqrt(4) = 2
  F = 4 + 5 - 2*2 = 5
```

流水线结果：官方逐字环境冒烟通过、12 段探针 159/159 ALL GREEN、三层对拍 2238 项 / 52 组配置 ALL GREEN、判题 driver 107/107 ALL ACCEPTED、变异 137 项剔除 10 个文档化真等价后 127/127 全 CAUGHT。落地 11 文件（四份双语文档 + meta + solution / starter / test_cases + variants/pytorch 三件套），selftest 双模式、全库双语、短护栏与全量护栏依次收口。
