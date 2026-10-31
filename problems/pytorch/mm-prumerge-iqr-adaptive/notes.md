# 解析：LLaVA-PruMerge 的 IQR 自适应 token 数

## 实现思路

先把输入校验干净：`cls_attn` 必须是 float32/float64 的二维张量、`B,N ≥ 1`、全部有限；除此之外不设非负约束——percentile 与严格大于比较对任意实数都良定义。

比率一步逐行对齐上游 `clip_encoder.py::outlier_dectection`：`attn.to(dtype=torch.float32).cpu().numpy().flatten()` 先把整个 `[B,N]` 降精度摊平，`np.percentile(·, 25/75)` 取 Q1/Q3（linear 插值），`upper = Q3 + 1.5*(Q3-Q1)`，`np.where(flat > upper)` 严格大于计数，`ratio = len/len`。注意三点：**float32 cast 发生在 percentile 之前**（float64 输入被降精度是可观察行为，测试用 `nextafter(1.0)` 的压线用例钉住：float64 下它是离群、cast 后与 1.0 归并、ratio 归零）；**摊平是跨 batch 的**（多 batch 用例区分逐行变异）；**比较是严格大于**（全相同值时 `IQR=0`、`upper=Q3`，零离群 → `ratio=0` → `k=0` 合法走空选分支，这正是与 536 抛 `ValueError` 的分工）。

选择一步与 536 同构：`argsort(descending=True, stable=True)` 实现「值降序、下标升序」，取前 `k` 列后重排升序发出；pruned 用布尔掩码 + stable `argsort` 取**后段** `[:, keep:]`——前段是 kept 的升序（0 排前且稳定），切前 `tokens-keep` 列是错的（本轮 selftest 曾在此翻车，样例输出 `[7,0,1,2,3,4,5]` 暴露）。`k=0` 时直接返回空 `[B,0]` 与完整 `0..N-1`。

## 与官方实现的刻意差异

- **顺序**。上游 `torch.topk` 的输出按值降序，本题沿用 536 的约定把 kept 重排成升序下标，使 token 序与因果顺序一致。
- **`k=0` 合法化**。536 把数量钉成入参、`k=0` 无定义故抛错；本题的比率是算出来的，`ratio=0` 是真实可达状态（注意力均匀时），空选是上游可观察行为，因此合法化并在契约写明。
- **batch 轴**。上游在真实推理里 `B=1`，摊平与否不可观察；本题显式规定摊平跨 batch，并把逐行比率作为被测试抓住的变异。
- **返回 ratio**。上游只把它当内部量传给 `topk`；本题作为第一返回值暴露，判题用精确相等（两侧都是整数计数比，无浮点容差问题）。

## 独立验证

oracle 不复用任何 numpy/torch 的分位与选择实现。percentile 用纯 Python 手写：先 `sorted`，虚位 `pos = q/100*(n-1)`，再按 numpy `_lerp` 的**双路**公式（`t ≥ 0.5` 时走 `b - (b-a)*(1-t)`）在 **float32 逐步算术**（每步 `_f32` 截断：减法、乘法、加减各一次）下计算——这条路径与 `np.percentile(float32 数组)` 做了 4 万组随机位级对拍零差异；纯 double 算术（即使最后 cast float32）在约 3% 的用例上差 1 ulp，会被压线比较放大成 ratio 分歧。选择用标量循环：`sorted(key=(-value, index))` 稳定降序、重排升序、补集推导。ratio 是整数计数比，两侧 Python float 精确相等；kept/pruned 用 `torch.equal`。

## 复杂度

摊平与两趟 percentile 各 `O(BN log(BN))`（排序主导），计数 `O(BN)`；每行稳定排序 `O(N log N)`、掩码 argsort `O(N log N)`，合计 `O(BN log N)`。中间量只有摊平后的 float32 序列与 `[B,N]` 的 int64 掩码。

## 来源与范围

依据已回源的 `42Shawn/LLaVA-PruMerge`，文件 `llava/model/multimodal_encoder/clip_encoder.py`：`outlier_dectection` 的 float32 摊平、`np.percentile` 25/75、`upper_bound = Q3 + 1.5 * IQR`（下行就是被注释掉的 lower_bound，不出题）、`np.where(attn_np > upper_bound)` 严格大于、`ratio = len(outlier_indices) / len(attn_np)`，以及 `token_prune_merge_advanced` 里 `if if_adaptive: reduction_ratio = outlier_dectection(cls_attn)` 的接线和后续 `torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)` 的选择。上游 ratio 旁有一个被注释的 `#*3.5` 缩放，未启用，不出题。

刻意不在范围内：536 已覆盖的余弦分配与残差合并；`token_prune_merge_advanced_plus` 的空间采样补充分支（arange 步长采样是另一套确定性账，如立项另行走查）；真实 CLIP 前向与 hook 抓取（k/q 由调用方保证）。IQR 下尾（lower_bound）上游自己也没用，同样不出。

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-prumerge-iqr-adaptive --mode submit
```

## 变异测试（7/7 全抓）

`>=` 上界（比较边界）、去掉 1.5 系数、Q1/Q3 互换、float32 cast 退化为 float64（压线用例专属）、kept 按值降序发出（随机用例 k>1 时抓住）、`int()` 换 `int(+0.5)` 舍入、升序重排变异。初版随机用例用 uniform(0,1) 几乎无离群值、全走 k=0 空选，导致 kept 顺序变异存活——改成「紧致主体 + 20% 注入离群」的混合分布后全抓；这也是本批的一个重要教训：**随机用例的分布必须让被测分支真的发生**。
