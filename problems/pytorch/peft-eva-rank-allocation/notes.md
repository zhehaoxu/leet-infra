# 详解：EVA 解释方差驱动的全局秩分配

## 出处

EVA（One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation，arXiv:2410.07170，Paischer 等，ELLIS Unit LIT AI Lab / NXAI）：对每层 LoRA 目标模块的输入激活做**增量主成分分析**（torch_incremental_pca.IncrementalPCA 的 partial_fit），把 LoRA 的秩预算按"解释方差"跨层调度；初始化用各层主成分方向作 A（可 whiten/按奇异值缩放）。官方仓 ml-jku/EVA：`svd_precompute.py`（CLI 壳）→ `src/svd.py::compute_svd`（235 行核心）+ SVDHook（增量 PCA + 余弦相似收敛判定）+ HashHook（等价输入检测）。PEFT 侧以 PR 形式集成（init 选项，非独立 tuner 目录）。

## 官方实现的逐行翻译

**指标四态（`_get_metric`）**：

```python
if metric == "raw":     return svd.explained_variance_
elif metric == "ratio": return svd.explained_variance_ratio_   # == ev/ev.sum()
elif metric == "sum":   return svd.explained_variance_ / svd.explained_variance_.sum()
elif metric == "max":   return svd.explained_variance_ / svd.explained_variance_.max()
else: raise ValueError
# 注意 ratio 与 sum 数学等价（两个官方分支），判题域同样保留两分支
```

**全局秩分配（`_get_rank_distribution`）**：

```python
exp_vars = {k: _get_metric(h.svd, metric)[:max_components] for k, h in hooks.items()}
#   max_components = round(rank * rho)   # rho >= 1 由 assert 保护
keys, values = zip(*[(k, c) for k, name in hook_layer_map.items() for c in exp_vars[name]])
#   hook_layer_map: {层: SVD源层}——等价层展开的是源层的向量（值重复！）
idx = torch.stack(values).argsort(descending=True)
counts = Counter([keys[i] for i in idx[:rank_budget]])       # rank_budget = len(hooks)*rank
counts = {k: counts.get(k, 0) for k in hook_layer_map.keys()} # 缺票层显式补 0
for k, k_hook in equal_inputs_map.items():
    rank, rank_hook = counts[k], counts[k_hook]
    if rank_hook >= rank:
        continue                                              # 单向交换：源层秩低才换
    counts[k_hook], counts[k] = rank, rank_hook
```

**组装（`compute_svd` 尾部）**：

```python
for name, rank in rank_dist.items():
    if rank == 0: continue                                    # 秩 0 层不进 dict
    hook = hooks[layer_hook_map[name]]                        # 等价层也取源层的 hook
    u = hook.svd.components_[:rank]
    if whiten:                    u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
    elif scale_by_singular_values:                             # elif：双 True 时 whiten 优先
        s = hook.svd.singular_values_[:rank]
        s /= s.max()
        u *= s.reshape(-1, 1)
svd_dict[name] = u
```

**SVDHook 收敛判定（不入题）**：每 batch `partial_fit` 后 `cosine_similarity(components, previous) >= sim_thresh` 逐分量 bool；全部收敛（且 `i > min_batches`）才退出循环。判题域 explained_variance 显式入参，整段剔除。

## 判题域剔除与红线

- IncrementalPCA / partial_fit / 真实模型 forward hook / DataLoader / attention_mask·labels 掩码：真实前向红线。
- 收敛循环（sim_thresh / min_batches / convergence_dict）：数据流依赖，不入题。
- HashHook 的 hash 碰撞检测（`hash(tuple(tensor.view(-1).tolist()))`）：Python hash 随机化，不可判题——判题域 equal_inputs_map 显式入参。
- `torch.save(svd_dict)` 序列化与 fileref 拼名：IO 红线。

## 关键怪癖（回源逐行确认）

1. **ratio 与 sum 是数学等价的两个官方分支**（都是 ev/ev.sum()）——判题域两分支都保留，测试验证"等价性"（同输入同输出）。
2. **每层先归一再跨层排序**：归一让小方差层反超（示例 l2 绝对值最小却拿满预算）——raw 与 ratio 的分配可以完全不同。
3. **等价层展开的是源层的向量**（值重复）：argsort 相等值按索引序（CPU 确定实现），源层在 hook_layer_map 中先展开 → 源层恒先入预算 → `if rank_hook >= rank: continue` 的交换分支在判题域内**永不 fire**（真等价），但官方逻辑保真写全，测试只断言交换后的不变量 `counts[源层] >= counts[等价层]`。
4. **whiten 与 scale 的 elif**：if-return 结构下 elif 与两个独立 if 行为一致（第一分支必 return）——"双 True 时 whiten 优先"是官方语义而非常规 else-if 排他，测试以双 True 用例钉死。

## 变异测试（23/23，2 等价剔除）

- 指标：ratio 丢归一 / sum 换 max / max 换 sum / 非法 metric 静默——全 CAUGHT。
- 分配：截断 +1 / max_components −1 / argsort 升序 / 预算 −1 / 缺票补 1 / 展开丢首分量 / 键名用源层 / 预算公式丢 rank / swap 自赋值——除 swap 外全 CAUGHT。
- 缩放：whiten 不开方 / max 换 min / 乘变除 / reshape 轴反 / identity 清零——全 CAUGHT。
- 组装：跳过失效 / 切片 +1 / 源层查自身 / 奇异值切片偏移 / 键名写源 / 跳过反向——全 CAUGHT。
- **2 真等价剔除**：① m-rd-swap（swap 自赋值）——判题域内等价层展开值与源层相同、argsort 相等值按索引序使源层恒先入预算，交换分支永不 fire（见怪癖 3）；② m-sc-elif（elif → if）——if-return 结构下第一分支必 return，两写法行为一致（见怪癖 4）。

## 与已有题的划界

- sparsity 组的 SVD 题（svdllm/zssvd 等）：那些是把 SVD 用在**权重白化/降秩**上，谱来自权重矩阵；本题的输入是**激活的 explained_variance 向量**（显式入参），考的是**跨层预算分配账**，不做任何 SVD。
- `peft-tinylora-svd-projection`（795）：冻结 SVD 因子的参数化，无跨层调度。
- PSOFT/OSF（① 组候选，未立项）：依赖 top-k SVD 的掩码初始化，oracle 需对拍真实 SVD；本题无 SVD 依赖（官方 svd 预计算离线，判题域入参化）。
