# notes — fw-rng-generator-ledger（order 713）

## 回源

- 权威源：本地判题环境 `.venv-eval/lib/python3.12/site-packages/torch`（v2.14.1）+ `_C.Generator` 实测。GitHub 抓取不看——RNG 语义随版本微调，实测优先。
- `torch.manual_seed(s)` 返回值：v2.14 返回 `torch._C.Generator`（且该对象**驱动默认流**——用它 `rand(generator=ret)` 与裸 `rand()` 同序列）。旧文档"返回 None"已过时。
- **seed 包装口径（最易记错的账）**：`initial_seed` 返回 `s mod 2**64` 的**无符号** Python int。`-5 → 18446744073709551611`、`2**63 → 9223372036854775808`（不是 `-2**63`！）。合法域 `[-2**63, 2**64-1]`，域外 `ValueError: Overflow when unpacking long long`。
- **fresh Generator 缺省种子 = 67280421310721**（MT19937 黄金常数 0x3D2B_... 的十进制），且两个 fresh 对象序列逐元素相同——"Generator() 是随机播种的"是误传。
- `manual_seed` 类型拒绝：`True/1.5/"7"` 一律 `RuntimeError`（bool 是 int 子类，须先于 int 分支拒——与 710 calculate_gain 的 bool param 同款优先级坑）。
- `get_state()` 返回 `uint8[5056]`（MT19937 全状态 + 位置指针），中流恢复逐值精确。

## 题面设计

- 七键 generator 包由判题面构造，`"default"` 直接引用 `torch.default_generator` 本身——fork 账在真实默认流上验证。
- `_compare` 必须在 solve 与 oracle 之间**快照/恢复 default 状态**：fork 账恢复后 post 消费会把默认流前移 2 次抽取，两次调用看到的 default 态不同步必挂（第一版就栽在这）。
- oracle 直调官方 `Generator.manual_seed/initial_seed` 作最硬锚点；`torch.random.fork_rng` 的官方语义交叉验证放在 check 函数（oracle 内不得调 `torch.manual_seed`，会污染 default 流）。
- 60 组随机对拍：随机合法 seed + 随机中流消耗量（`n_mid ∈ [0,9]`），中流 generator 独立重建保证 solve/oracle 同态。

## 变异（13 全抓 + 2 等价）

- 全抓关键：无符号包装 vs int64 重解释（`2**63` 用例）、bool 先拒、域双边界（**`-2**63` 恰合法用例不可漏**——首版漏掉导致 domain-lower 存活）、fresh 常数、fork/mid 恢复、dtype 流、消费原语与数量。
- 等价变异 2 条（SURVIVED，记此）：① midstream 恢复前冗余 `set_state(snap)`（恢复到当前态，序列不变）；② 域上界 `s <= 2**64-1` ↔ `s < 2**64`（整数语义恒等）。

## 坑

- **红线检查必须 AST 节点级**：首版用字符串匹配 `torch.manual_seed`，docstring 里的字样命中 → 变异 harness 下所有变异被误判 CAUGHT（13/13 假阳性）。改 AST 后 13/12 真实判定重跑，抓出真存活 domain-lower。
- str(seed) 作 dict 键：`str("7") == "7"` 与合法 int seed 7 撞键（本题 seeds 列表未同时含两者，无实际冲突；若扩展需 repr 区分）。
- 与 711 的划界：711 管 DataLoader 每 epoch 的 `_base_seed` 抽取与置换消耗；本题管生成器对象本身的包装/快照/fork 语义。`randperm(generator=...)` 在两题都出现但账不同层。
