# DataLoader 单进程装配账

## 题意

`DataLoader` 的索引装配链（单进程、`num_workers=0`）由三层组成：Sequential/RandomSampler 出索引流 → BatchSampler 切批 → collate。本题手写整条装配链（禁用 `torch.utils.data`），固定 N=7、batch_size=3，输出顺序/随机/有放回/len 四类账，oracle 用真官方类对拍。

## 求解

官方口径（`torch/utils/data/sampler.py` + `dataloader.py` v2.14.1）：

- **Sequential**：`range(N)` 顺序流；BatchSampler `drop_last=True` 丢尾批、`False` 保留（尾批非空即出）；
- **RandomSampler**（generator 显式、状态跨迭代延续）：`replacement=False` → `num_samples//n` 次完整 `randperm(n)`，随后**无条件**再 `randperm` 取前 `num_samples%n`——余数 0 时是空产出但**仍消耗 generator**（quirk，影响后续置换）；`replacement=True` → 32 分块 `randint` + 零头块（纯代码结构，RNG 序列与单次调用等价）；`num_samples` 非 int 或 ≤0 → `ValueError`（bool 是 int 子类且 >0，官方**放行**、True 被当 1）；`replacement` 非 bool → `TypeError`（与 BatchSampler 的 `ValueError` 不同类）；
- **DataLoader 每次迭代**（每个 epoch 的新 `_BaseDataLoaderIter`）：先从 `loader.generator` 抽一个 int64 作 `_base_seed`（`torch.empty((), dtype=torch.int64).random_`），**然后** RandomSampler 才 `randperm`——所以 DataLoader 的 epoch0 置换 ≠ 裸 `randperm(seed)`；多 epoch 同一 loader 时 generator 状态延续；
- **len**：`len(DataLoader) = len(BatchSampler)`；`drop_last → N//bs`，否则 `(N+bs-1)//bs`；batch_size > N 时 drop_last=True → 0 批、False → 1 批；
- **BatchSampler 校验**：`batch_size` 非 int/bool 或 ≤0、`drop_last` 非 bool → `ValueError`。

## 数据规模

- 账共 17 键：顺序批账 2、随机置换/批账 5（两 epoch 同 generator、replay、seed1、drop_last 切批）、有放回 2（num_samples=7 零头块、100 的 3+1 分块）、`num_samples=2n` quirk 账 2（两次置换拼接 + 空消耗后第四次置换探针）、len 账 4。
- oracle 直接调真 `DataLoader` / `RandomSampler`（官方类作锚点），批元素 int 化比对。
