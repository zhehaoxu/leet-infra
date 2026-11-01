# notes — fw-dataloader-assemble（order 711）

## 回源

- 出处：本地判题环境 `.venv-eval` 的 `torch/utils/data/sampler.py` 与 `dataloader.py`（v2.14.1）。关键实现：`RandomSampler.__iter__`（32 分块 randint / randperm + 无条件余数切片）、`BatchSampler.__iter__`（drop_last=True 的 zip 同迭代器多引用 vs False 的 islice）、`_BaseDataLoaderIter.__init__`（`_sampler_iter` 先创建（惰性）后 `_base_seed` 抽取）。
- 与 `fw-distributed-sampler`（253）划界：本题单进程装配（Sequential/Random/Batch 三层 + DataLoader 迭代器种子消耗），253 是 DistributedSampler 的多副本补齐/切片；与 `dataset-dataloader`（入门）划界：本题钉官方 RNG 消耗账。

## 判题面

- solve() 无参固定场景；oracle `_py` 直接用真 `DataLoader` / `RandomSampler`（官方类本身作锚点），批元素 int 化比对（DataLoader 对 int dataset collate 成 tensor）。
- 校验差异按官方异常类钉死：BatchSampler 的 batch_size/drop_last → ValueError；RandomSampler 的 num_samples → ValueError、replacement → **TypeError**。

## 锚点（全部经真 DataLoader 实测验证）

- **DataLoader epoch 链（本题最硬账点）**：每个 epoch 的新 `_BaseDataLoaderIter` 先从 generator 抽 `_base_seed`（int64 random_），随后 RandomSampler `randperm`；且 RandomSampler 每 epoch 结束**无条件**再 `randperm` 取 `num_samples%n`（=0 时空产出仍消耗）。N=7、seed=0 的官方链：epoch0 = `random_ + randperm + randperm(空)`、epoch1 同构。故 DataLoader 的 e0 ≠ 裸 `randperm(seed0)`（裸 = replay 场景）。
- **num_samples=2n quirk**：`num_samples//n=2` 次产出 randperm + 第三次空消耗 → 第四次置换探针与"无空消耗的第三次"不同（实测 [4,0,3,6,2,5,1] vs [1,4,5,3,0,2,6]）。
- **bool num_samples 放行**：RandomSampler 无 BatchSampler 式显式 bool 检查，True 是 int 子类且 >0 → 被当 1，恰出 1 个样本。
- len：drop_last=N//bs=2、ceil=3；bs=10>N → 0/1。
- replacement 32 分块：randint 3 整块 + 4 零头的产出与单次 `size=(100,)` 完全一致（见变异 notes）。

## 变异记录（10 条全抓）

seq-shift（range 位移）、drop-reverse、tail-drop-keep（尾批丢）、gen-reset-per-epoch（每 epoch 重置 seed）、replay-same-seed1、repl-no-block32、repl-single-call、quirk-no-consume、len-floor-always、epoch-no-baseseed、epoch-no-quirk-consume、bs-gt-n-len-swap 中 10 抓。

**等价变异 2 条记 notes（重要回源结论）**：`repl-no-block32`（`size=(N%32,)`→`size=(7,)`，N%32=7 恒等）与 `repl-single-call`（32 分块 → 单次 `size=(100,)`）——实测 CPU `torch.randint` 逐样本消耗 RNG，分块调用与单次调用产出**逐元素一致**：官方 32 分块是 yield 结构而非 RNG 语义（与 randperm 的空消耗 quirk 不同，后者真消耗状态）。测试还修正了两处自身 bug：`sorted(x) == list(range(7))*2` 的未排序对照、replay 与 e0 的 `_base_seed` 消耗差。

## 红线

solution 禁用 torch.utils.data（手写装配链，随机原语仅 randperm/randint + 显式 Generator）；多进程/ persistent_workers/prefetch 不出（num_workers=0 单进程账）；DataLoader 内层 collate 不考（索引账）；与 253 的 DistributedSampler 多副本账不重叠。
