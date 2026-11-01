import torch


def solve():
    """单进程 DataLoader 装配账（固定 N=7, batch_size=3）。手写
    SequentialSampler / RandomSampler / BatchSampler 的装配链——不得使用
    torch.utils.data 的任何类；随机原语只允许 torch.randperm / torch.randint
    配显式 Generator。

    官方口径（torch/utils/data/sampler.py v2.14.1）：

      RandomSampler(generator 显式时直接用之；状态跨迭代延续)：
        replacement=True：num_samples//32 整块 torch.randint(high=n, size=(32,))
        + num_samples%32 零头一块（32 分块影响 RNG 消耗序列）
        replacement=False：num_samples//n 次完整 randperm(n)，随后
        **无论余数是否为 0** 再调一次 randperm 取前 num_samples%n 个
        （余数 0 时是空产出但仍消耗 generator——quirk）
        num_samples 非 int/bool 或 <=0 -> ValueError；
        replacement 非 bool -> TypeError（官方两类异常不同）
      BatchSampler：drop_last=True 每满 batch_size 个出一批（zip 同迭代器
        多引用实现）；False 用 islice，尾批保留（非空即出）
        batch_size 非 int/bool 或 <=0 -> ValueError；drop_last 非 bool -> ValueError
        __len__：drop_last -> N//bs；else -> (N+bs-1)//bs
      DataLoader shuffle=True -> RandomSampler(dataset, generator=generator)；
        len(loader) = len(batch_sampler)；多 epoch 迭代同一 loader 时
        generator 状态延续（每次 iter(loader) 重新 __iter__ 但同一 generator）

    需要产出的账（键固定）：
      seq_drop_false / seq_drop_true     顺序索引批账
      shuf_e0_flat / shuf_e1_flat        同一 Generator(seed=0) 两 epoch 的 randperm
      shuf_e0_batches / shuf_e1_batches  上述置换按 bs=3 drop_last=False 切批
      shuf_drop_true_batches             e0 置换按 bs=3 drop_last=True 切批
      shuf_replay_flat                   重新 manual_seed(0) 单次 randperm
      shuf_seed1_flat                    manual_seed(1) 单次 randperm
      repl_n7                            replacement=True num_samples=7（单块 7 零头）
      repl_n100                          replacement=True num_samples=100（3 整块 + 4 零头）
      randperm_mod0_flat                 replacement=False num_samples=14（=2n）两次 randperm 拼接
      quirk_probe_perm                   第三次空消耗后第四次 randperm（受 quirk 影响）
      len_drop_true / len_drop_false     N=7,bs=3 的 len 账
      len_bs_gt_n_true / len_bs_gt_n_false   batch_size=10 > N 的 len 账
    """
    N, bs = 7, 3

    def batches_drop_true(idx):
        return [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]

    def batches_drop_false(idx):
        out = [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
        tail = idx[(len(idx) // bs) * bs:]
        if tail:
            out.append(tail)
        return out

    out = {}
    seq = list(range(N))
    out["seq_drop_false"] = batches_drop_false(seq)
    out["seq_drop_true"] = batches_drop_true(seq)

    g = torch.Generator().manual_seed(0)

    def epoch_perm(gen):
        # 官方 _BaseDataLoaderIter：每次 iter(loader) 先从 generator 抽一个
        # int64 作 _base_seed；随后 RandomSampler 每 epoch 走完整 __iter__：
        # randperm 产出（num_samples//n 次）+ **无条件**再 randperm 取
        # num_samples%n（=0 时是空产出但仍消耗 generator）
        _ = torch.empty((), dtype=torch.int64).random_(generator=gen)
        perm = torch.randperm(N, generator=gen).tolist()
        _ = torch.randperm(N, generator=gen).tolist()[:0]  # 空消耗
        return perm

    e0 = epoch_perm(g)
    e1 = epoch_perm(g)
    out["shuf_e0_flat"] = e0
    out["shuf_e1_flat"] = e1
    out["shuf_e0_batches"] = batches_drop_false(e0)
    out["shuf_e1_batches"] = batches_drop_false(e1)
    out["shuf_drop_true_batches"] = batches_drop_true(e0)

    g2 = torch.Generator().manual_seed(0)
    out["shuf_replay_flat"] = torch.randperm(N, generator=g2).tolist()
    g3 = torch.Generator().manual_seed(1)
    out["shuf_seed1_flat"] = torch.randperm(N, generator=g3).tolist()

    gr = torch.Generator().manual_seed(0)
    out["repl_n7"] = torch.randint(high=N, size=(N % 32,), generator=gr).tolist()
    gr2 = torch.Generator().manual_seed(1)
    r100 = []
    for _ in range(100 // 32):
        r100 += torch.randint(high=N, size=(32,), generator=gr2).tolist()
    r100 += torch.randint(high=N, size=(100 % 32,), generator=gr2).tolist()
    out["repl_n100"] = r100

    gq = torch.Generator().manual_seed(0)
    p1 = torch.randperm(N, generator=gq).tolist()
    p2 = torch.randperm(N, generator=gq).tolist()
    _ = torch.randperm(N, generator=gq).tolist()[:0]  # quirk：空产出仍消耗
    out["randperm_mod0_flat"] = p1 + p2
    out["quirk_probe_perm"] = torch.randperm(N, generator=gq).tolist()

    out["len_drop_true"] = N // bs
    out["len_drop_false"] = (N + bs - 1) // bs
    out["len_bs_gt_n_true"] = N // 10
    out["len_bs_gt_n_false"] = (N + 10 - 1) // 10
    return out
