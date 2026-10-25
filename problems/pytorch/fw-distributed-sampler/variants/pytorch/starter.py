import torch
from torch.utils.data.distributed import DistributedSampler


def solve():
    """单机模拟 DistributedSampler，返回 epoch/切片/补齐的全部索引账（无任何随机全局态）。

    固定构造：数据集长度 \(N=7\)（`len(dataset)` 返回 7 即可，元素内容无关），
    `num_replicas=3`、`seed=0`；rank 取 \(0,1,2\)。

    需要产出的账（每份都是 `[rank0列表, rank1列表, rank2列表]`）::

        1. num_samples / total_size：来自 shuffle=True, drop_last=False, seed=0 的
           epoch0 采样器对象属性（未补齐前 \(\lceil N/R\rceil\) 与 \(\lceil N/R\rceil\cdot R\)）
        2. epoch0_ranks / epoch1_ranks：shuffle=True, drop_last=False 在
           set_epoch(0) / set_epoch(1) 下三个 rank 各自 list(sampler)
        3. epoch0_again：把 epoch0 的配置原样重跑一遍（验证同 epoch 重放一致）
        4. drop_last_epoch0_ranks：shuffle=True, drop_last=True, epoch0
           —— 此时每 rank 只有 \(\lfloor N/R\rfloor\) 个索引
        5. no_shuffle_ranks：shuffle=False, drop_last=False, set_epoch(3)
           —— epoch 对非 shuffle 无效，序列就是 range(N) 补齐后切片

    引擎语义（自行推导可核对）::

        置换种子 = seed + epoch；full = perm + perm[:total_size - N]（补齐）
        或 perm[:total_size]（drop_last 截尾）；rank r 拿 full[r::num_replicas]
    """
    # TODO: 用 DistributedSampler 生成上述账目
    raise NotImplementedError
