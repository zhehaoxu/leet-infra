import torch


def solve(mask, weight, grad, sparsity, eps, u):
    """动态稀疏训练的一步掩码更新（RigL / MovERec 式 drop–grow）。
    mask/weight/grad/u float32 同形状（按展平索引处理），u∈[0,1) 为均匀数，仅用于打破并列。
    目标激活数 k = N - round(sparsity*N)。三步：
      0) 激活数 < k：从「输入 mask 中本就非激活」的元素里按 |grad| 降序（并列 u 降序、再索引升序）增长到 k；
      1) 激活数 > k：按 |weight| 升序（并列索引升序）剪到 k；
      2) 交换：r = round(eps × 当前非激活数)，上限为激活数与候选池大小；先剪 r 个 |weight| 最小的激活元素，
         再从候选池（输入 mask 中非激活、且未被步骤 0 增长的元素）按 |grad| 降序重激活 r 个。
    返回 (new_mask, reactivated, pruned)：new_mask float32 同形状；后两者为 int64 展平索引、升序。
    重激活集合 ⊆ 输入非激活集、剪枝集合 ⊆ 输入激活集，两者互斥。禁止 torch.nn.utils.prune。"""
    # TODO
    raise NotImplementedError
