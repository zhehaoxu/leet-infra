import torch


def solve():
    """单进程 DataLoader 装配账（固定 N=7, batch_size=3）。手写
    SequentialSampler / RandomSampler / BatchSampler 的装配链——不得使用
    torch.utils.data 的任何类；随机原语只允许 torch.randperm / torch.randint
    配显式 Generator。

    账点口径（题面钉死）：
      顺序账 = range(N) 切批；drop_last=True 丢尾批、False 保留
      RandomSampler（generator 显式、状态跨迭代延续）：
        replacement=False：num_samples//n 次完整 randperm + 余数切片
          （余数 0 仍空消耗一次 randperm——quirk，影响后续置换）
        replacement=True：32 分块 randint + 零头块
      len：drop_last -> N//bs；else -> ceil(N/bs)
      多 epoch 同 loader：generator 延续（e1 接着 e0 消耗）

    返回 dict（键见 solution 注释 / 测试断言）。
    """
    raise NotImplementedError
