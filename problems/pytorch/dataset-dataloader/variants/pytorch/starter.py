import torch


class SyntheticDataset(torch.utils.data.Dataset):
    # TODO: __len__ 返回 100；__getitem__(i) 返回 (float32 [i,2i,3i,4i], int64 i%2)
    pass


def solve() -> torch.utils.data.Dataset:
    # TODO: 返回数据集实例
    raise NotImplementedError
