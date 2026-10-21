"""参考答案：Dataset 与 DataLoader —— map-style 协议实现。"""

import torch


class SyntheticDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, i: int):
        # 显式拒绝负索引：Python 的 x[-1] 语义在数据集协议里是坑，标准约定是 IndexError
        if i < 0 or i >= len(self):
            raise IndexError(f"index {i} out of range for dataset of length {len(self)}")
        x = torch.tensor([i, 2 * i, 3 * i, 4 * i], dtype=torch.float32)
        y = torch.tensor(i % 2, dtype=torch.int64)
        return x, y


def solve() -> torch.utils.data.Dataset:
    return SyntheticDataset()
