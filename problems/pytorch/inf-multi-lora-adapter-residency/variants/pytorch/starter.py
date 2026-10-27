import torch


def solve(requests, adapter_bytes, capacity, pinned):
    """请求逐个到达时，adapter 常驻池怎么装、怎么换、峰值显存多大。

    requests:(M,) int64，按到达顺序排列的 adapter id;
    adapter_bytes:(A,) int64，各 adapter 的显存占用;
    capacity:int，同时最多驻留多少个 adapter;
    pinned:(P,) int64，常驻（被钉住、永不被驱逐）的 adapter id，仿真开始时就已加载。

    处理每个请求（adapter 记为 a）：
      a 已在驻留池 -> 命中，并刷新其「最近使用」次序；
      否则缺页：池未满就直接加载；池已满就驱逐最久未使用的**非 pinned** adapter 再加载；
      池满且全部驻留者都是 pinned -> 该请求无法服务，计入 rejected，不做任何改动。

    返回 (hit:(M,) bool、evicted:(M,) int64（本步被换出的 adapter，-1 表示没有换出）、
          loads:int64 标量（动态加载次数，不含初始 pinned）、
          rejected:int64 标量、peak_bytes:int64 标量（驻留 adapter 占用之和的峰值，含 pinned）)。
    """
    # TODO
    raise NotImplementedError
