import torch


def solve(x, output_size):
    # TODO: 自适应最大池化。output_size 为整数或 (oh, ow)。
    # 第 i 个输出覆盖输入区间 [floor(i*Hin/outH), ceil((i+1)*Hin/outH))，段内取 max。
    # 边界公式同 adaptive-avgpool，只是聚合换成最大值。
    raise NotImplementedError
