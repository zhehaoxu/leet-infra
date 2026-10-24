# 训练态显存四件套与 ZeRO 分片参考解
def solve(n_params, world):
    N = float(n_params)
    W = float(world)
    # 每参数字节：fp16 参数2 + fp16 梯度2 + fp32 master4 + m4 + v4 = 16
    stage0 = 16.0 * N                       # 不分区，每卡全套
    stage1 = N * (4.0 + 12.0 / W)           # 分区优化器状态(master+m+v=12)
    stage2 = N * (2.0 + 14.0 / W)           # 再分区梯度(+2 => 14)
    stage3 = 16.0 * N / W                    # 全分区
    return (stage0, stage1, stage2, stage3)
