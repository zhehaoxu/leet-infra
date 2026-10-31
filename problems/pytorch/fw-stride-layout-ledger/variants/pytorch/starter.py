def solve(shape, strides, offset, ops):
    """张量布局账：按 PyTorch 视图规则推演 shape/stride/storage_offset/连续性。

    一个张量 = 同一段存储 + (shape, strides, offset) 三个元数据；本题给定基元数据与一串
    视图操作，要求逐步算出操作后的元数据，并判断每一步是「复用存储的 view」还是
    「复制数据的新张量（copy）」。全程不许调用 torch 读元数据，只能按下面的规则算。

    参数::

        shape   : list[int]，基张量各维长度，非负，长度即 ndim（可为空列表 = 0 维）
        strides : list[int]，与 shape 等长的非负步长
        offset  : int，基张量的 storage_offset（非负）
        ops     : list，每个元素是 [kind, *args]，args 一律是 int（squeeze 的维可为 None）

    操作集与参数（d / d0 / d1 / index 允许负数）::

        ["transpose", d0, d1]        交换两维；d0 == d1（归一化后）时是不变操作
        ["permute", *dims]           dims 归一化后必须是 range(ndim) 的一个排列
        ["unsqueeze", d]             插入一个长度 1 的维；d 的合法范围是 [-ndim-1, ndim]
        ["squeeze", d_or_None]       d 给整数时只删该维（且仅当它的长度为 1），
                                     给 None 时删掉所有长度为 1 的维
        ["narrow", d, start, length] 沿维 d 从 start 起取 length 个，保留该维
        ["select", d, index]         取沿维 d 的第 index 个切片，并删掉维 d
        ["slice", d, start, end, step]
                                     沿维 d 取 [start, end) 且步长为 step，保留该维；step >= 1
        ["expand", *sizes]           长度 >= ndim，右对齐；-1 表示保持原长度
        ["reshape", *sizes]          目标形状，至多一个 -1 待推断
        ["flatten", d_start, d_end]  把 [d_start, d_end] 闭区间的维合并成一维
        ["contiguous"]               无参；已是 C 连续时原样返回，否则复制

    返回：与 ops 等长的 list，每个元素是 dict，键固定为::

        {"shape": list[int], "strides": list[int], "offset": int,
         "contiguous": bool, "kind": "view" | "copy"}

    ``kind`` 指该步相对**上一步**的存储：view = 复用同一段存储（offset 可非 0），
    copy = 新分配存储（此时 strides 取目标 shape 的 C 连续步长，offset 归 0）。

    规则（与 torch 的 ATen 实现一致，负数维先按 `d + ndim` 归一化）：

    1. **transpose/permute** 只重排 shape 与 strides 的分量，offset 不变。
    2. **unsqueeze(d)**：新维长度 1，步长取 `size[d] * stride[d]`（即「右边那一整段的
       跨度」）；当 d == ndim（追加到末尾）时步长为 1。其余维原样保留。
    3. **squeeze** 只是把对应分量从 shape/strides 里删掉，**不重算**留下的步长；
       被指定的维长度不是 1 时整步为空操作。
    4. **narrow/select/slice** 都会推进 offset：
       `offset += start_norm * stride[d]`，其中 start_norm 先 `+= size[d]`（若为负）、
       再夹紧到 `[0, size[d]]`；slice 的 end 同样先 `+= size[d]` 再夹紧到
       `[start_norm, size[d]]`，结果长度 `= ceil((end - start_norm) / step)`，
       该维步长 `*= step`（长度为 0 时依然按此式给出步长，offset 仍推进到 start_norm）。
       select 的长度为 1，等价于 narrow 后删掉该维。narrow 保留该维、步长不变。
    5. **expand(sizes)**：右对齐后逐维——目标长度与原长度相等则步长不变（含 1 与 1）；
       原长度为 1 而目标 > 1 则步长置 0；其余情况不合法。多出来的前置新维步长为 0，
       除非其长度为 1，此时步长取 `size[i+1] * stride[i+1]`（若它已是最后一维则取 0）。
    6. **reshape/flatten 的 view 判定**（ATen `computeStride`）：把旧形状从最内维向外切成
       若干**连续子空间**——设已累乘的 `tensor_numel`（当前段内旧长度之积）与段基步长
       `base`，则维 i 与 i+1 同段当且仅当 `size[i] == 1` 或
       `stride[i] == tensor_numel(i+1..) * base`；每段必须被目标形状从最内维向外「精确吃满」，
       段内第 j 个目标维的步长 = 该维之前在本段内累乘的长度 × base；长度为 1 的目标维可以
       挂在任意位置（它不消耗本段长度，步长按同一累乘给出）。任一段吃不满或目标维有剩余
       即判定为 copy。copy 后步长取目标形状的标准连续步长（长度 <= 1 的维不推进累乘），
       offset 归 0。旧形状总长度为 0 时 reshape 恒为 view：与旧形状完全相同则照抄旧步长，
       否则按目标形状的标准连续步长给出。0 维基张量的 reshape 一律给出全 1 步长。
       `flatten(a, b)`：a == b 时为空操作；旧 ndim == 0 时等价于 reshape 到长度 1 的一维；
       其余情况先按规则 6 对合并后的目标形状判定。
    7. **contiguous** 在当前布局已 C 连续时是 view（元数据不变），否则是 copy。
       C 连续判定：ndim == 0 或总长度为 0 时恒为真；否则从最内维向外，期望步长从 1 起，
       每维要求 `stride == expected` 或 `size == 1`，然后 `expected *= size`。
       注意长度为 1 的维的步长是「自由」的，但 torch 仍会按规则 2/5/6 填成确定值，
       本题要求连这些填充值也一致。

    非法输入一律抛 `ValueError`（不必是 torch 的报错文本）：shape/strides 长度不等或含负、
    offset 为负、ops 不是列表、op 为空或 kind 不在操作集内、args 含非 int、维/index 归一化后
    越界、permute 不是排列、0 维张量上使用 transpose/permute/narrow/select/slice/flatten
    （squeeze 与 unsqueeze 在 0 维上合法：前者为空操作，后者得到长度 1 的一维）、
    narrow 的 start+length 超出该维长度或 length 为负、
    slice 的 step <= 0、expand 的尺寸冲突或给了 -1 以外的负数、reshape 出现两个 -1 或
    总长度对不上（含「总长度为 0 时用 -1 推断」这一无法确定的情形）、flatten 的
    d_start > d_end、contiguous 带参数。

    题目保证 ops 里每一步都能在真实 torch 上成功执行，因此不会要求你复现 torch 的异常文本。

    不得调用 torch（或 numpy）构造张量、再读取其 `shape/stride/storage_offset/is_contiguous`
    来查表得到答案；请按上述规则自行计算。
    """
    raise NotImplementedError
