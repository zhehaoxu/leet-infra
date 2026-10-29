def solve(ops, align, grow_unit):
    """缓存分配器（caching allocator）块账：按 best-fit 分配、按需扩段、释放时合并相邻空闲块。

    ops       : 操作序列，每项 (kind, id, nbytes)
                kind="alloc"：申请 id（唯一，分配后不会重复申请）nbytes 字节
                kind="free" ：归还 id（此前已 alloc 且未释放），nbytes 位恒为 0，忽略
    align     : 对齐粒度（>=1），所有申请向上取整到 align 的整数倍才真正占用
    grow_unit : 最小扩段字节数（>=1）；空闲块不足时向设备再要一段 max(need, grow_unit)
                （两者都向上对齐到 align）

    模型（务必按此定义，整卡视作一条从 0 开始的连续地址空间）：
      need          = ceil(nbytes / align) * align
      空闲块集合     : 一组互不重叠、按 start 升序的 [start, size)，始终是「极大块」
                      （相邻空闲块在释放后必须立即合并）
      best-fit      : 在所有 size >= need 的空闲块里选 size 最小者；
                      并列（size 相同）取 start 更小者
      分配          : 从选中的块头部切出 need；若块更大，剩余 [start+need, start+size)
                      仍为空闲块（分配后仍保持极大性，无需再合并）
      扩段          : 没有 size >= need 的空闲块时，reserved += grow，
                      grow = max(need, ceil(grow_unit / align) * align)；
                      新段 [旧 reserved, 旧 reserved+grow) 加入空闲集合并与相邻块合并，
                      然后重新做一次 best-fit（可能落在旧的小块上，不一定落在新段）
      释放          : 把该块的 [start, need) 放回空闲集合，并与左右相邻空闲块合并
      peak_reserved : reserved 的历史最大值（reserved 只增不减）

    返回 (peak_reserved, final_reserved, num_grow, num_free_blocks, largest_free, free_blocks)：
      peak_reserved / final_reserved : int，字节
      num_grow       : int，发生扩段的次数
      num_free_blocks: int，结束时空闲块个数（合并正确时相邻块不会各算一个）
      largest_free   : int，结束时最大空闲块字节数，没有空闲块记 0
      free_blocks    : list，结束时空闲块 [[start, size], ...]，按 start 升序
    """
    # TODO
    raise NotImplementedError
