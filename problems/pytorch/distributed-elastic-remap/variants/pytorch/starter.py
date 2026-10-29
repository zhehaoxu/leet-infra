def solve(world_size, failed_ranks, param_bytes):
    """Checkpoint-backed shard restoration / 基于检查点的故障分片恢复账本。

    Inputs / 输入:
        world_size: positive Python int, original rank count / 原 rank 数，正整数。
        failed_ranks: list[int], old IDs in [0, world_size); may be empty,
            unsorted, or duplicated. Duplicates count once (set semantics).
            旧 rank ID 列表，可为空、无序、重复；重复 ID 按集合语义仅计一次。
        param_bytes: nonnegative Python int divisible by world_size; total
            parameter bytes across equal original shards; zero is allowed.
            原等大分片的参数总字节数，非负且必须整除 world_size，允许为零。
        All integers exclude bool / 所有整数均不含 bool。
        A consistent durable checkpoint contains EVERY failed shard; survivors
            keep their own shard at that same checkpoint state. This routine
            only plans restoration and cannot recover unique lost RAM.
            一致的持久化检查点包含每个故障分片，幸存者保留同一检查点状态的自身分片。
            本函数仅规划恢复，不能恢复没有持久化副本的唯一内存数据。

    Returns / 返回:
        (new_world_size, new_rank_of_old, failed_shard_dest, migration_bytes):
        new_world_size: int in [1, world_size], number of survivors / 幸存 rank 数。
        new_rank_of_old: list[int] of length world_size; -1 for failures,
            otherwise [0, new_world_size) in ascending OLD-ID order.
            长度为 world_size；故障为 -1，幸存者按旧 ID 升序映射到连续新 ID。
        failed_shard_dest: list[int], length len(set(failed_ranks)); aligned
            with SORTED UNIQUE failed IDs. Entry i is new rank i % new_world_size.
            与排序去重后的故障 ID 对齐，第 i 项为新 rank i % new_world_size。
        migration_bytes: nonnegative int, FAILED-shard checkpoint restore payload
            only: len(set(failed_ranks)) * (param_bytes // world_size).
            仅故障分片检查点恢复载荷字节数，不含幸存分片或完整重启开销。
        No failures: identity remap, empty destinations, zero payload / 无故障时恒等映射、空目的列表、零载荷。

    Raises / 异常:
        ValueError for invalid types/ranges, non-divisible param_bytes, or all
        ranks failing after deduplication / 类型、范围、整除约束不满足，或去重后全部 rank 故障。
    """
    raise NotImplementedError
