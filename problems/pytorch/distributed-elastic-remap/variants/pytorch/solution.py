def solve(world_size, failed_ranks, param_bytes):
    """Survivor renumbering and failed-shard checkpoint restore accounting."""
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive int")
    if type(param_bytes) is not int or param_bytes < 0 or param_bytes % world_size:
        raise ValueError("param_bytes must be a nonnegative int divisible by world_size")
    if not isinstance(failed_ranks, list) or any(
        type(r) is not int or not 0 <= r < world_size for r in failed_ranks
    ):
        raise ValueError("failed_ranks must be a list of valid integer old rank IDs")
    failed = set(failed_ranks)
    if len(failed) == world_size:
        raise ValueError("at least one rank must survive")
    surviving = [r for r in range(world_size) if r not in failed]
    new_world_size = len(surviving)
    new_rank_of_old = [-1] * world_size
    for new_id, old_id in enumerate(surviving):
        new_rank_of_old[old_id] = new_id
    sorted_failed = sorted(failed)
    failed_shard_dest = [i % new_world_size for i in range(len(sorted_failed))]
    shard_size = param_bytes // world_size
    migration_bytes = len(sorted_failed) * shard_size
    return new_world_size, new_rank_of_old, failed_shard_dest, migration_bytes
