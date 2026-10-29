def solve(world_size, dims, order):
    """Build least-significant-axis-first rank coordinates and communication groups."""
    # world_size: int in [1,512]; dims: 1-4 distinct string axes -> positive int sizes.
    # order: list[str], each dims key exactly once, fastest-changing axis first.
    # Raise ValueError if order is not a permutation or product(dims) != world_size.
    # Return world_size dicts ordered by rank, with rank(int), coord(dict[str,int]),
    # groups(dict[str,list[int]], members ascending), local_ranks(dict[str,int]),
    # group_ids(dict[str,int], dense per-axis IDs, ordered by minimum group member).
    raise NotImplementedError
