import math


def solve(image_height, image_width, max_patches, patch_height=16, patch_width=16):
    for name, v in (
        ("image_height", image_height),
        ("image_width", image_width),
        ("max_patches", max_patches),
        ("patch_height", patch_height),
        ("patch_width", patch_width),
    ):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must be an int")
        if v <= 0:
            raise ValueError(f"{name} must be positive")

    scale = math.sqrt(
        max_patches * (patch_height / image_height) * (patch_width / image_width)
    )
    num_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    resized_height = max(num_rows * patch_height, 1)
    resized_width = max(num_cols * patch_width, 1)
    total_patches = num_rows * num_cols

    # 行优先展开，row_id/col_id 偏移 +1（0 留给 padding）
    patch_ids = [
        [r + 1, c + 1] for r in range(num_rows) for c in range(num_cols)
    ]
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "total_patches": total_patches,
        "padded": max_patches - total_patches,
        "patch_ids": patch_ids,
    }
