import math


def solve(height, width, patch_size, merge_size, min_pixels, max_pixels):
    values = (height, width, patch_size, merge_size, min_pixels, max_pixels)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("dimensions, patch sizes and budgets must be positive integers")
    factor = patch_size * merge_size
    if not factor * factor <= min_pixels <= max_pixels:
        raise ValueError("budgets must satisfy factor**2 <= min_pixels <= max_pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio must not exceed 200")
    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / scale / factor) * factor)
        resized_w = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * scale / factor) * factor
        resized_w = math.ceil(width * scale / factor) * factor
    grid_h, grid_w = resized_h // patch_size, resized_w // patch_size
    return {
        "resized_hw": (resized_h, resized_w),
        "grid_thw": (1, grid_h, grid_w),
        "num_patches": grid_h * grid_w,
        "num_visual_tokens": grid_h * grid_w // (merge_size * merge_size),
        "in_budget": min_pixels <= resized_h * resized_w <= max_pixels,
    }
