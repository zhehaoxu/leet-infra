import torch


def solve(image, crop_size, mean, std):
    """Floor-centered crop followed by float32 RGB normalization."""
    height, width = image.shape[:2]
    if (
        not isinstance(crop_size, (tuple, list))
        or len(crop_size) != 2
        or any(type(v) is not int or v <= 0 for v in crop_size)
    ):
        raise ValueError("crop_size must contain two positive integers")
    crop_h, crop_w = crop_size
    if crop_h > height or crop_w > width:
        raise ValueError("crop_size must fit inside the image")

    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    pixels = image[top:top + crop_h, left:left + crop_w].to(torch.float32)
    means = torch.tensor(mean, dtype=torch.float32)
    stds = torch.tensor(std, dtype=torch.float32)
    pixels = (pixels / 255.0 - means) / stds
    return pixels.permute(2, 0, 1).contiguous()
