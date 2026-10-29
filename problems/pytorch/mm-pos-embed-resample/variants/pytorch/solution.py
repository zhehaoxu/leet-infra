import torch


def _axis_taps(source, target):
    """Return clamped indices and Keys weights at unclamped half-pixel centers."""
    u = (torch.arange(target, dtype=torch.float64) + 0.5) * (source / target) - 0.5
    taps = torch.floor(u).to(torch.int64)[:, None] + torch.tensor([-1, 0, 1, 2])
    distance = (u[:, None] - taps).abs()
    a = -0.75
    inner = ((a + 2) * distance - (a + 3)) * distance.square() + 1
    outer = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
    weights = torch.where(distance <= 1, inner, torch.where(distance < 2, outer, 0.0))
    return taps.clamp(0, source - 1), weights


def solve(pos_embed, old_hw, new_hw, num_prefix_tokens=1):
    """Explicit non-AA bicubic interpolation of the spatial tokens only."""
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    prefix_count = num_prefix_tokens
    if pos_embed.shape[1] != prefix_count + old_h * old_w:
        raise ValueError("old_hw does not match the number of spatial tokens")
    if old_h == new_h and old_w == new_w:
        return pos_embed.clone()

    prefix = pos_embed[:, :prefix_count]
    grid = pos_embed[0, prefix_count:].reshape(old_h, old_w, -1).to(torch.float64)
    iy, wy = _axis_taps(old_h, new_h)
    ix, wx = _axis_taps(old_w, new_w)
    result = torch.zeros((new_h, new_w, grid.shape[-1]), dtype=torch.float64)
    # Keep all 16 contributions, including repeated boundary indices.
    for y_tap in range(4):
        for x_tap in range(4):
            values = grid[iy[:, y_tap, None], ix[None, :, x_tap]]
            weight = wy[:, y_tap, None] * wx[None, :, x_tap]
            result += values * weight[..., None]
    spatial = result.reshape(1, new_h * new_w, -1).to(pos_embed.dtype)
    return torch.cat((prefix, spatial), dim=1)
