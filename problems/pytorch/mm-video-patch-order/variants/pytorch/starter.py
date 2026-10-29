import torch


def solve(video, temporal_patch_size, patch_size, merge_size):
    """视频时空 patch 的 processor 重排 / Processor-side video patch ordering.

    中文契约：
    - video：CPU 张量 [T,C,H,W]，dtype 为 int64/float32/float64。
      1<=T<=9，1<=C<=4，1<=H,W<=32；输入可以不连续。
    - temporal_patch_size=tau、patch_size=p、merge_size=m 均为 1..4 的整数。
      H、W 必须都能被 p*m 整除，否则抛出 ValueError；其余输入保证合法。
    - 时间长度不足 tau 的整数倍时，重复最后一帧补齐，不能补零。
      gt=ceil(T/tau)，gh=H/p，gw=W/p，N=gt*gh*gw。
    - 返回 (patches, grid_thw, coords)。patches 为同 dtype 的 CPU 张量
      [N,C*tau*p*p]；grid_thw 是 Python 整数元组 (gt,gh,gw)；coords
      为 CPU int64 [N,3]，每行是该 token 的 (t,h,w) patch 网格坐标。
    - token 顺序：t、block_h、block_w、inner_h、inner_w；h=block_h*m+
      inner_h，w=block_w*m+inner_w。每个 token 内顺序：C、tau、p_h、p_w。
      各轴均从小到大，后列出的轴变化更快。
    - 这是重排而非 Conv3d，不求均值、不投影、不实际合并 token。
      后续 LLM 合并槽数为 N/m^2，但不返回；不得修改输入。

    English contract:
    - video is a possibly noncontiguous CPU tensor [T,C,H,W], with dtype
      int64, float32, or float64; 1<=T<=9, 1<=C<=4, 1<=H,W<=32.
    - tau=temporal_patch_size, p=patch_size, m=merge_size are integers in
      1..4. Raise ValueError if H or W is not divisible by p*m.
      All other inputs are valid.
    - Repeat the last frame until time is divisible by tau; never zero-pad.
      gt=ceil(T/tau), gh=H/p, gw=W/p, N=gt*gh*gw.
    - Return (patches, grid_thw, coords): same-dtype CPU patches of shape
      [N,C*tau*p*p], a Python integer tuple (gt,gh,gw), and CPU int64 coords
      [N,3] containing patch-grid (t,h,w), not pixel coordinates.
    - Tokens traverse t, block_h, block_w, inner_h, inner_w, with
      h=block_h*m+inner_h and w=block_w*m+inner_w. Within a token, traverse
      C, tau, p_h, p_w. The last listed axis varies fastest.
    - This is processor rearrangement, not Conv3d, averaging, projection,
      or token merging. N/m^2 later LLM slots are not returned.
      Do not mutate the input.
    """
    raise NotImplementedError
