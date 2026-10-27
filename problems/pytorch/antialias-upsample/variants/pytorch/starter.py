import torch


def solve(x, size, mode="bilinear", antialias=False, align_corners=False):
    """抗锯齿 resize 与 trilinear 3D：语义 = F.interpolate(mode=..., antialias=..., align_corners=...) 的 CPU 行为。

    入参：
    - x: float32 / float64 张量。mode="trilinear" 时为 (N,C,D,H,W) 5-D；
      mode="bilinear"/"bicubic" 时为 (N,C,H,W) 4-D。其他维度数或 dtype 抛 ValueError。
    - size: 正整数序列，长度 2（对应 (H_out,W_out)）或 3（trilinear，对应
      (D_out,H_out,W_out)）；长度不符或含非正整数抛 ValueError。
    - mode: "bilinear" | "bicubic" | "trilinear"，其他取值抛 ValueError。
    - antialias: bool，默认 False。仅 bilinear/bicubic（4-D 输入）支持；
      antialias=True 与 trilinear 组合抛 ValueError。
    - align_corners: bool，默认 False。

    坐标映射（每维独立，n_in→n_out）：
    - align_corners=False（half-pixel）：scale = n_in/n_out，
      源中心 src = (dst+0.5)·scale − 0.5；bilinear/trilinear（线性核）下 src<0 时
      clamp 到 0，bicubic 不 clamp（负中心只影响权重相位）。
    - align_corners=True：scale = (n_in−1)/(n_out−1)，src = scale·dst，不 clamp；
      n_out=1 时该维 scale=0，即整维映射到源索引 0（实测原生如此）。

    非 antialias 权重（每维独立，tap 索引 clamp 到 [0, n_in−1]，同索引权重累加）：
    - bilinear/trilinear：2 tap，k0=floor(src)、k1=k0+1（k0=n_in−1 时 k1=k0），
      t = clamp(src−k0, 0, 1)，权重 (1−t, t)。
    - bicubic：4 tap，k0−1..k0+2（k0=floor(src)），Keys a=−0.75 的 cubic 卷积核
      （|x|<1: ((a+2)x−(a+3))x²+1；1≤|x|<2: ((ax−5a)x+8a)x−4a），x=相位距离。

    antialias=True 权重（实测原生 CPU 口径，与 PIL 下采样对齐）：
    - 仅当 scale ≥ 1（该维缩小）时加宽窗口：support = (interp_size/2)·scale，
      invscale = 1/scale；scale < 1（放大/恒等）时 support = interp_size/2、
      invscale = 1（interp_size：bilinear 2，bicubic 4）。
    - 采样中心 center = scale·(dst+0.5)（源空间中心 = center−0.5）。注意：即使
      align_corners=True 也用该 half-pixel 口径（实测原生如此），因此
      align_corners=True 的 antialias 结果与非 antialias 不同。
    - tap 范围：xmin = max(floor(center−support+0.5), 0)，tap 取
      xmin .. min(floor(center+support+0.5), n_in)−1（至多 ceil(support)·2+1 个）。
    - 权重 = filter((tap−center+0.5)·invscale)：bilinear 用三角核 1−|x|（|x|<1）；
      bicubic 用 Keys a=−0.5（PIL 口径；与非 antialias 的 a=−0.75 不同，实测原生如此，
      故 bicubic 放大时 antialias=True 与 False 结果不同）。
    - 权重按其和归一化（和为 0 的退化情形输出记 0）。
    - bilinear + align_corners=False 放大时 antialias=True 与 False 数学上等价
      （同 tap 同权重；原生两条 CPU 路径的 float32 舍入可能有 ~1e-7 级差异，
      参考实现两者逐位一致）；bicubic / align_corners=True 的放大不满足等价（原因如上）。

    数值口径：float32 输入时 scale/中心/权重按 float32 精度计算（与原生 CPU
    antialias/separable 路径一致），累加在 float64 中完成，最终回投 float32；
    float64 输入全程 float64。某维 n_out==n_in 时该维逐位透传（恒等尺寸下输出
    与输入逐位相等，antialias 两态皆然）。不使用任何随机数。

    返回：与 x 同 dtype 的张量，形状 (N,C,*size)。
    """
    # TODO
    raise NotImplementedError
