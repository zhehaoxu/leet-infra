import math


def solve(width, height, base_size=1024, image_size=640, cropping=True,
          min_crops=2, max_crops=6, patch_size=16, downsample_ratio=4):
    """DeepSeek-OCR 五档动态分辨率与 tile 账（DeepSeek-OCR-vllm
    process/image_process.py 的 dynamic_preprocess + tokenize_with_images
    视觉 token 序列账的标量无状态化）。

    五档配置（config.py）：Tiny 512/512、Small 640/640、Base 1024/1024、
    Large 1280/1280 均 crop_mode=False；Gundam 1024/640 crop_mode=True。

    官方口径要点：
      - 小图门：cropping 且 width <= image_size 且 height <= image_size
        （and，一维越界即切）-> crop_ratio (1,1)；cropping=False 直接 (1,1)
      - ratio 集 {(i,j): n∈[min,max], 1<=i,j<=n, min<=i*j<=max}——i/j
        上界是循环变量 n 而非 max_num；sorted 按乘积升序
      - 最近匹配：diff = |w/h - i/j| 严格 < 更新；diff 相等时 area 抢位
        （area > 0.5*image_size^2*i*j 即改选当前 ratio）
      - tile 切分：resize 拉伸到 (image_size*i, image_size*j)（非等比）；
        box 行优先 x0=(k%nw)*image_size, y0=(k//nw)*image_size
      - token 账：q = ceil((image_size//patch)/down)；qb 同 base_size；
        全局 = (qb+1)*qb + 1；nw>1 或 nh>1 时追加局部 (q*nw+1)*(q*nh)
        （README 的 64/100/256/400 是 qb^2 纯编码器口径，序列账多行 sep）

    返回 dict（键固定）：crop_ratio [nw, nh] / tiles / tile_boxes（行优先
    [x0,y0,x1,y1]）/ global_tokens / local_tokens / num_image_tokens。
    红线：不出 PIL 图像；纯标量账。
    """
    raise NotImplementedError
