import torch


# 参数契约 / Parameter contract
# image: CPU uint8 RGB 张量 [H,W,3]，1<=H,W<=128，可非连续；不得修改。
#        CPU uint8 RGB tensor [H,W,3], 1<=H,W<=128; strides may be noncontiguous.
# crop_size: (h,w)，两个 Python 正整数（不含 bool），h<=H、w<=W。
#            A tuple/list of two positive Python integers (not bool), h<=H, w<=W.
#            非法类型、长度、元素或越界尺寸须抛 ValueError；不做 padding。
#            Invalid crop type, length, entries or bounds must raise ValueError; no padding.
# mean/std: 各为 3 个 Python 数值的 RGB 列表；|mean|<=10**4，10**-4<=std<=10**4。
#           Three-number RGB lists with |mean|<=10**4 and 10**-4<=std<=10**4.
# 其他输入保证合法；无需其他校验。All other inputs are valid; no other validation required.
# 返回 / Return: CPU float32 contiguous [3,h,w]；不得修改任何输入 / do not modify inputs.
# 先从 ((H-h)//2,(W-w)//2) 裁剪，再转 float32，/255，减 mean，除 std。
# mean/std 也转 float32，各步按 float32 运算。
# Crop at ((H-h)//2,(W-w)//2), cast to float32, divide by 255, subtract mean,
# then divide by std. Convert mean/std to float32; arithmetic is float32.
# 输出不 clamp。不做 resize，不得使用 PIL/torchvision；无需反向传播。
# Do not clamp the output, resize, or use PIL/torchvision. No backward pass required.
def solve(image, crop_size, mean, std):
    """中心裁剪并按 RGB 归一化 / Center-crop and normalize RGB channels."""
    raise NotImplementedError
