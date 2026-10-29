# 题解：CLIP 中心裁剪与像素归一化

## 真实来源与范围

本题的裁剪及归一化约定来自 **Hugging Face Transformers 的共享图像变换工具**，不是凭 CLIP 名称推定的预处理流程。已核对本仓库环境中 `transformers==5.20.2` 的文件：

- `.venv-eval/lib/python3.12/site-packages/transformers/image_transforms.py`：`center_crop` 第 446 行起，偏移公式见第 493–498 行；`normalize` 第 385 行起，类型与通道参数转换见第 418–440 行。
- 同文件 `rescale` 第 90–125 行：默认输出 float32，但内部先用 float64 乘缩放因子，再回投 float32。`normalize` 对已有浮点输入保留其类型，因此在默认 rescale 之后走 float32，并非所有调用都强制 float32。
- [上游源码入口](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py)。上述版本与行号以本地核对文件为准，上游 main 可能变化。

**本题是限定范围的教学实现，不是完整 CLIP/CLIPImageProcessor 的精确复刻。** 输入固定为 CPU uint8 HWC RGB，只覆盖合法区域的中心裁剪、除以 255、逐通道归一化和布局转换；没有 resize、PIL、颜色转换或 padding。HF 原工具在图像不足时可补边，本题明确抛 ValueError。本题约定先转 float32 再除以 255，不照搬 HF rescale 内部的 float64 乘法流程，也不承诺跨后端逐位一致。

中心裁剪采用 HF 的向下取整，而不是 torchvision 常见的 `int(round(diff/2))`。本地标量实跑中，差值为 3 时 `3//2` 得 1，`round(3/2)` 得 2；测试专门覆盖两轴及单轴差 3，避免偏移规则混用。这里没有调用 torchvision，亦不宣称复刻其完整裁剪接口。

## 实现步骤

1. 仅校验裁剪参数：必须是两个正整数并落在源图范围内；非法类型、长度、元素或越界尺寸抛 ValueError。其余输入按 starter 保证合法，不添加额外错误契约。
2. 分别计算高度、宽度的 floor 偏移，切出 HWC 子区域。
3. 转成独立的 float32 张量；均值和标准差按 RGB 顺序构造 float32 向量。让尾部通道轴广播，依次执行除以 255、减均值、除标准差。
4. `permute(2, 0, 1)` 后调用 `contiguous()`。仅改变维度顺序并不保证 CHW 连续存储。

转成 float32 已与原 uint8 存储分离，后续计算不写回原图。不要先在 uint8 上相减，不要在归一化后再除以 255，也不要交换 RGB/BGR 或把输出限制在像素范围内。

对裁剪区域而言，时间与输出空间均为 $O(hw)$，通道数固定；无需复制裁剪区域以外的像素。

## 独立验证与实跑样例

测试 oracle 不导入参考解、不复用其切片/广播路径：先把输入转为 Python 嵌套列表，按通道、行、列逐像素读取，用 Python float 运算，并以 `struct.pack/unpack` 将缩放、相减、相除各阶段舍入为 float32，最后构造期望张量。浮点比较使用 `atol=rtol=1e-6`；另查 dtype、形状、CPU、连续性、输入及参数不变性。

本地 PyTorch **2.14.1** 环境下，`add_problem.py --selftest mm-clip-preprocess --mode submit` 得到 **12/12 ACCEPTED**。覆盖差 3、矩形、完整裁剪、不同通道、0/255、非连续及跨步通道、单轴/单像素、float32 舍入、非法裁剪、最大尺寸和固定 `torch.Generator` 的随机输入。`check_example.example=True`。

题面小图的实跑输出（按 R/G/B 平面）为：

```text
R = [[-1.0, 1.0], [1.0, -1.0]]
G = [[ 1.0, 0.0], [1.0,  0.0]]
B = [[-2.0, 2.0], [2.0, -2.0]]
```

这些数值由参考实现运行记录，随后通过独立 oracle 校验；没有将完整 HF 处理器当成测试黑盒，也没有下载或运行 CLIP 模型。
