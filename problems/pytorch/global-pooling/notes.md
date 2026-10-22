# 全局池化（Global Pooling）详解

## 定义

对**整幅空间特征图**做归约，把 $(N,C,H,W)\to(N,C,1,1)$：

- Global Average Pooling（GAP）：$\text{out}(n,c)=\frac{1}{HW}\sum_{i,j}x(n,c,i,j)$
- Global Max Pooling（GMP）：取全图最大

空间维被完全吸收，无需展平 + 大全连接。

## 为什么重要

- ResNet 用 GAP 取代 `Flatten→FC`：参数量骤减、天然接受任意输入分辨率、抑制过拟合。
- CAM（类激活图）正是靠 GAP 的通道权重 × 空间特征图来定位目标区域。
- Transformer/ViT 也常用「global token / mean pooling」得到句向量，本质同类。

## 实现坑

1. **归约维度要按 rank 动态取**：2D 是 dim=(2,3)，3D 体素是 (2,3,4)。通用写法 `dims=tuple(range(2, x.dim()))`，别硬编码 (2,3)。
2. **keepdim=True**：保留 $(N,C,1,1)$ 便于后续 broadcast（如 CAM 加权），忘记 keepdim 会塌成 $(N,C)$ 破坏链路。
3. `amax` 而非 `max`（后者返回索引且要再取 `[0]`）。

## 与 adaptive 池化关系

GAP 是 `adaptive_avg_pool2d(x, 1)` 的特例；也是 avgpool 把 kernel 设为整图、stride 任意的退化情形。
