# 张量形状与索引操作 详解

## 这题练什么

四个高频「变形 / 切片」操作，串起 PyTorch 里最容易踩的**视图 vs 拷贝、stride、维度对齐**三件事。`op` 为字符串分发，非法值抛 `ValueError`。

## 逐个口径

| op | 参考解 | 要点 |
|----|--------|------|
| `flip` | `torch.flip(x, dims=[-1])` | 末维反转。切片 `x[:, ::-1]` 在本版本不支持负步长；且切片得到的是**负 stride 视图**，很多后续算子要求连续内存。`torch.flip` 直接产出连续新张量。 |
| `narrow` | `x[:, : x.shape[1] // 2]` | 沿列方向取前半（宽度整除 2）。这是「窄化视图」，零拷贝、与原张量共享 storage。 |
| `concat` | `torch.cat([x, x + 1], dim=1)` | 沿 dim=1 拼接 `x` 与 `x+1`，列数翻倍。`cat` 会分配新张量并拷贝。 |
| `reshape` | `x.reshape(x.shape[1], x.shape[0])` | 交换前两维尺寸做总元素数不变的重排（非转置；方阵时才像 transpose）。可能返回视图也可能拷贝，取决于内存布局。 |

## flip / narrow / reshape 的视图陷阱

- **narrow / reshape 常给视图**：改它会透过影响原张量；flip 给拷贝。需要独立副本时 `.contiguous()` / `.clone()`。
- **transpose 得到的非连续张量不能直接 `reshape`**：`x.transpose(...).reshape(...)` 会隐式 copy；想避免隐藏拷贝用 `.view()` 会直接报错提醒你。
- 维数不确定时用 `x.reshape(-1, ...)` 让某一维自动推。

## 非法 op

必须 `raise ValueError`——判题用 `pytest.raises` 校验负路径，返回 None 或不抛都会失败。

## 关联

broadcast-ops 讲 unsqueeze 对齐；concat-stack-split 是本题 `cat` 的家族；stride/contiguous 概念与 grid-sample、unfold 等题相通。
