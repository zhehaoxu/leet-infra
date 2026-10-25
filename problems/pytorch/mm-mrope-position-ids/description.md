# 多模态：Qwen2-VL 风格 M-RoPE 位置编号

## 题目描述

Qwen2-VL 用 **M-RoPE（Multimodal Rotary Position Embedding）** 把位置编号拆成 \((t,h,w)\) 三个分量，让图像 token 在高度/宽度方向各占一维、文本 token 三轴同步递增，从而在同一序列里同时编码 1-D 文本与 2-D 视觉位置。本题实现一个**只含图像与文本（无视频）**的简化、确定性版本。

规则（自一个「运行计数器」`running` 出发，从左到右扫序列）：
- **文本 token**：三轴编号都取 `running`，即 \((t,h,w)=(r,r,r)\)；随后 `running += 1`。
- **图像 token**：一段**连续**的 \(h\times w\) 个图像 token 一次处理（\(h,w\) 由该图的网格给出）。第 \(k\) 个（行主序，\(r=k\,/\!/ \,w,\ c=k\,\%\,w\)）编号为
  \[ (t,h,w)=(\text{running},\ \text{running}+r,\ \text{running}+c) \]
  即整块 \(t\) 固定、块内按行/列偏移。处理完整张图后 `running += max(h, w)`。

## 规则（接口）

实现 `solve(mm_token_type, grid_thw)`。`mm_token_type`（int，\((L,)\)，1=图像 token、0=文本 token，图像 token 必成连续块）；`grid_thw`（int，\((M,3)\)，按出现顺序给出每张图的 \((t,h,w)\)，本题 \(t\) 恒供 1、只用 \(h,w\)）。返回 int64 张量 \((3, L)\)，三行依次是 \(t/h/w\) 位置编号。

## 示例

序列 `文本 文本 [2×3 图像] 文本`，`mm_token_type=[0,0,1,1,1,1,1,1,0]`、`grid_thw=[[1,2,3]]`：

```
t: [0, 1, 2, 2, 2, 2, 2, 2, 5]
h: [0, 1, 2, 2, 2, 3, 3, 3, 5]
w: [0, 1, 2, 3, 4, 2, 3, 4, 5]
out shape = (3, 9)
```

## 约束

- 图像 token 连续成块、与 `grid_thw` 顺序一一对应，块大小 \(h\cdot w\) 与 `mm_token_type` 中 1 的数量一致。
- 纯整数编号构造，无浮点。
