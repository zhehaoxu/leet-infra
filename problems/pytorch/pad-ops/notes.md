# 常量 Padding 详解

## 定义

在张量各维两端补元素。`F.pad` 的 `padding` 元组顺序是**本关最大的坑**：

> **从最后一个维向前**，每两数一组 `(左, 右)`。

即 `padding=(pl, pr, pt, pb)` 对应 `(W左,W右, H上,H下)`，若还有 D 继续往前。少于维数的维默认不补。这与 `Conv2d(padding=)` 的「每维统一」不同。

## 参考解的坐标映射

```
for i in range(len(padding)//2):
    d = x.dim() - 1 - i        # 末维向前
    lefts[d], rights[d] = padding[2i], padding[2i+1]
```

输出每维长 `s + left + right`；用 `torch.full(value)` 铺底，再把原 `x` 写进 `slice(l, l+s)` 的中心窗口。

## 模式差异（本关仅 constant）

- `constant`：填给定 `value`（默认 0）。卷积输入常用 0。
- `reflect` / `replicate`：镜像/复制边缘，风格迁移、边缘敏感时避免 0 伪影。
- `circular`：环形填充。
本关约定 `mode != "constant"` 直接 `raise ValueError`——别用 `assert`（会被测试的 `except AssertionError` 误捕），要抛 `ValueError`。

## 坑

1. padding 元组长度须为偶数、且不超过 `2*ndim`。
2. 顺序写成「首维向前」或与卷积混用，行列会全反。
3. `value` 要与 dtype 匹配（整型张量给小数会被截断）。
