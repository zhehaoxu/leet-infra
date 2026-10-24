# 余弦相似度（Cosine Similarity）

## 题目描述

余弦相似度衡量两个向量的**方向**接近程度，与长度无关：

$$ \cos(a,b) = \frac{a\cdot b}{\lVert a\rVert_2\,\lVert b\rVert_2} \in [-1, 1] $$

它是检索/推荐（embedding 召回）、对比学习（InfoNCE/SimCLR 的正负样本打分）、
语义相似度（sentence-BERT）里的默认度量。请实现它，并正确处理零向量的除零保护。

## 规则

- 实现 `solve(a, b, dim=-1, eps=1e-8)`，`a`、`b` 形状相同（或可广播）。
- 沿 `dim` 计算点积与各自信道模长，输出形状为该维被移除后的形状。
- 分母用 `clamp(min=eps)` 保护（与 `F.cosine_similarity` 一致），避免零向量除零。
- 结果须与 `F.cosine_similarity(a, b, dim=dim, eps=eps)` 一致。

## 示例

```
输入：a = [1, 2, 3], b = [4, 5, 6]
  a·b = 4+10+18 = 32
  ‖a‖ = √14 ≈ 3.7417, ‖b‖ = √77 ≈ 8.7749
  cos = 32 / (3.7417 × 8.7749) ≈ 0.97463
输出：[0.97463]
```

## 约束

- float32 输入；输出在 `[-1, 1]`
- 自身与自身的相似度应为 1（正数倍缩放不改变方向）
