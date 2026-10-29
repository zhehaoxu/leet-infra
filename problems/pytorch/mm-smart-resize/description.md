# Smart Resize 对齐与视觉 Token 预算

## 场景

视觉编码器先按像素 patch 切图，再将相邻 patch 合并为语言模型的视觉槽位。给定原图高宽、patch 边长、merge 边长和面积预算，计算单张图的目标尺寸、patch 网格与槽位数；不执行像素插值。

本题采用 **Hugging Face transformers 5.20.2 的 Qwen2-VL `smart_resize`** 计算顺序。它与独立 `qwen-vl-utils` 的下限处理和默认预算不同，不应混用。

## 尺寸规则

令原尺寸为 $H,W$，patch 边长为 $p$，merge 边长为 $m$，对齐因子 $f=pm$，面积下限、上限分别为 $L,U$。

先计算

$$
h=f\operatorname{round}(H/f),\qquad w=f\operatorname{round}(W/f).
$$

`round` 使用 Python 的 **ties-to-even**，例如 `round(2.5)=2`、`round(3.5)=4`。

- 若 $hw>U$，令 $\beta=\sqrt{HW/U}$，重新计算
  $$
  h=\max(f,f\lfloor H/\beta/f\rfloor),\qquad
  w=\max(f,f\lfloor W/\beta/f\rfloor).
  $$
- 否则若 $hw<L$，令 $\beta=\sqrt{L/(HW)}$，重新计算
  $$
  h=f\lceil H\beta/f\rceil,\qquad w=f\lceil W\beta/f\rceil.
  $$
- 否则保留首次对齐结果。

缩放因子使用**原始面积** $HW$，不是首次对齐后的面积。只选择一个分支，不迭代修补预算。为对齐上游实现，按上式顺序使用 Python `float`、`math.sqrt`、`floor`、`ceil`，不换成精确有理数开方。

## 网格与预算账

返回一个字典：

- `resized_hw`：Python 整数元组 $(h,w)$。
- `grid_thw`：Python 整数元组 $(1,h/p,w/p)$。这里只处理单张图，没有视频时间分组。
- `num_patches`：Python 整数 $(h/p)(w/p)$。
- `num_visual_tokens`：Python 整数 `num_patches / m²`。
- `in_budget`：Python 布尔值，等于 $L\le hw\le U$。

**不要假定 `in_budget` 总为真。** 对齐、每边最小值和一次性分支可能使最终面积不满足预算。例如原图 $28\times5600$、$p=14,m=2,L=U=3136$，结果为 $28\times784$，预算检查为假。也可能缩小后低于下限。

## 输入范围

所有参数须为正 Python 整数，不含 `bool`。保证高宽不超过 $10^6$，patch/merge 不超过 128，预算不超过 $10^{12}$。以下情形抛出 `ValueError`：参数不是正整数、$f^2\le L\le U$ 不成立，或长边/短边 **大于** 200；恰好 200 合法。

## 示例

原尺寸 `(560, 1120)`，`patch_size=14`，`merge_size=2`，`min_pixels=3136`，`max_pixels=156800`。

缩小后得到

```text
{
  'resized_hw': (280, 560),
  'grid_thw': (1, 20, 40),
  'num_patches': 800,
  'num_visual_tokens': 200,
  'in_budget': True
}
```
