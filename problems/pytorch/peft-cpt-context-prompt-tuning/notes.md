# 解题笔记 — CPT

## 官方账

- HF PEFT `peft/tuners/cpt/model.py`（`CPTEmbedding`）；论文 arXiv 2410.17222。
- 判题域：mask 族解析、ε 半径、L2 球投影、嵌入组合、decay 加权损失。hook/gather/词表 I/O 不在域内。

## 踩坑

- **ε 全程 fp32**：官方 `torch.Tensor([...])` / `torch.Tensor(mask)` 默认 fp32，锚值要用 fp32 表示（0.05→0.05000000074505806、1e-10→1.000000013351432e-10），容差 1e-12/1e-18 级别。
- **shift 语义易错**：labels[...,1:] 与 mask[...,1:] 同步 shift；valid 位置数 = t-1 中非 -100 个数；组 id 查找用 **shift 后**的 mask（漏 shift 变异 CAUGHT-ERR 由 mask 形状差异兜住）。
- **decay 倒序**：unique（升序）后 flip——最大组 id 权重 1，之后乘 factor。正向遍历是另一变异点（权重互换）。
- **投影只收缩不放大**：`clamp(norm, min=eps)` 保证 scale ≤ 1；`clamp(max=eps)` 变异会把小 norm 行放大到 ε 球面，被 norms 断言抓。
- **真等价三类**（剔除）：非负整数域 `(m>0)&(m%4==k)` ≡ `m%4==k`（0%4=0 不落 1/2/3）；零行在 `>=0` mask 下缩放因子恒 1（eps/eps=1）；加法交换律。
- **proj-p1 幸存根因**：主 delta 无零行时 `new_w[mask] *= (k,1)` 与 `new_w *= (n,1)` 广播相同——主用例必须含一行精确零，使 k≠n 触发广播错误。
- test 顶部 `set_default_dtype(float64)` 必须在 import F 之前；CE 的 ignore_index=-100 位置返回 0 而非排除，与官方 `loss[bool]` 过滤后再 mean 语义不同，手算对拍要按 valid 位置切片。

## mut 14 变异清单

mask：%4==1→2、view 转置；eps：format/input 互换、归一化分母 1024、底座 MIN→0；proj：clamp min→max（放大）、全行缩放（广播）、`>0` 掩码已证真等价剔除；embed：丢 delta、换序（真等价剔除）；loss：mask 漏 shift、ignore 失效、倒序→正序、factor 翻倍、type 判反、mean→sum。14/14 全 CAUGHT。
