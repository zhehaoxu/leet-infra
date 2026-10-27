# 冷门激活：softsign/hardtanh/celu/log-sigmoid

## 题目描述

ReLU/ELU/GELU 之外，torch 激活列表里还有一批"分段/有界"的常客：softsign 是 tanh 的有理替代、hardtanh 是带硬饱和的 clamp、celu 是把 alpha 放进指数分母的 ELU 变体、log-sigmoid 是 BCE 损失的数值地基。本题实现一个逐元素激活分派器，四种 mode：

\[\mathrm{softsign}(x)=\frac{x}{1+|x|},\qquad
\mathrm{hardtanh}(x)=\mathrm{clamp}(x,\ \text{min},\ \text{max}),\]
\[\mathrm{celu}(x;\alpha)=\max(0,x)+\min\big(0,\ \alpha\,(e^{x/\alpha}-1)\big),\qquad
\mathrm{logsigmoid}(x)=-\,\mathrm{softplus}(-x).\]

口径与 torch 官方文档一致（F.softsign / F.hardtanh / F.celu / F.logsigmoid）。数值实现是本题重点：celu 在大 alpha 下朴素 \(e^{x/\alpha}-1\) 会灾难性抵消，必须用 expm1 加高精度中间计算；logsigmoid 必须用 \(-\mathrm{softplus}(-x)\) 稳定式，朴素 \(\log(\mathrm{sigmoid}(x))\) 在大负值处下溢成 \(-\infty\)。逐元素作用、无随机数。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。mode 取 `"softsign"` / `"hardtanh"` / `"celu"` / `"logsigmoid"` 之一，其余取值抛 ValueError；输入是 float 张量（float32/float64 均支持）、形状任意，逐元素作用，返回与输入同形状、同 dtype 的张量。
- **softsign**：\(x/(1+|x|)\)；min_value/max_value/alpha 一概忽略。任何有限输入都严格落在 \((-1,1)\) 内、逐元素保号。
- **hardtanh**：clamp 到 [min_value, max_value]（默认下界 −1.0、上界 1.0）；alpha 忽略。饱和段输出**恰等于**边界值（判题要求精确相等，不是近似）。
- **celu**：\(\max(0,x)+\min(0,\alpha(e^{x/\alpha}-1))\)，约定 \(\alpha>0\)；min_value/max_value 忽略。数值实现须与该公式数学恒等且**恒有限**：可用 expm1 与高精度（float64）中间计算，避免大 alpha 下 \(e^{x/\alpha}-1\) 的灾难性抵消。
- **logsigmoid**：\(-\mathrm{softplus}(-x)\)。**必须**用该稳定式，禁止 \(\log(\mathrm{sigmoid}(x))\) 直算：\(x\le-104\) 时 fp32 sigmoid 下溢为 0、\(\log(0)=-\infty\)；softplus 形式恒有限（x 很负时 softplus(−x) 走线性段 \(\approx-x\)，结果 \(\approx x\)）。

## 示例

输入 \(x=[-2,\,-0.5,\,0,\,0.5,\,2]\)，四种 mode 均取默认参数（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
softsign   [-0.6666666865, -0.3333333433,  0.0, 0.3333333433,  0.6666666865]
hardtanh   [-1.0,          -0.5,           0.0, 0.5,           1.0         ]
celu(α=1)  [-0.8646647334, -0.3934693336,  0.0, 0.5,           2.0         ]
logsigmoid [-2.1269280910, -0.9740769863, -0.6931471825, -0.4740769863, -0.1269280165]
```

其中 celu 的 \(-0.8646647=-\,(1-e^{-2})\)（负支 \(\alpha(e^{x/\alpha}-1)\) 在 \(\alpha=1\) 时即 \(e^x-1\)），logsigmoid(0) \(=-\ln2=-0.6931471825\)。

## 约束

- mode 非法抛 ValueError；输出形状/dtype 与输入一致；float64 下四种 mode 与 torch 参考对拍容差 1e-9，float32 为 1e-6
- 判别探针直接断言：softsign 传入非默认的 min/max/alpha 时结果不变，输出严格落在 \((-1,1)\) 且保号（在 \(\pm10^4\) 处 |y|>0.999 但仍 <1）；hardtanh 饱和段与边界 torch.equal（\([-7,-1,0,1,7]\to[-1,-1,0,1,1]\)）、自定义边界生效且 alpha 被忽略；celu 与 F.celu 在 \(\alpha\in\{0.5,1.0,2.5\}\) 一致、**\(\alpha=10^6\) 时整支退化成恒等**（朴素 fp32 实现小 |x| 处负支整体归零、其余点漂移到 1e-2 量级）；logsigmoid 在 \(\{-120,-104,-50,0,50,104,120\}\) 全部有限且与 F.logsigmoid 一致（朴素公式在 −120/−104 处实测非有限）
