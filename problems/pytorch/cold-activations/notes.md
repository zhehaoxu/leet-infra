# 详解：冷门激活：softsign/hardtanh/celu/log-sigmoid

## 出处

- 四个算子的口径全部取 torch 官方文档：`torch.nn.functional.softsign` / `hardtanh` / `celu` / `logsigmoid`（含 hardtanh 的 min_value/max_value 边界参数与 celu 的 alpha 参数语义）。
- softsign 与 celu 的原始论文出处不在本题已核实清单内，标**待回源**；本题只写 torch 文档口径，不引任何未核实编号。
- logsigmoid 的稳定式与 softplus 的阈值语义（beta=1、z>20 走线性段）同 torch 文档口径；F.logsigmoid 自身即按 \(-\mathrm{softplus}(-x)\) 路线实现。

## 完整推导与下标约定

逐元素算子，无下标约定；形状任意、四种 mode 都不改变形状与 dtype。按 mode 逐个推导。

**softsign 的形状**：\(x/(1+|x|)\)。保号显然（分母恒正）；值域：\(x\ge0\) 时 \(y=x/(1+x)=1-\frac{1}{1+x}<1\)，\(x<0\) 时 \(y=x/(1-x)=-1+\frac{1}{1-x}>-1\)——**任何有限输入都严格落在 \((-1,1)\)**，且以 \(1-1/|x|\) 的速度缓慢逼近边界；对比 hardtanh 在 \(|x|\ge1\) 后**精确**等于 \(\pm1\)。它也是 tanh 的有理替代：无指数、饱和慢得多（tanh(4)=0.9993，softsign(4)=0.8）。判别探针在 \(\pm10^4\) 处验证 |y|>0.999 且仍严格 <1。

**hardtanh 的三段结构**：clamp 即 \(\min(\max(x,\text{min}),\text{max})\)，开区间 (min,max) 内斜率 1、界外斜率 0——比 sigmoid/tanh 便宜（无 exp）、梯度既不消失也不爆炸，是 ReLU 的"有界版"。判题对饱和段要求**精确相等**（torch.equal：\([-7,-1,0,1,7]\to[-1,-1,0,1,1]\)）：clamp 的输出就是边界浮点值本身，没有任何运算，因此任何近似饱和（如乘性压缩）都过不了。

**celu 的分段与大 alpha 极限**：正支 \(\max(0,x)\) 是恒等映射；负支 \(x<0\) 时 \(\alpha(e^{x/\alpha}-1)<0\)，在 \(x=0\) 处值为 0、导数 \(e^0=1\)，与正支 C¹ 衔接。与 ELU 的差别在 alpha 的位置：ELU 负支是 \(\alpha(e^x-1)\)（alpha 在指数外、负支形状固定），celu 是 \(\alpha(e^{x/\alpha}-1)\)（alpha 在分母）——于是有**恒等极限**：

\[\alpha\big(e^{x/\alpha}-1\big)=x+\frac{x^2}{2\alpha}+O(\alpha^{-2})\ \xrightarrow{\alpha\to\infty}\ x,\]

alpha 越大负支越直，整个函数退化为恒等。\(\alpha=10^6\) 的判别用例正踩这条线：\(\mathrm{celu}(-5;10^6)\) 的真值 \(-4.9999875\)（与 −5 差 \(1.25\times10^{-5}\)），判题容差 atol=1e-4 与 F.celu 的 1e-6 都以此为锚。

**celu 的数值陷阱（本题核心）**：朴素 fp32 计算 \(e^{x/\alpha}-1\) 在 \(\alpha=10^6\) 时两种方式出错：

1. **负支整体归零**：\(|x/\alpha|\lesssim3\times10^{-8}\)（即 \(|x|\lesssim0.03\)）时 \(e^z\) 在 fp32 里舍入成**恰好的 1.0**，\(e^z-1\) 算出 0，整支消失——实测 \(x=-10^{-3}\) 处朴素输出 0.0，而真值 \(-0.001\)，误差恰为整个负支（100%）。
2. **抵消吃精度、乘 \(\alpha\) 放大**：1 附近 fp32 的表示间隔是 \(2^{-24}\approx5.96\times10^{-8}\)，而 \(e^{-5\times10^{-6}}\) 距 1 只有 \(5\times10^{-6}\)——减 1 把有效数字一次抵消，残差乘 \(\alpha=10^6\) 后成为可见绝对误差：实测 \(x=-5\) 处朴素给出 \(-5.00679\)（真值 \(-4.9999875\)，误差 \(6.8\times10^{-3}\)），\(x=-0.5\) 处误差 \(2.3\times10^{-2}\)。

两步修复：① \(e^z-1\) 换成 `expm1(z)`（在 z→0 处保持相对精度）；② 中间量抬到 float64 再 cast 回，消除 \(z\) 本身与 \(\times\alpha\) 环节的残余舍入。稳定实现与 F.celu 在 atol=1e-6 下逐点一致，朴素实现连 α=1e6 的恒等判别（atol=1e-4）都过不了。

**logsigmoid 的稳定式**：\(\log\sigma(x)=x-\log(1+e^x)\) 两个方向都病态：x 大正时 \(\log(1+e^x)\) 溢出、x 大负时 \(\sigma(x)\) 下溢。恒等式

\[\log\sigma(x)=-\log\big(1+e^{-x}\big)=-\mathrm{softplus}(-x)\]

把两个方向的病态都转移到 softplus(−x)：\(x\to-\infty\) 时输入 \(-x\to+\infty\)，softplus 走线性段（torch 的 softplus 在 z>20 时直接返回 z），结果 \(\approx x\) 精确有限；\(x\to+\infty\) 时 softplus(−x)→0，下溢无害。按题面口径 \(x\le-104\) 时 fp32 sigmoid 下溢为 0；判题探针取 −120 与 −104 实测朴素式确为 −inf、稳定式给出恰 \(-120\) 与 \(-104\)（线性段）。**线性段不是可选项**：softplus(50) \(=50+2\times10^{-22}\)，fp32 里 \(2\times10^{-22}\) 不可表示，函数返回恰 50.0，故 \(\mathrm{logsigmoid}(-50)=-50.0\) 与 F.logsigmoid 逐位一致。

## 核心与易错点

1. **logsigmoid 绝不 log(sigmoid(x))**：−inf 探针专抓；用 \(-\mathrm{softplus}(-x)\)，且 softplus 要带线性段阈值（z>20 直接返回 z）。
2. **celu 用 expm1 + float64 中间量**：朴素 fp32 在 α=1e6 下要么负支整体归零（小 |x|，\(e^z-1\) 舍入成 0）、要么抵消后残差被 α 放大（大 |x|）——α=1e6 判别用例两种病态都踩。
3. **忽略语义**：softsign 忽略 min/max/alpha、hardtanh 忽略 alpha、celu 忽略 min/max；把"忽略"实现成"校验/使用"会挂在传非默认参数的对拍上。
4. **hardtanh 饱和段精确等于边界**：clamp 本身无舍入，别画蛇添足做近似饱和。
5. **softsign 是严格开区间**：任何有限 x 都 |y|<1；按 tanh 的直觉写成"饱和到 ±1"就错了。
6. **mode 分派与错误面**：未知 mode 抛 ValueError；dtype/形状逐元素保持（float64 一致性对到 1e-9）。
7. **四式各自的"邻题影子"**：softsign ↔ tanh（有理版）、hardtanh ↔ ReLU（有界版）、celu ↔ ELU（alpha 进分母）、logsigmoid ↔ softplus（对数侧）——写混公式是最低级也最常见的翻车点。

## 与邻近题的区别

- **vs `relu` / `leaky-relu`**：ReLU 负支是 0/线性斜率、下方无界；celu 负支有下界 \(-\alpha\) 且在原点 C¹ 光滑衔接；hardtanh 是"有界 ReLU"（中段斜率 1、两端硬饱和）。
- **vs `elu` / `selu`**：ELU 负支 \(\alpha(e^x-1)\)——alpha 在指数外，负支形状不随 alpha 变直；celu 的 \(\alpha(e^{x/\alpha}-1)\) 把 alpha 放进分母，大 alpha 时负支趋线性（恒等极限），ELU 无此性质。selu 另带固定 \((\alpha,\lambda)\) 的自归一化缩放。
- **vs `sigmoid`**：logsigmoid 是 sigmoid 的"对数侧"读数（BCE 的数值地基）；sigmoid 本身不在这题里，但 logsigmoid 的稳定式正是为了让 \(\log\sigma(x)\) 永不下溢。
- **vs `mish` / `swish` / `hardsigmoid` / `hardswish`**：同属平滑/分段激活族——swish 用 sigmoid 门控、mish 用 softplus 门控（\(x\cdot\tanh(\mathrm{softplus}(x))\)，与本题的 softplus 稳定式同源）、hardsigmoid/hardswish 是 sigmoid/swish 的分段线性近似（与 hardtanh 同属"硬"系）。本题四件套补齐 torch 激活列表里剩下的常客。

## 对拍口径与复杂度

- 主判据是与 `torch.nn.functional` 对应算子直接 allclose（softsign/hardtanh/celu/logsigmoid；fp32 atol=1e-6、fp64 1e-9），叠加稳定式的结构要求（softplus 阈值 20、expm1+float64）；示例值与极端探针（\(\pm10^4\)、α=1e6、x=−120）均为实测回填。
- 判别探针：参数忽略语义三件套、hardtanh 饱和 torch.equal、celu 恒等极限（α=1e6，atol=1e-4）、logsigmoid 有限性 + "朴素式确实非有限"的反例断言、非法 mode ValueError。
- 复杂度：时间 \(O(\mathrm{numel}(x))\)（每个 mode 都是一遍逐元素运算；celu 额外一次 float64 cast，中间内存 \(2\times\)）；额外空间同量级临时张量。
