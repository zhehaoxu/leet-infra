# ReLU6 详解

## 为什么给 ReLU 加一个上界

ReLU 输出无上界，激活值可能散布到很大范围，做 int8 量化时要为「偶发的极大值」留很宽的
标度，精度被稀释。ReLU6 把每层输出钉在 $[0,6]$，量化格点只需覆盖这个小区间，等效比特分辨率
更高、更适合移动端定点推理——这正是 Howard et al. 2017《MobileNets》选它的原因。

$$ \operatorname{ReLU6}(x)=\min(\max(0,x),6) $$

## 与 ReLU / clipped 的关系

- `relu6 == torch.clamp(x, 0, 6)`，是 ReLU 再套一层 `min(·,6)`。
- 上界「6」是经验超参（可推广为带任意 `max_value` 的 `HardTanh`），并非数学必然。
- 梯度：$0<x<6$ 处为 1，$x\le0$ 或 $x\ge6$ 处为 0——两端都会「死区」，饱和侧也可能像 ReLU
  那样因大激活被截断而丢信息。

## 后继：ReLU6 → Swish/HardSwish

MobileNetV3 用 **Swish / HardSwish** 替换 ReLU6 取得更好精度；HardSwish 本质也是「带下界
0、上界 3 的近似有界激活」（见本合集 swish / hardswish 题）。ReLU6 是理解这条「为量化而设计
激活」演化的起点。

## 实现要点

逐元素 `clamp` 即可，务必保持输入 dtype（float32/float64）与形状；无状态、无需分支，
在硬件上是单条 min/max 融合指令，极为廉价。
