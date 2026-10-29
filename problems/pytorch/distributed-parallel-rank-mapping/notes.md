## 从步长求组

沿 `order` 累乘轴大小，得到每个轴的 stride。某 rank 在轴 $a$ 上的坐标为 `(rank // stride) % size`。从 rank 中减去该坐标贡献，可得该组最小成员 `base`；其它成员为 `base + k*stride`，自然升序。

组内 local rank 就是坐标。组号相当于从混合进制 rank 中删掉该轴这一位：

$$group\_id=rank\bmod stride+\lfloor rank/(stride\cdot size)\rfloor\cdot stride.$$

低位坐标保持原样，高位压紧一位，得到从 0 开始且无空洞的编号。它与按组的最小成员排序的结果一致。

## 容易混淆的约定

`order` 不是输出展示顺序而已，它改变 rank 对坐标的解释。测试固定字典顺序、遍历四轴全部 24 种排列，能识别忽略 `order` 的实现。大小为 1 的轴有 singleton 组，但该轴 local rank 恒为 0；一个轴覆盖全部 world 时则只有一个通信组。

这里的首轴最快与常规 row-major reshape 的末轴最快不同。网格工具的维度声明需和编码约定保持一致，不能只因为都叫 device mesh 就直接套用 reshape。

## 独立验证与复杂度

oracle 反向排列轴后枚举笛卡尔积，直接给坐标网格依次分配 rank，再以“其它坐标相等”筛出组，不复用参考解的 stride 公式。还验证每轴通信组恰好分割整个 world、组员对成员和组号的认知一致、local rank 可反查本人。

设轴数为 $A$，轴大小总和为 $D$。显式为每个 rank 返回全部组成员需要 $O(WD)$ 输出空间与时间，步长和当前坐标仅需 $O(A)$ 辅助空间；这是输出规模本身的成本。
