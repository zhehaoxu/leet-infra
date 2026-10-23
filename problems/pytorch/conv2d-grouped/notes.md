# 分组卷积详解

## 通道划分与「组间隔离」

标准卷积里每个输出通道会看**全部** `Cin` 个输入通道；分组卷积把 `Cin` 切成 `groups` 段，
第 `g` 段的输出通道**只看自己那一段**的输入通道：

$$ \text{out}[\,g{:}o\,] = \sum_{c \,\in\, \text{group } g} \text{conv}(x_{c},\, w_{g,o,c}) $$

因此 `weight` 的输入通道维只剩 $C_{in}/\text{groups}$。这带来 $1/\text{groups}$ 的参数量与算力，
代价是组之间不再有信息交换——需要靠后续 1×1 或网络结构（如 ShuffleNet 的通道洗牌）重新混合。

## 两个退化关系

- `groups == 1` → 普通卷积。
- `groups == Cin` → **深度卷积（depthwise）**，每个通道一个独立核，权重视图 `(Cin, 1, kh, kw)`。
  这就是本题「分组局部性」探针要检出的边界：扰动通道 0（属组 0）绝不能改变组 1 的输出。

## 易错点

- **切片顺序**：输入、输出通道都按组**连续切分**，别把 `weight` 的维度顺序弄反
  （是 `(Cout, Cin//g, kh, kw)`，不是 `(Cin//g, Cout, ...)`，后者是转置卷积的约定）。
- `Cin`、`Cout` 必须都能被 `groups` 整除，否则 reshape 报错。
- bias 仍按完整 `Cout` 加，不受分组影响。

## 应用

ResNeXt 用 `cardinality`（即 groups）在同等算力下提升容量；MobileNet/ShuffleNet 用 `groups=Cin`
的 depthwise 卷积 + pointwise 组成「深度可分离卷积」，把 $3\times3$ 稠密卷积的 $C_{in}C_{out}kk$
降到 $C_{in}kk + C_{in}C_{out}$，是移动端 CNN 的基石。
