# LSTM 单元详解

## 三道门在做什么

把细胞状态 $c$ 想成一条「传送带」，门是作用在其上的**按位可微开关**（$\sigma$ 输出 $[0,1]$）：

- **遗忘门** $f$：决定上一刻记忆 $c_{t-1}$ 保留多少 → $f\odot c_{t-1}$。
- **输入门** $i$ 与**候选** $g=\tanh$：决定当前新写入多少 → $i\odot g$。
- **输出门** $o$：决定从更新后的记忆读出多少 → $h_t=o\odot\tanh(c_t)$。

$c_t=f\odot c_{t-1}+i\odot g$ 里那条加法（而非普通 RNN 的 $\tanh(W h + U x)$）是关键：
梯度能沿「$\oplus$」几乎无损地回传到很久以前，缓解梯度消失。

## 门排布约定 `[i, f, g, o]`

PyTorch `LSTMCell` 把四门拼在 `gate` 的通道维、顺序固定为 **i, f, g, o**（CUDNN 也是这套）。
`chunk(4, dim=1)` 后必须按同名解包；写反成 `i, g, f, o` 会让「候选/遗忘」串门，结果全错却不报错——
这是从零实现 RNN 最经典的静默 bug。

## 偏置初始化技巧

经验上把**遗忘门偏置初始化成正值**（如 +1~+2），使 $f$ 起步接近 1、倾向「记住」，可显著
改善长序列训练（Saxe & Harris 2013）。其余门维持 0 初始化。

## 从 LSTMCell 到序列

`nn.LSTM` = 把本 Cell 沿时间步循环 + 多层堆叠 + 双向拼接 + 打包变长序列。理解单步即可
读懂 `for t in range(T): h, c = cell(x[t], (h, c))`，也是实现自定义 recurrent（如
minLSTM、URNN、带残差的 cell）的模板。

## 现代地位

虽然 Transformer 取代了多数序列任务里的 LSTM，但 LSTM 仍活跃在：小数据/低算力序列建模、
时序控制（如 RL 的 Recurrent Policy）、以及作为「门控归纳」启发新架构（xLSTM、
RetNet 的门控衰减都带 LSTM 影子）。
