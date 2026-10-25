# 详解：Focal Loss

## 出处
Lin, Goyal, Girshick, He, Dollar, *Focal Loss for Dense Object Detection*（ICCV 2017，arXiv:1708.02002，RetinaNet）。二分类定义 \(\text{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t\)，\(p_t=y\,p+(1-y)(1-p)\)。本题取其多类别（softmax）形式，与交叉熵的关系是 CE 前乘调制因子 \((1-p_t)^\gamma\)。

## 核心与易错点
1. **调制因子在概率域、不在 logits 域**：先 softmax 得 \(p_t\)，再算 \((1-p_t)^\gamma\)。
2. **\(\log p_t\) 用 `log_softmax` gather**：直接 `log(softmax)` 在极端 logits 下易上溢；`log_softmax` 数值稳定。
3. **退化即交叉熵**：\(\gamma=0,\alpha=\text{None}\) 时 \(-(1-p_t)^0\log p_t=-\log p_t\)＝标准 CE 的逐样本 NLL，故应与 `F.cross_entropy(reduction='none')` 完全一致——本题用它做对拍锚点。
4. **alpha 是标量权重**：本简化版对每样本乘同一个 `alpha`（对应原论文正类权重 \(\alpha_t\)），不是逐类向量。

## 对拍口径
参考解 `log_softmax→gather→loss=-(1-pt)^gamma*log_pt`，按 reduction 归约，非法 reduction 抛 `ValueError`。判题 `_oracle` 纯 Python：手算每行 softmax（减最大值）取真类概率、按定义累加（结构不同、数值等价）。另设「γ=0 与 F.cross_entropy 一致」「γ 越大难样本占比越高（同一 batch 均值相对 CE 单调下降）」「alpha 线性缩放」「reduction none/sum/mean 一致」断言。

## 复杂度
\(O(N C)\)。
