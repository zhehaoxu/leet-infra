# QK-Norm（对 Q/K 做归一化的注意力）详解

## 问题：注意力 logits 爆炸

训练大模型时，$q\cdot k$ 的幅度随隐层维和训练推进不断增大，softmax 饱和、梯度消失，甚至 loss 尖峰发散。**QK-Norm** 在做点积前先分别归一化 $Q$、$K$，把 logits 幅度锁在有界范围，显著提升训练稳定性（ViT-22B、Chameleon、多数现代 LLM 采用）。

## 本关做法：RMSNorm 后点积

$$ \tilde q=\text{RMSNorm}(q)\cdot\gamma_q,\quad \tilde k=\text{RMSNorm}(k)\cdot\gamma_k,\qquad \text{scores}=\frac{\tilde q\,\tilde k^\top}{\sqrt d} $$

$$ \text{RMSNorm}(t)=\frac{t}{\sqrt{\frac1d\sum t_i^2+\epsilon}}\quad(\text{不减均值}) $$

- 归一化沿**末维 $d$**、逐 token 行；$\gamma$ 形状 $(d,)$ 广播。
- 之后再普通缩放点积，$\sqrt d$ 仍保留（与 SDPA 关同源）。

## 为什么用 RMSNorm 而非 LayerNorm

RMSNorm 不减均值、少一次归约，且**尺度不变性**正是这里要的：把 $q$ 整体放大不会改变归一化结果，从而抑制模长膨胀。LayerNorm 也可，但 RMSNorm 更省、够稳。详见「RMSNorm」关。

## 坑

- eps 在开方**内**；`torch.rsqrt(ms+eps)`。
- 归一化在 $Q$、$K$ 各做各的（用不同 $\gamma_q,\gamma_v$），别对 $V$ 做（V 不归一）。
- 只归一化到点积前，别和 softmax 后的加权混为一谈。
