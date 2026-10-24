# 解题说明 · Router z-loss

## 为什么是 logsumexp 的平方

$\mathrm{logsumexp}_e(x_e)$ 是 softmax 分母的 log。若所有 $x_e$ 同时加常数 $c$，softmax 不变但 logsumexp 增加 $c$。z-loss 惩罚「logsumexp 的绝对大小」= 惩罚「+c 这个方向」，等价于把 logits 往 0 拉；不改变 softmax 的相对形状。

## 与 entropy 正则的区别

熵最大化：鼓励每行 softmax 更平；会破坏「模型已经决定」的信号。
z-loss：不改变 softmax 形状，只约束 logits 的绝对值域；对模型偏好无损、纯数值稳定。

## logsumexp 稳定性

`torch.logsumexp` 内部 `max + log(sum(exp(x - max)))`，天然防上溢。判题不担心 `exp(1e10)`。
