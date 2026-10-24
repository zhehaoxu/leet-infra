# 解题说明 · DPO 损失

## 从 RLHF 到 DPO 的一步

标准 RLHF 目标 $\max_y r(y,x) - \beta\,\mathrm{KL}(\pi_\theta\|\pi_\text{ref})$ 有闭式最优解 $\pi^*(y|x)=\frac{1}{Z}\pi_\text{ref}(y|x)e^{r/\beta}$。反解出 $r=\beta\log\frac{\pi^*}{\pi_\text{ref}}+\beta\log Z$，代入 Bradley–Terry 偏好模型 $P(y_w\succ y_l)=\sigma(r_w-r_l)$，配分函数 $Z$ 相消，得到只含策略与参考策略对数比的 DPO 损失。因此 DPO 里那个"隐式奖励" $=\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ 才是本题 ① − ② 的由来。

## 为什么必须稳定 log-sigmoid

$\log\sigma(x)$ 直接写 `torch.log(torch.sigmoid(x))` 在 $x$ 很负时 sigmoid 下溢到 0、log 得 $-\infty$。恒等式 $\log\sigma(x)=\min(0,x)-\log(1+e^{-|x|})$ 保证 $e^{-|x|}\le 1$ 不溢出。`F.logsigmoid` 内部就是这么算的，oracle 用 python 的 `math.log1p` 复刻。

## 序列 logprob 的处理

真实实现里 chosen_logp 是"回答各 token logprob 之和"，并且要按 response mask 只统计生成部分（prompt 段不计）。本题把该聚合放在外面、直接给定序列级 logprob，聚焦损失组装，避免构造完整 causal-LM 前向。

## β 的作用

$\beta$ 越大，logit 尺度越大，sigmoid 越接近阶跃——对"chosen 必须显著优于 rejected"更苛刻，也意味着对参考模型 KL 约束更紧。$\beta=0$ 时 loss 退化为常数 $\ln 2$（无梯度），测试专门覆盖该边界。
