# 详解：Flow Matching / Rectified Flow 的 Euler 积分

## 出处

Lipman et al., *Flow Matching for Generative Modeling*（arXiv:2210.02747）给出条件流匹配目标；Liu et al., *Rectified Flow*（arXiv:2209.14577）用 \(x_t=(1-t)x_0+t\epsilon\) 的直线路径与 \(v=\epsilon-x_0\) 把采样变成单调 ODE；Esser et al., *Scaling Rectified Flow Transformers*（SD3，arXiv:2403.03206）把它做进潜空间 DiT。SD3 / Flux 的采样器就是这条 Euler 步。

## 核心与易错点

1. **时间方向**：训练用 \(t=0\) 数据、\(t=1\) 噪声，采样却是 \(t:1\to0\)，所以 \(\Delta t<0\)。把方向写反、却没把速度取反，会得到「越采越噪」。
2. **速度的定义**：\(v=\epsilon-x_0\)（指向噪声），不是 \(x_0-\epsilon\)。直线路径上它是常数，这也是 RF 一阶 Euler 精确的原因；换成 \(\epsilon\)-prediction 的 \(v\) 会把两套记号混在一起。
3. **轨迹必须是新建张量**：把状态存进 list 后再原地 `x += ...` 会污染历史轨迹，判题用「首元素仍等于起点」抓这类错误。
4. **返回语义**：`x_final` 与 `trajectory[-1]` 必须是同一个值（不是同一份内存也没关系），且长度严格 `n_steps + 1`。
5. **标量速度**：允许 `velocity_fn` 返回 python float（如 \(v\equiv c\)），实现要把它转成张量再广播，而不是直接乘 `None` 或报错。

## 对拍口径

参考解内部 float64 累加、状态以 float32 保存。判题 `_oracle` 先算出整条时间栅格，再用 python float 逐元素加乘（与参考解的张量运算不同源），另加四组结构断言：栅格与调用次数（速度场恰被调用 `n_steps` 次）、常速度闭式 \(x_{\text{init}}+(t_{\text{end}}-t_{\text{start}})c\)、RF 直线路径在 1/2/7 步下都精确还原且中间态落在直线上、零速度时轨迹不动。阈值 `allclose(atol=1e-5)`。

## 复杂度

时间 \(O(\text{steps})\) 次速度场调用；每次调用的代价由 `velocity_fn` 决定，状态更新的访存是 \(O(\text{steps}\cdot N)\)。
