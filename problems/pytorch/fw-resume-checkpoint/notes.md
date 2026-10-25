# 详解：断点续训的连续性账

## 出处

PyTorch 官方 *Tips for Reproducibility* 与 checkpoint 惯例：想断点续训后逐位复现，需要同时保存/恢复 model、optimizer、scheduler、`torch.get_rng_state()` 四件（多卡再加 CUDA RNG）。`Optimizer.state_dict()` 的行为在 `torch/optim/optimizer.py`：state 里的张量直接引用活体 buffer；`load_state_dict` 在 CPU 上不拷贝、把传入张量绑进新优化器——同一份 ckpt 被两条路径加载时，后者的原地 `opt.step()` 会污染前者读到的动量。`torch.Tensor.to(device)` 同设备时同样返回自身。

## 核心与易错点

1. **四本账缺一不可**：权重对、动量对、调度器 `last_epoch` 对，但只要 RNG 没恢复，续跑阶段的 \((x,y)\) 取数就整体后移，loss 从第 4 步起彻底发散（本例第 4 步 0.0067 → 1.0938）。路径 3 就是这条「漏账」的对照实验。
2. **优化器 state 是别名不是快照**：`o.state_dict()["state"][0]["momentum_buffer"] is` 活体 buffer；`o2.load_state_dict(sd)` 后 `o2.state` 里就是这些对象。两条恢复路径必须各自取新克隆（参考解 `_clone_opt_sd` 存档与加载各克隆一次），否则「前一条路径的账」会被后一条路径原地改错——题目里 `momentum_restored`、`losses_no_rng` 双双记错就是这个坑。
3. **快照时机**：比较用的终值/动量要在路径 2 结束、路径 3 开跑前就 `clone()` 下来；等三条路径全跑完再读会读到被污染的对象（这是本仓库参考解第一版的真实 bug）。
4. **lr 账**：循环内 `sched.step()` 垫后，第 \(i\) 步生效 \(lr=0.1\cdot0.5^{\lfloor i/2\rfloor}\)（StepLR 按 `last_epoch` 查表，`__init__` 已预告 0 号位）。
5. **`resumed_bitwise_identical` 用列表 `==`**：loss 都是 `float(loss.item())`，同一 IEEE 运算序列在 Python 里逐位可复现；用 allclose 会掩盖真实的错位。
6. **Module 的 `state_dict()` 不踩别名坑**：`load_state_dict` 对 nn.Module 逐参数 `copy_` 写入，存/取都安全；只有 optimizer/scheduler 的 state 是直接换引用。

## 对拍口径

solution 跑真三路径；`_oracle` 纯 Python：同一 seed 重放 \((x,y)\) 数据流（torch 的 Philox 随机流无法用 numpy 复现，取数是与框架共享的唯一原语），前向/反向用 \(z=W_1x+b_1,\ a=\tanh z,\ \hat y=w_2\cdot a+b_2\)、\(\partial L/\partial \hat y=2(\hat y-y)\) 与 \(1-a^2\) 的解析式，SGD 动量 \(b\leftarrow0.9b+g,\ w\leftarrow w-lr\cdot b\)，\(lr=0.1\cdot0.5^{\lfloor i/2\rfloor}\) 全程 float64 手算。loss 用 `allclose(1e-5,1e-5)`；`losses_resumed == losses_full`、前 3 项逐位相同、尾部逐项差 \(>10^{-3}\)、布尔与 `last_epoch` 精确断言。

## 复杂度

\(O(T\cdot d)\)，\(T=6\) 步、每步 2 维样本，毫秒级；内存只有几份小张量。
