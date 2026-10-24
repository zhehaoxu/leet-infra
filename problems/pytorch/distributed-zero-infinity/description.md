# ZeRO-Infinity：HBM / DRAM / NVMe 三层卸载账

## 题目描述

ZeRO-Offload 把优化器状态卸到 **CPU 内存**（两层）。ZeRO-Infinity 再往下加一层 **NVMe SSD**，并把卸载分片**沿数据并行维度切开分散到各 rank**（每个 rank 只存 $1/d$ 的卸载数据，而不是每节点存整份），从而用近乎无限的存储训超大模型。本题算清三层各占多少、以及一步要把多少字节从低层搬进显存（纯计算，自定简化模型）。

## 模型（务必按此定义）

参数量 $N$，数据并行度 $d$。训练态分三部分，**都按 ZeRO-3 沿 $d$ 切分**，每个 rank 各持 $1/d$：

$$ P = \frac{\texttt{param\_bytes}\cdot N}{d},\quad G = \frac{\texttt{grad\_bytes}\cdot N}{d},\quad O = \frac{\texttt{opt\_bytes}\cdot N}{d} $$

放置策略（两个开关 + 一个目标层）：

- **梯度分片 $G$ 恒驻 HBM**（每步反算即时用，不卸载）。
- `offload_param=True` → 参数分片 $P$ 离开 HBM，落入 `offload_target` 层；否则常驻 HBM。
- `offload_optimizer=True` → 优化器分片 $O$ 离开 HBM，落入 `offload_target` 层；否则常驻 HBM。
- `offload_target ∈ {"dram","ssd"}`：所有被卸载的分片都放同一低层。

于是每 rank：

- `hbm_bytes` $= G + \mathbb{1}[\neg\text{offload\_param}]\cdot P + \mathbb{1}[\neg\text{offload\_optimizer}]\cdot O$
- 卸载字节 $= \mathbb{1}[\text{offload\_param}]\cdot P + \mathbb{1}[\text{offload\_optimizer}]\cdot O$；全进 `dram_bytes`（target=dram）或 `ssd_bytes`（target=ssd），另一层为 0
- `step_transfer_bytes` $=$ 卸载字节（每步把这些分片从低层预取进 HBM；本模型不叠加梯度换出，忽略重叠/缓存）

## 规则

实现 `solve(n_params, dp_degree, offload_param, offload_optimizer, offload_target, param_bytes=2, grad_bytes=2, opt_bytes=12)`，返回字典 `{hbm_bytes, dram_bytes, ssd_bytes, step_transfer_bytes}`（均为 float 字节数）。

## 示例

```
N=7e9, d=8, offload_param=True, offload_optimizer=True, offload_target="ssd", 默认字节
P = 2·7e9/8 = 1.75e9,  G = 1.75e9,  O = 12·7e9/8 = 10.5e9
hbm_bytes          = G = 1.75e9
dram_bytes         = 0
ssd_bytes          = P + O = 1.225e10
step_transfer_bytes= P + O = 1.225e10
→ {"hbm_bytes":1.75e9, "dram_bytes":0.0, "ssd_bytes":1.225e10, "step_transfer_bytes":1.225e10}
```

## 约束

- `n_params ≥ 1`；`dp_degree ≥ 1`；`offload_target ∈ {"dram","ssd"}`；字节数默认 $2/2/12$
- 返回四个 float 字节数，相对误差 $10^{-6}$ 判等
- 忽略激活显存、PCIe/NVMe 带宽与计算重叠、offload 粒度调度——纯按上面定义算
- `offload_param` 与 `offload_optimizer` 全 False 时即 ZeRO-3 全驻 HBM：`hbm_bytes = (P+G+O)`、`step_transfer_bytes = 0`
