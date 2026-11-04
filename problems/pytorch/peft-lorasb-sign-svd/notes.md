# 详解：LoRA-SB 符号梯度估计的 SVD 初始化

## 出处

- 论文：**LoRA-SB: Initialize LoRA Using Sign-estimation for Efficient LLM Fine-tuning**（arXiv:2411.19557）。
- 官方实现：`CERT-Lab/LoRA-SB`（utils/svd_utils.py、utils/initialization_utils.py、utils/gradient_utils.py、utils/latent_utils.py）。
- 动机：PiSSA/LoRA-GA 一系用权重 SVD 拿主成分，但仍需要一次完整训练的前几步来「对齐」；LoRA-SB 用**一次小样本梯度**的符号估计直接构造 \(B@A\approx\widehat{\Delta W}\)，r_squared 模式把全部可训练参数压到 \(r^2\)。

## 官方实现的逐行翻译

**符号更新**（gradient_utils.py 尾部，逐行）：

```python
for name, grad in running_grads_sum.items():
    processed_grads[name] = (-1 * lr * torch.sign(grad))   # sign(0)=0，幅值统一 lr
```

**svd_grad**（svd_utils.py，逐行）：

```python
input_matrix_float = input_matrix.to(torch.float32)   # 官方调用传 weight.T -> (in,out)
torch.manual_seed(42)                                  # 固定种子（判题域改为入参 seed）
R = torch.randn(size, rank, ...)                       # dummy！从未参与计算
U, s, V = torch.svd_lowrank(input_matrix_float, q=rank, niter=n_iter)
return U, diag_s, V.T                                  # "Vrt" = V.T
```

```python
# 怪癖：R 在 svd_lowrank 之前消耗 RNG 流。svd_lowrank 内部用全局 RNG 生成随机投影，
# dummy 抽样把流推进 size*rank 个正态样本 -> 内部投影不同 -> SVD 结果逐位不同。
# 删掉 R 行（mutation m-sv-dummy）与改 seed（m-sv-seed）都会被值锚定抓到。
```

**拆分与装配**（initialization_utils.py，逐行）：

```python
final_enc = torch.tensor(Ur @ sr, ...)   # (in, r)   -> replace 时 .T -> A_w (r,in)
final_dec = torch.tensor(Vrt, ...)       # (r, out)  -> replace 时 .T -> B_w (out,r)
# half_init_dec: kaiming_uniform_(matrix[ceil(rows/2):, :], a=sqrt(5)) 对 dec 转置前下半行
# random_init:  kaiming(enc) 先、kaiming(dec) 后 —— 同一全局流，顺序是契约
# B_w@A_w = V diag(s) U^T ≈ sign_update (out,in)；符号矩阵满谱 -> 重构粗是设计使然
```

**r_squared 前向**（latent_utils.py，逐行）：

```python
result += (
    self.lora_B[self.active_adapter[0]](
        self.default_lora_latent_mapping(
            self.lora_A[self.active_adapter[0]](self.lora_dropout[...](x))
        )
    ) * self.scaling[self.active_adapter[0]]
)
# get_delta_weight: output_tensor = transpose(B @ M @ A, fan_in_fan_out) * scaling
# grad 模式: target.default_lora_latent_mapping.weight.data = Sr（diag 奇异值）
#            A/B requires_grad = False —— 冻结 SVD 因子，只训 r*r
# weight 模式(find_and_initialize): M 用 init_module_weights(sigma=1e-5) normal_ 初始化
# 非 r_squared: init_module_weights(target.lora_A.default, sigma=1e-5) —— A 近零
```

## 判题域剔除与红线

- **剔除**：`Accelerator`/distributed 同步、`OffloadContext`、bfloat16 设备搬运、tqdm——纯工程。
- **剔除**：`get_record_gradient_hook` 的梯度累积钩子（判题域不跑反向）。
- **保留**：sign 压缩、svd_lowrank 全流程（含 dummy RNG 怪癖）、kaiming a=√5 账、latent 前向/合并等价。
- **锚定**：s/enc/dec 的 fixtures 逐位锚定（float32 管线确定性已验证）；`torch.equal` 跨 dtype 按值比较，anchor 本身就是 float32 值的精确 double 表示，dtype 契约需显式 dtype 断言。

## 关键怪癖（回源逐行确认）

- **dummy R 消耗 RNG**：删掉这行 SVD 结果就变——判题域用 `svd_grad` vs `svd_grad_nodummy` 双路径直接测「不相等」。
- **fan_in 看转置前矩阵**：enc (in,r) 的 kaiming fan_in=r（不是 d_in）；dec (r,out) 的 fan_in=d_out。bound \(=\sqrt{6/(6\cdot fan\_in)}=1/\sqrt{fan\_in}\)。
- **half_init 的 RNG 痕迹**：half_init+random_init 组合的值 ≠ 纯 random_init（half_init 已消耗流），但 random_init 的 bound 断言仍成立。
- **非连续输入的 uniform_ 内存序**：svd_grad 返回的 dec=V.T 非连续，uniform_ 按内存序填充 → 与连续复算的 RNG 顺序不同。顺序复算测试用独立连续输入。
- **torch.sgn == torch.sign（实数域）**：m-su-zero 真等价剔除。
- **torch.equal 跨 dtype 按值比较**：dtype 契约必须显式断言 `.dtype == torch.float64`。

## 变异测试（26/26，1 等价剔除）

- m-su-zero（sign→sgn）：实数域真等价，剔除。
- 26 个变异覆盖：符号更新缩放/线性化、seed/dummy/niter/q/V.T/enc/dtype/rank 校验、half 行选择/ceil/顺序/转置/fan_in/bound/clone、latent 三段/scaling/合并、diag/近零 std/零初始化。

## 与已有题的划界

- **TinyLoRA（795）**：SVD 来自权重 W、√S 均分两侧、R=固定随机投影组合（v 可训）。LoRA-SB：SVD 来自 **sign 梯度估计**、U·s/Vᵀ 不对称拆分、M=diag(Sr) 初值可训。
- **PVeRA（804）**：kaiming a=0 全矩阵；本题 a=√5 下半行 + 整体重初始化 + 顺序契约。
- 题库无 svd_lowrand、无 sign 更新账（grep 已查重）。
