# PEFT：BEFT 旁路偏置高效微调

## 题目描述

**BEFT**（Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes，arXiv:2509.15974，ACL 2026，已入 HF PEFT `tuners/beft`）把参数效率推到极致：不改权重矩阵、不加低秩因子——只给 value 投影**新增**一个旁路偏置 \(\mathrm{beft\_bias}\ (1, out\_features)\) 作为唯一可训练参数。论文结论：低数据场景下微调 \(b_v\) 优于 \(b_q\)/\(b_k\)（softmax 结构性论证：\(b_k\) 在 softmax 前被平移抵消）；对无 bias 的 LLM 就是「加 bias 再训练」路径。

`beft Linear.forward` 的确定性链（官方 `layer.py`）：

```
result = base_layer(x)                     # 原权重与原 bias 原封不动
beft_bias = sum(beft_bias[a].flatten()     # 多 adapter 先求和
                for a in active_adapters)
result = result + beft_bias                # (..., out) + (out,) 广播
```

`init_weights=True` 时 \(\mathrm{beft\_bias}\) 全零——**训练起点 \(\Delta = 0\)**。merge/unmerge 状态机：

```
merge:   base.bias += beft_bias.squeeze()    # 原 bias 不存在 -> ValueError
safe_merge: 先 clone 相加再 isfinite(...).all() 检查，非 finite 不写回
unmerge: base.bias -= beft_bias.squeeze()    # 恢复原 bias
```

`update_layer` 布局账：\(\mathrm{beft\_bias} = \mathrm{randn}((1, out))\)，init 时 reset 全零；**每层可训练参数量 = out_features**（无任何权重矩阵增量）。

## 规则

- `mode="forward_delta"`：x、weight、beft_bias、base_bias（可为 None）入参，返回 base 线性输出 + 旁路 bias（广播到 \((\dots, out)\)）。
- `mode="adapter_sum"`：多 adapter 的 beft_bias 求和账——先各自 flatten 累加，官方 forward 把**和**一次性加到 base 输出。
- `mode="merge_bias"`：base_bias 与 beft_bias 相加返回 merged bias；base bias 不存在抛 ValueError；`safe=True` 先做 finite 检查，**检查对象是相加后的和**。
- `mode="unmerge_bias"`：merged_bias − beft_bias，恢复原 bias（merge→unmerge round trip 恒等）。
- `mode="layer_setup"`：返回 `{"shape": [1, out], "param_count": out}`，`init_weights=True` 附全零 init；False 只回形状账（随机初始化不出数值锚）。
- 全程 float64。

## 示例

\(in=4, out=6, x\ (2,3,4)\)：

```
forward_delta(x, W, beft, base_bias=b) -> (2,3,6) = x@Wᵀ + b + beft
adapter_sum([beft, beft2]) -> beft.flatten() + beft2.flatten()
merge_bias(b, beft) -> b + beft.squeeze()        # (out,)
unmerge_bias(merged, beft) -> b                  # round trip 恢复
layer_setup(6) -> {shape: [1,6], param_count: 6, init: zeros}
```

## 约束

- 旁路不修改原权重与原 bias：零初始化起点 forward 恰等于 base 输出。
- merge/unmerge 加性可逆：多 adapter merge 后逐个 unmerge 也恢复原 bias。
- safe_merge 拦截两类非 finite：操作数含 NaN、**两个有限数相加溢出**（检查对象是和）。
- 无 base bias 的层 forward 仍可算（base = x@Wᵀ），但 merge 必须报 ValueError。
