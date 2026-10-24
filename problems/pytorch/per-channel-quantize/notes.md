# 按通道量化 详解

## per-tensor 为什么在权重上不够

权重的不同输出通道常量级悬殊：某通道有 $|w|=50$ 的离群值、另一通道都在 $|w|<0.5$。per-tensor 用全局 absmax=50 定 scale，于是小值通道 $q=\operatorname{round}(0.5/ (50/127))=\operatorname{round}(1.27)=1$——整通道只剩几个格点，信息几乎全丢。per-channel 让小通道按自己的 absmax=0.5 定 scale，分辨率拉满。这就是几乎所有 PTQ/QAT 方案对**权重**默认 per-channel 的原因（激活一般仍 per-tensor，因为激活的动态范围随输入变化、无法离线按通道固定 scale）。

## axis 语义与归约维

「per-channel along axis $a$」= 通道是 $a$ 维、每个通道内**其它所有维**共享统计。对 `[C, in]` 且 `axis=0`，absmax 要沿 `dim=1` 归约得长度 `C` 的 scale。多维（conv 权重 `[O, I, kh, kw]`、`axis=0`）则沿 `dim=(1,2,3)` 归约。写成 `reduce_dims = 除 axis 外全部` 最通用；别把 axis 当成「归约掉的维」——恰好相反。

负 `axis` 先 `axis % dim` 规整。广播 scale 时把 shape 变成 `[1,…,C,…,1]`（仅 axis 维为 C）。

## 除零：oracle 的隐藏约定

某通道全零 → absmax=0 → scale=0。直接 `x/0` 得 NaN，但 `torch.fake_quantize_per_channel_affine` 对 scale=0 通道**返回 0**。所以必须把 scale=0 的通道单独处理（本题解法用 `safe_scale` 把 0 换成 1 参与除法，再乘回真 scale=0 使输出归零）。这是最容易被漏、且与 oracle 对不齐的点。

## fake-quant 与真整型存储

本题和 `symmetric-quantize` 都是 **fake-quant**：算完仍返回浮点（量化→反量化走一圈），用于 QAT 让梯度直通、或用于评测模拟量化误差。真正部署时才把 $q$ 存成 int8、只留 per-channel scale 向量（见 `weight-only-int8-matmul`，那里 scale 参与推理）。

## 关联

- 上游：`symmetric-quantize`（per-tensor 版，本题是它的通道级推广）。
- 下游：`group-wise-quantize`（比 per-channel 更细的分块粒度）、`weight-only-int8-matmul`（真存 int8 后怎么用）。
- 判题：`check_example` 用 dump 值；`check_matches_oracle` 对拍 `torch.fake_quantize_per_channel_affine`；另有全零通道、3 维 conv、负 axis、float64 用例。
