# 解析：视频时空 Patch 与合并友好顺序

## 来源与适用范围

参考 `transformers==5.20.2` 的 [`models/qwen2_vl/video_processing_qwen2_vl.py`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py#L275-L313)，`patchify` 第 275–313 行的布局变换。

原始 batch 版拆分形状为：

```text
[B, gt, tau, C, gh/m, m, p, gw/m, m, p]
```

其轴置换为 `permute(0,1,4,7,5,8,3,2,6,9)`。本题仅抽取单视频的尾帧补齐、时空 patch 打包和合并友好排序，并额外返回坐标；不包含 resize、归一化、Conv3d、视觉编码器或实际 patch merging。

`mm-patchify` 的普通单图 raster 展平不足以解决本题；`mm-patch-merger` 不包含时间维打包。`N/m²` 只是后续合并后的 LLM 槽数，本题仍返回 `N` 个 token。

## 参考方法

1. 检查 `H % (p*m)` 与 `W % (p*m)`。先验证，避免底层 reshape 错误代替要求的 `ValueError`。
2. 用 `(-T) % tau` 计算补帧数，拼接最后一帧的重复副本。`T=1` 且 `tau>1` 也采用同一规则。
3. 将补齐视频 reshape 为：

   `[gt, tau, C, gh/m, m, p, gw/m, m, p]`

4. 执行 `permute(0,3,6,4,7,2,1,5,8)`，得到：

   `[gt, gh/m, gw/m, m, m, C, tau, p, p]`

5. 前五轴展平为 token 轴，后四轴展平为特征轴。用同一组前五轴的网格生成 `(t, block_h*m+inner_h, block_w*m+inner_w)` 坐标。

参考实现使用 `reshape` 而非假定输入连续的 `view`。置换后再次 reshape 可能复制数据，但不能修改输入。

## 正确性要点

令当前 token 的五轴索引为 `(t,bh,bw,ih,iw)`，则其序号为：

```text
n = ((((t*(gh/m)+bh)*(gw/m)+bw)*m+ih)*m+iw)
h = bh*m+ih
w = bw*m+iw
k = ((c*tau+dt)*p+ph)*p+pw
```

轴置换恰好把五个 token 轴放在前面，把 `C,tau,p_h,p_w` 放在后面。因此，`patches[n,k]` 等于原视频的

`video[min(t*tau+dt,T-1), c, h*p+ph, w*p+pw]`。

补齐帧只引用最后一帧；所有原始像素均保留。每个 `(t,h,w)` 恰好出现一次，坐标可将输出逐元素逆写回补齐视频。这也说明该操作是可逆重排，不是卷积或降采样。

## 复杂度

设 `T' = ceil(T/tau)*tau`、`P = T'*C*H*W`、`N = gt*gh*gw`。时间复杂度与含输出的空间复杂度均为 `O(P+N)`；参考实现中的补齐和重排可能产生同阶临时张量。

## 测试与易错点

共有 **13 个 `check_*`**，其中 `check_example.example = True`。独立 oracle 严格按 token 五轴及元素四轴嵌套循环逐元素读取，不复用参考解的 reshape/permute；逆重建按返回坐标反向遍历 token，逐元素写回并校验原帧与补帧。

覆盖：奇数 `T` 的尾帧复制、`T=1`、恰好整除、`m=1/2/3/4`、非 raster 顺序、矩形、多通道内部顺序、非连续带偏移视图、三种 dtype、最大参数、随机对拍、非法空间尺寸、不修改输入与底层存储，以及显式补帧等价性和数值平移不变性。

常见错误是补零、把 `tau` 放在 `C` 前、先全局 raster 再分组、返回合并后的数量，以及把网格坐标误当成像素或帧坐标。

代码与独立测试先通过以下提交模式自测，再实跑样例并编写文档；结果为 **13/13 AC**：

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-video-patch-order --mode submit
```
