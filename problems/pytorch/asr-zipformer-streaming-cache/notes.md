# 踩坑记录（order 828）

## 回源口径

- 仓路径 `k2-fsa/icefall` 的 `egs/librispeech/ASR/zipformer/zipformer.py`（2465 行）+ `scaling.py` 的 `ChunkCausalDepthwiseConv1d`。官方 doctest（`_test_zipformer_main`）只验证 forward 能跑，数值断言全部自建对拍。
- 每层 6 组缓存：`(cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2)`，state 布局 `states[i*6:(i+1)*6]`。
- chunkwise_conv 构造时**自带 `padding=kernel_size // 2`**（对称卷积自补零），streaming_forward 的 `x[..., left_pad:]` 切片后直接 conv——漏掉这个 padding 输出长度就少 k-1 帧（我首版写 padding=0 立即 RuntimeError，形状兜住）。

## 测试设计踩坑

1. **广播坑（b=1 时 (h,b,dp)*(h,dp) → (h,h,dp)）**：手工对拍 `ph[:,:,t,:] * pos[0,rel,:].reshape(h,dp)`，b=1 时 b 轴被 h 广播，manual 形状变 (h,h,seq,r) 且数值错位，softmax 对拍全挂。修复：reshape(h,1,dp) 显式对齐。**判题镜像**：solution 走 matmul 不踩这坑，但对拍侧必须显式。
2. **torch.stack 轴错位**：外层 `stack(dim=-2)` 作用在 (h,b,L+seq) 上把新轴插进 b 位置 → (h,seq,b,r)，b=1 时形状 (h,3,1,5) 与期望 (h,1,3,5) 数值完全不同。修复：显式 dim=2。教训：**stack 列表推导造 4 维张量时 dim 必须显式写数字，不用负数**。
3. **chunk_scale 重叠区**：cs 与 k 重叠时（cs=8,k=5 列 3-4）左右修正相加——首版断言 `out[:, :5] == 1+scale[0]` 在重叠列恒假。对拍要按三段（左带/重叠/右带）分开断。
4. **单 chunk 语义 assert**：官方 streaming_forward 每次"一个 chunk"，我首版让 mode 同时接受 seq=8+chunk_size=4（想一步到位），导致 chunk_scale(cs=4) 与 x_chunk(seq=8) 形状互斥。修复：mode 内 `assert seq == chunk_size` 钉死单块语义，跨块组合交给 stream_equiv。
5. **as_strided 推导**：官方 stride 变换 `s2-s3` + `storage_offset=s3*(seq-1)` 展开后是 `[h,b,t,(seq-1-t)+j]`。推导时容易把相对方向写反（t-j vs j-t）；用单元素三方对照（manual dot / ps 原始取值 / direct dot）钉死。

## mut 设计（15 变异全 CAUGHT）

- 缓存族：取头/不更新/错位 roll —— cache_roll 与 conv cache 双断言兜住；
- 相对位置：索引去 t —— 逐项手算对拍必须保留（gather 内部一致性的变异单靠 softmax 查不出）；
- mask -1000→-1：mask 列权重 < 1e-12 断言（-1 时约 0.27 不满足）；
- value_out 丢缓存：形状 RuntimeError 兜住 + 恒等断言；
- 门控族（nonlin/conv sigmoid 丢弃）："门必须真参与"断言（与无门版本 allclose 1e-3 不成立）；
- chunkwise padding 丢失：形状 RuntimeError；
- causal 权重串用 chunk_w：闭式对拍挂；
- scale 左右互换 / 丢 1.0 基底：三段断言 + 全表对拍；
- whole 恒等破坏（causal 少一帧偏移）：整除恒等断言挂。
