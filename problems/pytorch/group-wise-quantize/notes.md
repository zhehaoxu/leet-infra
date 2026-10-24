# 分组量化 详解

## 为什么 LLM 权重要分组

一条输出通道的权重沿输入维 K 分布不均：有的 K 段含离群大值、有的段都很小。per-channel 给整条通道一个 scale，小值段被大值段「稀释」到几乎全为 0 格点。INT8 per-channel 尚可接受，但 **INT4 只有 8 个正格点**，per-channel 精度崩塌。group-wise 把通道按 K 切成小组（128/64），每组独立 absmax 定 scale，等效于「局部自适应分辨率」，是 GPTQ/AWQ 能把 LLM 压到 W4 仍保精度的关键。代价：多存一批 scale + 反量化时按组切换，硬件要对 `group_size` 对齐。

## group_size 与短尾

$n$ 不能被 `group_size` 整除时，最后一组只有 $n \bmod \text{gs}$ 个元素。**必须按它真实长度估 absmax**——若为对齐补零，0 不影响 absmax（0 不是新极值），但补「最大值」或错误 reshape 会污染。最稳的写法是切片 `start:start+gs`，短尾自然取到行尾。

## qmax 随位宽

对称量化正格点数 $q_{max}=2^{\text{bits}-1}-1$：int8→127、int4→7、int8 的 $-128$ 端不用（保持关于 0 对称，和 `symmetric-quantize` 一致）。bits 是变量时别把 127 写死。

## 与其它量化粒度的关系

| 粒度 | scale 组数（[out,in]） | 典型用途 |
|---|---|---|
| per-tensor | 1 | 激活、简单部署 |
| per-channel | out | 权重 int8（`per-channel-quantize`） |
| group-wise | out × ceil(in/gs) | 权重 int4（GPTQ/AWQ） |

粒度越细精度越高、元数据越多。本题沿 in 维（最后一维）分组。

## 判题要点

- 全零组 absmax=0 → 原样返回 0（除零陷阱）。
- `_ref` 用 `index_reduce_` 向量化按组求 absmax，与解法的逐组切片循环是两条不同推导路径，互相独立核对。
- `check_example` 用 dump 的 int8/int4 值；另有短尾、gs>n、3 维、int4、不修改输入用例。
