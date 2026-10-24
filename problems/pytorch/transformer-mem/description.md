# Transformer 激活显存账（含梯度检查点）

## 题目描述

训练时反向传播需要前向保存的**激活（activation）**，是大模型显存的主要开销之一（与参数、梯度、
优化器状态并列）。**梯度检查点（activation checkpointing / 重计算）** 用「只存每层输入、反向时
重算中间激活」换取显存，是训练深 Transformer 的标准技巧。本题按一个**简化的教学模型**估算整个
模型的激活显存：不启用 / 启用检查点两种情形的字节数。

## 规则

- 实现 `solve(b, s, layers, hidden, ffn, heads, dtype_bytes=2)`，返回 `(full_bytes, checkpoint_bytes)`。
- **每层前向需保存的激活**（本简化模型规定的张量清单，单位：元素数）：

  | 张量 | 形状 | 元素数 |
  |---|---|---|
  | Q、K、V 投影输出 | 各 `(b,s,hidden)` | $3\,b\,s\,h$ |
  | 注意力输出投影（残差前） | `(b,s,hidden)` | $b\,s\,h$ |
  | 两处残差/LayerNorm 输入 | `(b,s,hidden)` | $b\,s\,h$ |
  | FFN 两处中间（fc1 出、act 出） | `(b,s,ffn)` | $2\,b\,s\,f$ |
  | 注意力概率矩阵 | `(b,heads,s,s)` | $b\,a\,s^2$ |

  合并同类项得每层完整保存量：
  $$ E_{\text{full}} = b\,s\,(6h + 2f) + b\,a\,s^2 $$
- **不启用检查点**：所有 `layers` 层的激活都保留：
  $$ \text{full\_bytes} = \text{layers}\cdot E_{\text{full}}\cdot \text{dtype\_bytes} $$
- **启用检查点**：只常驻每层的**输入**张量 $(b,s,h)$，反向遍历到某层时再重算该层的完整激活
  （峰值时额外占用 1 层的完整激活）：
  $$ \text{checkpoint\_bytes} = \big(\text{layers}\cdot b\,s\,h + E_{\text{full}}\big)\cdot \text{dtype\_bytes} $$
- 返回整数字节数。

## 示例

```
输入：b=4, s=1024, layers=6, hidden=512, ffn=2048, heads=8, dtype_bytes=2(fp16)
  E_full = 4*1024*(6*512+2*2048) + 4*8*1024*1024 = 62,914,560 元素
  full_bytes      = 6 * 62,914,560 * 2 = 754,974,720   (= 720 MiB)
  checkpoint_bytes= (6*4*1024*512 + 62,914,560)*2 = 150,994,944 (= 144 MiB)
输出：(754974720, 150994944)   —— 检查点把激活显存降到约 1/5
```

## 约束

- 全部为正整数；`dtype_bytes` 常见取 2(fp16/bf16)、4(fp32)
- 这是**简化教学模型**（不含参数/梯度/优化器状态显存，注意力按概率矩阵估算），只考核对
  激活量级与检查点「存输入 + 重算一层」的理解，不追求与某框架 profiler 完全一致
- 需返回**整数**字节数（用 Python int，勿丢精度）
