# Multi-Head Attention 前向

手写 Transformer 的 Scaled Dot-Product Multi-Head Attention 前向传播（不含投影矩阵），重点练习 `reshape/transpose` 的多头拆分与广播 mask。

## 计算流程

1. 把每个头的切片拆出来：`(B, L, D) -> (B, H, L, d)`。第 `h` 个头取 `q[:, :, h*d : (h+1)*d]`（即按特征维**连续切分**，再 `reshape/transpose`）。
2. 对每个头：$\text{Attn} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$
3. 若 `causal=True`：位置 `i` 不允许 attend 到 `j > i` 的 key。
4. 把各头拼接回 `(B, L, D)`。

## 规则

1. 输出形状 `(B, L, D)`，dtype 与输入一致。
2. `D % num_heads != 0` 时抛出 `ValueError`。
3. softmax 沿 key 维（最后一维）归一化；数值稳定（先减最大值，或用 `F.softmax` 计算注意力权重是允许的——考察点是拆分与 mask，不是 softmax 本身）。
4. 禁止直接调用 `F.scaled_dot_product_attention` / `torch.nn.Attention` 等现成 attention。

## 示例

```python
B, L, D, H = 2, 5, 8, 2
q = k = v = torch.randn(B, L, D)
out = solve(q, k, v, num_heads=H, causal=True)
out.shape  # torch.Size([2, 5, 8])
```

## 提示

- 拆分写法：`x.view(B, L, H, d).transpose(1, 2)` 得到 `(B, H, L, d)`
- causal mask：`torch.ones(L, L, dtype=torch.bool).tril()` 配合 `masked_fill(float("-inf"))`
- 拼接回去：`transpose(1, 2).reshape(B, L, D)`，注意 `transpose` 后需要 `contiguous`（或直接用 `reshape`）
- `-inf` 行全 mask 不会出现（对角线永远可见），不必处理全 mask 组

## 约束

- `1 <= B <= 8`，`1 <= L <= 64`，`H >= 1`，`d >= 1`
