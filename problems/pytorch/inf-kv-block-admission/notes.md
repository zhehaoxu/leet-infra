# 解题说明 · 块预算准入与顺延

## 先确定模拟骨架，再抠细节

这题的正确性 90% 取决于骨架是否被严格遵守：**每个 step 内先解码、后准入，两个子循环都按 id 升序**。任何一处顺序换了，输出就会漂：

- 若先准入后解码，刚完成的请求归还的块要到下一轮才能被利用，`admit_step` 会整体推迟。
- 若解码阶段倒序遍历，块会先给后面的请求，`stalls` 的归属就变了。

## 占用块数是 token 数的函数，不是累加量

不要维护「已占用块数」再去 `+1`。每一步都从 token 数重新算：

```
need = ceil((prompt_len + generated + 1) / block_size)
extra = need - held
```

因为块是**按页**占的：`prompt_len` 不是 `block_size` 的整数倍时，前几个生成 token 都还落在同一页里，`extra` 是 0。只有跨页时才需要真正去要块。

## 完成即归还

`generated == max_new` 的那一刻就把 `held` 加回空闲池并清 0。忘了这一步，后面的请求会被永久饿死，`finish_step` 直接变成 `-1`。

## 峰值要在「分配的那一刻」记录

峰值不是最终占用，而是所有时刻的最大值。最简单稳的做法是每次成功分配后立刻更新 `peak = max(peak, num_blocks - free)`，而不是在阶段结束时算——阶段结束时空闲可能已经被归还的块恢复了。

## 常见的两个错

1. **准入时按 id 顺序遇到装不下的就 `break`**：这把策略退化成了队头阻塞，一个超长 prompt 会让后面所有小请求饿死。正确做法是 `continue`。
2. **把 `max_new == 0` 当成非法输入**：它在引擎里是有意义的（纯 prefill、拿完首 token 就撤），要能正确地在准入步完成并归还块。

## 复杂度

每步 O(N) 的两次扫描，总复杂度 O(max_steps × N)。真实引擎用双端队列 + 块计数器，但那是常数因子的优化，不影响判定口径。

## 相关题目

- `inf-continuous-batching-scheduler`：同一套 running 集合，但用固定 `max_batch_size` 计数、不建模块预算。
- `inf-paged-kv-block-alloc`：块从哪来、怎么按序分配。
- `inf-paged-cow-block-share`：块被多条序列共享时怎么记账。
