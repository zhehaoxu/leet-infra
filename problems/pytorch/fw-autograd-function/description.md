# 自定义 autograd.Function 的前反向账

## 题目描述

`torch.autograd.Function` 是把自己写的 CUDA/融合核接进反向图的正门。新版 API 把保存中间量的职责从 `forward` 拆给 `setup_context`，`ctx.save_for_backward` 决定哪些张量在反传后被释放；forward 还可以返回不进计算图的不可微对象。本题要求实现一个融合算子 \(y=c\,x^2\tanh(x)\)（带一个 `int` 型第二输出），并把前向/反向的调用行为记成账。

## 规则

实现 `solve()`，按 starter docstring 中的固定流程（输入 \(x=[-2.5,-0.75,0.5,1.25,3.0]\)、\(c=2.0\)、上游梯度 \(g=[1,2,3,4,5]\)）完成：

1. 新版 API：`forward(x, c)` **不带 ctx**，上下文全部在 `setup_context(ctx, inputs, output)` 里建立；\(x\) 走 `ctx.save_for_backward`，\(c\) 直接挂 `ctx.c`。
2. 第二输出 `neg_count` 是纯 Python `int`：框架原样透出（`neg_count_type == "int"`），不建图，backward 里对应位置的梯度参数为 `None`（`extra_grad_is_none == True`）。
3. `backward` 里 `ctx.needs_input_grad` 应为 `[True, False]`（\(c\) 不可微），返回 `(gx, None)`。
4. 手写梯度必须与「同式的标准算子组合 + autograd」结果一致（`atol=rtol=1e-5`）。
5. 重复对同一个图反传两次：第二次必须抛 `RuntimeError`（saved tensors 已释放），但 `backward` 入口计数仍会累加——`bwd_calls` 记的是**入口被调用次数**。

## 示例

```
输入：无
输出：
{
 "y_flat": [-12.33267879486084, -0.71454256772995, 0.23105858266353607,
            2.650886297225952, 17.910985946655273],
 "neg_count": 2, "neg_count_type": "int", "y_has_grad_fn": true,
 "grad_x_flat": [10.198546409606934, 5.15321159362793, 3.9523744583129883,
                 20.47085952758789, 60.59123229980469],
 "grad_x_ref_flat": [10.198546409606934, 5.15321159362793, 3.9523744583129883,
                     20.47085952758789, 60.591224670410156],
 "grads_match_ref": true, "extra_grad_is_none": true,
 "needs_input_grad": [true, false], "fwd_calls": 2, "bwd_calls": 3,
 "second_backward_raises": true
}
```

`grad_x_flat` 与 `grad_x_ref_flat` 末项差一个 ULP（\(60.59123\) vs \(60.59122\)）：手写解析式与链式法则图的浮点求值顺序不同，属预期内。

## 约束

- 数值对拍 `atol=1e-5, rtol=1e-5`；布尔/计数/类型名精确一致。
- 不要用旧式 `forward(ctx, ...)` 混写 `setup_context`——新版 API 下会直接 `TypeError`。
- 全流程无随机数。
