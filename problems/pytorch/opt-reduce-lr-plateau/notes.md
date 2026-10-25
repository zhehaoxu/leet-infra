# 详解：ReduceLROnPlateau 指标驱动平台期降速

## 出处
- 权威口径 = PyTorch 原生实现本身：`.venv-eval` 里的 `torch/optim/lr_scheduler.py::ReduceLROnPlateau`
  （`__init__` 的 `best=mode_worse`、`step` 的四段顺序、`_reduce_lr` 的 `max(old*factor, min_lr)` + `old-new>eps` 判定、
  `in_cooldown` 属性、`_is_better` 的四种 mode×threshold_mode 组合）。该调度器没有对应论文，
  自 PyTorch 0.2（2017）起就是这套语义，本题以源码逐行回源。
- 概念上属经典 hill-climbing 的「无进展就退一步」启发式（Sutton & Barto 教材里的 reduce-on-plateau 记法），
  但**不要**按直觉版实现：直觉版和源码在 patience 边界、冷却扣减时机、eps 抑制三处都不同。
- todos.md #90③ 未给 arXiv 号（只说"原生可直接对拍"），本题照此执行。

## 核心与易错点
1. **`num_bad > patience` 是严格大于**。`patience=2` 表示「容忍 2 个坏步」，第 3 个坏步才降速；写成 `>=` 会整体提前一步。
2. **首个指标只建立基线**。`best` 初值是 ±inf，所以第一步必然判为"更优"、`num_bad` 恒为 0，绝不降速；
   `patience=0` 时最早也要第二步才可能降。
3. **`threshold` 是"改善必须越过动态阈值"**：`min/rel` 用 `best*(1-threshold)`，等于 0 的改善（指标原地不动）
   会被判成变差、累加 `num_bad`。`abs` 模式则是 `best-threshold`。`mode="max"` 时两式方向全部翻转。
4. **冷却期的记账时机**：`cooldown_counter>0` 时先 `-=1` 再把 `num_bad=0`，本步的坏计数被作废；
   而 `cooldown_counter` 是在**降速之后**才被置为 `cooldown`，所以降速那一步本身不受冷却影响，
   随后 `cooldown` 步被吞掉。少写 `num_bad=0` 会导致连续过度降速。
5. **`eps` 抑制的是写回、不是记账**：`new=max(old*factor, min_lr)`，若 `old-new <= eps` 就完全不写 lr，
   但 `num_bad`/`cooldown_counter` 依旧重置。到 `min_lr` 地板之后的降速正是靠这条变成幂等空操作。
6. **`min_lr` 地板在乘法之后取 max**，不是把 `factor` 改成自适应；`lr` 一旦落到地板，后面每次降速都保持地板值。
7. `step()` 只改 `param_groups[*]["lr"]`，不碰参数张量；评测靠读原始张量与 lr 序列两者确认没有越权更新。

## 对拍口径
- 外部参照：**原生 `torch.optim.lr_scheduler.ReduceLROnPlateau`**。用同一个 `SGD(lr=base_lr)` 起头，
  喂同一指标流，逐步比较 `param_groups[0]["lr"]`，容差 1e-15（覆盖 `mode`×`threshold_mode` 四象限、
  `patience=0/1/2`、`cooldown=0/2/3`、`min_lr` 地板、`eps=1e-3` 共 5 组配置 × 40 步固定指标流）。
- 独立重算：`_oracle` 用纯 Python 走一遍同样语义但写法不同的状态机（阈值写成参考比率/参考差值、
  地板用条件式而非 `max()`），容差 1e-12，用来卡住"照抄原生但顺序写错"的解。
- 示例数值由参考解实跑得到，与原生参照输出完全相同（逐位相等）。
- 退化/不变量：`cooldown=2` 的手算台阶序列；`min_lr` 到底后幂等；`old-new<=eps` 的首次降速被整体忽略；
  `patience=0` 的首步基线规则；0 维张量指标；`lr` 单调不增且 `\ge min_lr`；降速次数账本（恰好 5 次）。

## 复杂度
每次 `step(metrics)` 为 \(O(1)\)（每参数组常数次浮点比较），无状态张量、无随机性；
指标流长度 \(T\) 时总代价 \(O(T)\)，全程 Python float，判定可在毫秒级完成。
