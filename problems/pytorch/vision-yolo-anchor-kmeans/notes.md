# 出题笔记：vision-yolo-anchor-kmeans

## 回源

AlexeyAB/darknet `scripts/kmeansiou.c`（391 行全文）：`closest_center`/`kmeans_expectation`/`kmeans_maximization`/`WCSS`/`smart_centers`/`do_kmeans` 逐一核对。`dist()` 里 `inter = min·min`、`return 1 - iou`。原版 pjreddie darknet 无 k-means 代码（论文作者离线跑），AlexeyAB 版是社区事实标准。

## 关键口径判断

- **空簇 quirk 是意外收获**：`kmeans_maximization` 里 `zero_matrix(centers)` 后 `if(counts[i])` 才除——空簇质心就是 (0,0)，不是重采样也不是丢弃。首版测试想用 k=2 init=[0,0] 触发它，但 assignments 从 calloc 全 0 起、首 E 即收敛，quirk 根本不执行；k=3 init=[0,1,1] 才真触发（M 步 cluster2 空）。
- **k=1 特判的因果**：全 0 assignments + 全分到 0 → expectation 恒收敛 → 没有 M 步 → 质心停留随机初始值。官方 `if(k==1) kmeans_maximization(...)` 就是补这个洞。变异 `k1-no-special` 钉住。
- **WCSS 名不副实**：`sum += (1 - dist(...))` 即 IoU 均值——官方函数名 WCSS 但算的是 avg IoU，转写按代码不按名。
- **smart_centers（D² 采样）不入题**：依赖 rand() 流，转写为显式下标入参；notes 声明。

## 边界划定

- 不做 `calc_anchors` 的 tries 多重启（依赖随机流）；不做 k-means++ D² 采样本身；不做 darknet 的 `-c` 计数输出。
- 与 `vision-anchor-match` 划界：那是推理期 anchor-GT 匹配，这是训练前聚类。
- 变异教训：`inter-max`（min→max）会把 IoU 变成 >1 的怪值，破坏 Lloyd 单调收敛 → 官方无 max_iter → **harness 死循环挂起**。换成缩放保序变异 `*0.5`（有界、保序、可收敛，靠锚点 avg_iou 抓）。写「破坏距离度量」类变异前必须确认其收敛性。

## 测试设计

- oracle `_py` 独立 Lloyd 重放（E/M 各自重写）。
- 9 个 check：双簇锚点（手算 avg_iou 分数）、quirk 未触发、quirk 触发、k=1 特判、并列取小、iou 边界、随机 150 组、非法输入 6 档。
- 变异 9/9：inter 缩放、union 不减、并列取大、空簇改 (1,1)、k=1 特判去除、均值 +1、WCSS 分母 +1、init 宽高互换、收敛检测短路。
