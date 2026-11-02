# notes — ocr-dbnet-unclip-thresh（order 718）

## 回源

- 权威源：PaddlePaddle/PaddleOCR `ppocr/postprocess/db_postprocess.py`（main，2026）。DBPostProcess 默认 thresh=0.3、box_thresh=0.7、min_size=3、unclip_ratio=2.0、score_mode="fast"。
- **三库语义实测**（临时环境 shapely 2.2.0 + pyclipper 1.4.0 + cv2 5.0.0，判题环境三者皆无）：
  - shapely `Polygon.area` = |shoelace|（顶点序无关，cw 也是正）；`Polygon.length` = 闭合周长；退化（共线）area=0。
  - pyclipper（Clipper 整数库）：**AddPath 输入坐标向零截断**（C cast；10.7→10），**Execute 输出坐标 round-half-away**（floor(v+0.5)；50.5→51、-10.9→-11——与 numpy 银行家 50.5→50 相反）。JT_ROUND 对矩形输出角部圆弧多边形（点数 5~24 随距离），**bbox 极值 = trunc 输入边 ± distance 后取整，整数 box 时精确**；非整数 box 坐标有圆弧离散 ±1 误差（probe：box [10.7,50.9] dist 3.3 → x_hi 53 而理论 54），故用例钉死整数/半整坐标。
  - cv2 `fillPoly`：坐标 astype(int32) 向零截断后光栅化，轴对齐整数矩形 = **全闭区域**（含右/下边界，probe [2,6]×[1,4] → mask_sum 20 = 5×4 全含）；负坐标向零截断（-3.7→-3）且超出 mask 区域部分裁剪；单点 polygon sum=1。`cv2.mean(img, mask)` = 掩码均值，**全零 mask 返回 0.0**。
  - `cv2.minAreaRect` 轴对齐整数输入精确；boxPoints 有 float32 噪声（-9e-17）且旋转矩形 size 带 float32 误差——旋转框不入题（notes）。
- **官方可达域观察**：contour 坐标 ∈ [0,W-1]×[0,H-1] 时 ceil(max) ≤ W-1 恒成立——bbox 上界 clip 是死防御；下界 clip 同理。本题作为域约束放宽（负坐标入题）时下界 clip 才生效，上界越界会 IndexError（官方 numpy 切片静默截断）——用例不造上界越界输入。
- **门序**：sside_in 门 → box_score_fast → box_thresh 门 → unclip（distance/expansion）→ sside_out 门 → dest 缩放。gate1/box_thresh 失败时后续账面 None（官方 continue 语义）；mask 全 0（框全出图）score=0.0 → box 门吞掉。
- 官方 `if len(box) > 1: continue`（Execute 返回多路径丢弃）在轴对齐正常输入下不触发，不入题记 notes。
- dest 缩放分母是 **prob map 的 W/H**（bitmap.shape），不是 src 图尺寸；round 为 numpy 银行家舍入。

## 测试设计

- oracle = numpy 向量化独立实现（meshgrid mask、np.roll shoelace/周长、np.rint 缩放）；score/distance 容差 1e-9（同 float64 不同求和顺序 ULP 差）。
- 常数区域一律用二进制精确值（0.75/0.5/0.25）——0.8 累加 16 次 = 0.8000000000000002，`== 0.8` 断言挂（首版踩过）。
- 银行家锚点：expanded 坐标只有 bx_lo/bx_hi 两值，构造 bx_hi/W*dw = 0.5（box [−1,0]、ratio 4 → bx_hi=1，W=6、dw=3）→ round=0，half-up 变异给 1。
- sside_out 恰 4 用例（ratio 0.225）：+1 span 变异（span 误当像素数）跨 min_size+2 门 → caught；span 恰 3 用例不够（4 仍 <5）。
- 随机 200 组：坐标整数或 .5 半整（半整 trunc 精确）；ratio/门参数/dest 随机；perim=0（min_size=0 + 单点）ValueError 路径单独锚定。

## 变异（15 全抓 + 1 等价）

- 全抓：thresh 严格性、bbox floor→ceil、bbox 下界 clip-off（负索引回绕）、mask trunc→floor（负坐标）、mask 并 bbox（两套坐标口径）、mask 丢上界行、|shoelace| 摘除（cw）、周长漏闭合边、distance 减半、pc_round floor 化、sside_out +1、min_size+2→+1、银行家→half-up、dest 竖向 clip-off、box_thresh >→>=。
- 等价变异 1 条（SURVIVED 记此）：bbox 上界 clip 的 min/max 交换次序——数学等价。
- **教训**：_compare 首版把 score/distance isclose 后置 0 再整体 ==——返回值被篡改致下游手算断言全挂（比较与返回分离）。

## 坑

- 用例 box 超出 prob map（y1=4 而 H=4）→ solve 循环 IndexError（官方 cv2 切片静默）——box 限官方可达域。
- 手算断言里 ratio 写错（1.5 vs 默认 2.0）两轮假失败——锚点断言与用例参数同行核对。
- 与 668（ctc-greedy）/691（layout-column-sort）/692（olmocr-merge）划界：本题管 DB 检测后处理的得分/外扩/门序；判题环境无 cv2/shapely/pyclipper，全部标量化。
