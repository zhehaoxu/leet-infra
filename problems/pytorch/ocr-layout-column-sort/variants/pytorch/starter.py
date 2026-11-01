# PP-Structure recovery 双栏版面阅读顺序账
#
# PaddleOCR `ppstructure/recovery/recovery_to_doc.py` 的 `sorted_layout_boxes`
# 在导出 docx 前对版面区域做"先分栏、再左栏后右栏"的阅读顺序重排。
# 纯账本：所有 bbox 为 [x0, y0, x1, y1] 坐标入参，不读图。
#
# 实现 solve(bboxes, w)：
#
#   输入
#     bboxes  list[list[float]]  版面框 [x0, y0, x1, y1]，下标即原始顺序
#     w       float              页宽
#
#   输出 dict：
#     order    list[int]   重排后的原始下标序列（输出顺序）
#     layouts  list[str]   与 order 对齐，每框 "single" 或 "double"
#
# 规则（逐条照抄官方实现，浮点除法保留）：
#   1. len(bboxes) == 1：不重排（也不必排序），直接 order=[0]、layouts=["single"]。
#   2. 其余：先按 (y0, x0) 升序**稳定**排序得到扫描序列。
#   3. 维护左右两个桶（res_left / res_right），自左向右扫描：
#        a. 若当前框是序列最后一框：
#             - 若 y0 > 前一框的 y1 且 x0 < w/2 且 x1 > w/2
#               （末框与前一框之间有纵向间隙且横跨中线）：
#               先 flush 左桶再 flush 右桶，末框标 "single" 追加。
#             - 否则若 x1 > w/2：末框标 "double" 追加进右桶，
#               然后 flush 左桶、flush 右桶（末框在右桶组内）。
#             - 否则若 x0 < w/2：末框标 "double" 追加进左桶，
#               然后 flush 左桶、flush 右桶（末框在左桶组内）。
#             - 否则（仅可能 x0 == x1 == w/2 的退化零宽框）：**静默丢弃**，
#               且已攒的左右桶**一并丢弃不输出**。
#        b. 否则（非末框）：
#             - x0 < w/4 且 x1 < 3w/4：标 "double"，进左桶。
#             - 否则 x0 > w/4 且 x1 > w/2：标 "double"，进右桶。
#             - 否则：先 flush 左桶、flush 右桶，本框标 "single" 追加。
#   4. 所有比较均为严格不等号；flush 顺序恒为"先左桶后右桶"。
#      官方实现循环出口只有末框分支与空序列出口，出口后的残余 flush
#      实际不可达（桶在末框分支内必被处理或清空）。
#
# 示例：bboxes=[[50,100,300,200],[600,100,900,200],
#               [50,250,300,350],[600,250,900,350]], w=1000
#   两栏交替：扫描序 id0,id1,id2；id0→左桶，id1→右桶，id2→左桶；
#   id3 为末框：y0=250 不大于前框 y1=350，横跨中线不成立，x1>500 →
#   入右桶后 flush → order=[0,2,1,3]，layouts 全 "double"。
