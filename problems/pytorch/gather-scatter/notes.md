# Gather / Scatter 详解

## 一对互逆的「按索引搬数据」

- **gather**：从 `input` 沿 `dim` 按 `index` **取**元素。
  $$ \text{out}[i,\dots]=\text{input}[\text{index}[i,\dots]\ \text{在}\ dim\ \text{轴}] $$
  规则：`out`、`index` **形状相同**；除 `dim` 外各维必须与 input 对应维一致；只有 `dim` 轴可以不同（从那里挑）。
- **scatter**：把 `src` 按 `index` **写**进新张量。
  $$ \text{out}[\text{index}[i]]=\text{src}[i] $$
  one-hot / index_put 都靠它：`zeros.scatter_(dim, index, 1)` 造 one-hot。

## dim 语义（本关最易错）

`gather(dim=1)` 对 2D `(row, col)` 是「第 row 行、取第 index[row,col] 列」；`dim=0` 则跨行选。理解成「沿哪个轴挑选、其余轴位置固定」。负 `dim` 要先归一 `dim += ndim`。

## 手写法（本参考解）：meshgrid 高级索引

对 `index` 的每一维建 `arange` 坐标网格（`indexing="ij"`），把 `dim` 那一层网格替换成 `index`，再 `input[tuple(gather_index)]`。这精确复刻 torch `gather` 语义，也最能说明「其他维用坐标、dim 维用 index」。

## 坑

1. **index 必须同 rank、且非 dim 维形状匹配**，否则 broadcast/形状报错。
2. `index` 需 `long` 类型（`torch.gather` 要求整型索引）。
3. scatter 的 `dim_` 后缀是**原地**版本；重复索引时写入顺序不确定（要累加用 `scatter_add_`）。
4. gather 不连续内存/多设备时注意 `device` 一致。
