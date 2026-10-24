// 题目列表的筛选与 family 折叠分组纯逻辑（ProblemTable.vue 消费，单测覆盖）。
import type { ProblemSummary } from '../api'

export interface FilterState {
  keyword: string
  difficulty: '' | 'Easy' | 'Medium' | 'Hard'
  status: '' | 'solved' | 'unsolved'
}

/** 关键词（标题子串，大小写不敏感）+ 难度 + 解题状态三重过滤 */
export function filterProblems(list: ProblemSummary[], f: FilterState): ProblemSummary[] {
  const kw = f.keyword.toLowerCase()
  return list.filter((p) => {
    if (kw && !p.title.toLowerCase().includes(kw)) return false
    if (f.difficulty && p.difficulty !== f.difficulty) return false
    if (f.status === 'solved' && !p.solved) return false
    if (f.status === 'unsolved' && p.solved) return false
    return true
  })
}

// family 归组：operators 按领域标签归族，非 operator（框架题）统一归「框架 / 其他」
export const TAG_FAMILY: Record<string, string> = {
  activation: '激活函数',
  normalization: '归一化',
  conv: '卷积',
  cnn: '卷积',
  vision: '池化 · 视觉',
  pooling: '池化 · 视觉',
  gemm: '矩阵乘 · 线代',
  attention: '注意力',
  tensor: '张量操作',
  loss: '损失',
  optimizer: '优化器',
  quantization: '量化',
  sampling: '采样',
  rnn: '循环网络',
  metric: '检测 · 度量',
  systems: '显存 · 性能',
  distributed: '分布式',
}

export const GROUP_ORDER = [
  '激活函数', '归一化', '卷积', '池化 · 视觉', '矩阵乘 · 线代', '注意力',
  '张量操作', '损失', '优化器', '量化', '采样', '循环网络', '检测 · 度量', '显存 · 性能',
  '分布式', '框架 / 其他',
]

/** 取 tags 中在 GROUP_ORDER 里排名最靠前的族；规避不可靠的「首个非 operator 标签」 */
export function familyOf(p: ProblemSummary): string {
  if (!p.tags.includes('operator')) return '框架 / 其他'
  let best = ''
  let bestRank = Infinity
  for (const t of p.tags) {
    const fam = TAG_FAMILY[t]
    if (!fam) continue
    const r = GROUP_ORDER.indexOf(fam)
    if (r >= 0 && r < bestRank) {
      bestRank = r
      best = fam
    }
  }
  return best || '框架 / 其他'
}

export interface Section {
  name: string
  items: ProblemSummary[]
}

/** 分组开启时按 family 切段（按 GROUP_ORDER 排序），否则单段平铺（name 为空） */
export function buildSections(filtered: ProblemSummary[], groupOn: boolean): Section[] {
  if (!groupOn) return [{ name: '', items: filtered }]
  const buckets = new Map<string, ProblemSummary[]>()
  for (const p of filtered) {
    const fam = familyOf(p)
    const arr = buckets.get(fam)
    if (arr) arr.push(p)
    else buckets.set(fam, [p])
  }
  return [...buckets.entries()]
    .sort((a, b) => GROUP_ORDER.indexOf(a[0]) - GROUP_ORDER.indexOf(b[0]))
    .map(([name, items]) => ({ name, items }))
}
