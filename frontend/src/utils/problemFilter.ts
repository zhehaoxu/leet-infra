// 题目列表的筛选与 family 折叠分组纯逻辑（ProblemTable.vue 消费，单测覆盖）。
import type { ProblemSummary } from '../api'

export interface FilterState {
  keyword: string
  difficulty: '' | 'Easy' | 'Medium' | 'Hard'
  status: '' | 'solved' | 'unsolved'
}

/** 关键词（对标题 + slug + 领域标签拼接后子串匹配，大小写不敏感）+ 难度 + 解题状态三重过滤 */
export function filterProblems(list: ProblemSummary[], f: FilterState): ProblemSummary[] {
  const kw = f.keyword.trim().toLowerCase()
  return list.filter((p) => {
    if (kw) {
      const hay = `${p.title} ${p.slug} ${p.tags.join(' ')}`.toLowerCase()
      if (!hay.includes(kw)) return false
    }
    if (f.difficulty && p.difficulty !== f.difficulty) return false
    if (f.status === 'solved' && !p.solved) return false
    if (f.status === 'unsolved' && p.solved) return false
    return true
  })
}

// family 归组：operators 按领域标签归族，非 operator（框架题）统一归 other。
// 返回稳定的族 key，展示文案由 i18n 的 family.* 提供。
export type FamilyKey =
  | 'activation'
  | 'normalization'
  | 'conv'
  | 'vision'
  | 'detection'
  | 'gemm'
  | 'attention'
  | 'tensor'
  | 'loss'
  | 'autograd'
  | 'numerics'
  | 'optimizer'
  | 'quantization'
  | 'sampling'
  | 'rnn'
  | 'metric'
  | 'systems'
  | 'tokenization'
  | 'context'
  | 'distributed'
  | 'llm'
  | 'moe'
  | 'inference'
  | 'alignment'
  | 'kvcache'
  | 'speculative'
  | 'peft'
  | 'diffusion'
  | 'sparsity'
  | 'multimodal'
  | 'audio'
  | 'retrieval'
  | 'evaluation'
  | 'rl'
  | 'agentic'
  | 'linear-attention'
  | 'interpretability'
  | 'reasoning'
  | 'ocr'
  | 'asr'
  | 'other'

export const TAG_FAMILY: Record<string, FamilyKey> = {
  activation: 'activation',
  normalization: 'normalization',
  conv: 'conv',
  cnn: 'conv',
  vision: 'vision',
  pooling: 'vision',
  detection: 'detection',
  gemm: 'gemm',
  attention: 'attention',
  tensor: 'tensor',
  loss: 'loss',
  autograd: 'autograd',
  numerics: 'numerics',
  optimizer: 'optimizer',
  quantization: 'quantization',
  sampling: 'sampling',
  rnn: 'rnn',
  metric: 'metric',
  systems: 'systems',
  tokenization: 'tokenization',
  context: 'context',
  distributed: 'distributed',
  llm: 'llm',
  moe: 'moe',
  inference: 'inference',
  alignment: 'alignment',
  kvcache: 'kvcache',
  speculative: 'speculative',
  peft: 'peft',
  diffusion: 'diffusion',
  sparsity: 'sparsity',
  multimodal: 'multimodal',
  audio: 'audio',
  retrieval: 'retrieval',
  evaluation: 'evaluation',
  rl: 'rl',
  agentic: 'agentic',
  'linear-attention': 'linear-attention',
  interpretability: 'interpretability',
  reasoning: 'reasoning',
  ocr: 'ocr',
  asr: 'asr',
}

export const GROUP_ORDER: FamilyKey[] = [
  'activation', 'normalization', 'conv', 'vision', 'detection', 'gemm', 'attention',
  'tensor', 'loss', 'autograd', 'numerics', 'optimizer', 'quantization', 'sampling', 'rnn', 'metric', 'systems',
  'tokenization', 'context', 'distributed', 'llm', 'moe', 'inference', 'alignment', 'kvcache', 'speculative', 'peft', 'diffusion', 'sparsity', 'multimodal', 'audio', 'retrieval', 'evaluation', 'rl', 'agentic',
  'linear-attention', 'interpretability', 'reasoning', 'ocr', 'asr',
  'other',
]

/** 取 tags 中在 GROUP_ORDER 里排名最靠前的族；规避不可靠的「首个非 operator 标签」 */
export function familyOf(p: ProblemSummary): FamilyKey {
  if (!p.tags.includes('operator')) return 'other'
  let best: FamilyKey = 'other'
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
  return best
}

export interface Section {
  /** family key；关闭分组时为空字符串 */
  name: FamilyKey | ''
  items: ProblemSummary[]
}

/** 分组开启时按 family 切段（按 GROUP_ORDER 排序），否则单段平铺（name 为空） */
export function buildSections(filtered: ProblemSummary[], groupOn: boolean): Section[] {
  if (!groupOn) return [{ name: '', items: filtered }]
  const buckets = new Map<FamilyKey, ProblemSummary[]>()
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
