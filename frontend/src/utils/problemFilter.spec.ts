import { describe, it, expect } from 'vitest'
import type { ProblemSummary } from '../api'
import { filterProblems, familyOf, buildSections, GROUP_ORDER, TAG_FAMILY } from './problemFilter'

function p(over: Partial<ProblemSummary> & { slug: string }): ProblemSummary {
  return {
    framework: 'pytorch',
    title: over.slug,
    difficulty: 'Easy',
    tags: ['operator'],
    solved: false,
    ...over,
  }
}

const list: ProblemSummary[] = [
  p({ slug: 'relu', title: 'ReLU', difficulty: 'Easy', tags: ['operator', 'activation'], solved: true }),
  p({ slug: 'rope', title: 'RoPE', difficulty: 'Hard', tags: ['operator', 'attention', 'numerical-stability'] }),
  p({ slug: 'gelu', title: 'GELU', difficulty: 'Medium', tags: ['operator', 'activation'] }),
  p({ slug: 'dataloader', title: 'DataLoader', difficulty: 'Medium', tags: ['framework', 'dataloader'], solved: true }),
]

describe('filterProblems', () => {
  it('关键词大小写不敏感匹配标题', () => {
    expect(filterProblems(list, { keyword: 'ge', difficulty: '', status: '' }).map((x) => x.slug))
      .toEqual(['gelu'])
  })
  it('关键词命中领域标签', () => {
    expect(filterProblems(list, { keyword: 'attention', difficulty: '', status: '' }).map((x) => x.slug))
      .toEqual(['rope'])
  })
  it('关键词命中 slug', () => {
    expect(filterProblems(list, { keyword: 'dataloader', difficulty: '', status: '' }).map((x) => x.slug))
      .toEqual(['dataloader'])
  })
  it('难度过滤', () => {
    expect(filterProblems(list, { keyword: '', difficulty: 'Easy', status: '' }).map((x) => x.slug))
      .toEqual(['relu'])
  })
  it('状态过滤 solved / unsolved', () => {
    expect(filterProblems(list, { keyword: '', difficulty: '', status: 'solved' }).map((x) => x.slug))
      .toEqual(['relu', 'dataloader'])
    expect(filterProblems(list, { keyword: '', difficulty: '', status: 'unsolved' }).length).toBe(2)
  })
  it('三重过滤取交集', () => {
    const r = filterProblems(list, { keyword: 'e', difficulty: 'Medium', status: 'unsolved' })
    expect(r.map((x) => x.slug)).toEqual(['gelu'])
  })
})

describe('familyOf', () => {
  it('非 operator 标签统一归 other', () => {
    expect(familyOf(p({ slug: 'x', tags: ['framework'] }))).toBe('other')
  })
  it('operator 但无领域标签也归 other', () => {
    expect(familyOf(p({ slug: 'x', tags: ['operator'] }))).toBe('other')
  })
  it('多领域标签取 GROUP_ORDER 排名最靠前者（attention 优先于 num-stability 兜底）', () => {
    const rope = p({ slug: 'rope', tags: ['operator', 'numerical-stability', 'attention'] })
    expect(familyOf(rope)).toBe('attention')
  })
})

describe('buildSections', () => {
  it('每个 TAG_FAMILY 族 key 都在 GROUP_ORDER 内（新合集登记后分组顺序完整）', () => {
    const fams = [...new Set(Object.values(TAG_FAMILY))]
    for (const f of fams) expect(GROUP_ORDER).toContain(f)
  })
  it('关闭分组时单段平铺，name 为空', () => {
    const s = buildSections(list, false)
    expect(s).toHaveLength(1)
    expect(s[0].name).toBe('')
    expect(s[0].items).toHaveLength(4)
  })
  it('开启分组时按 family 切段并按 GROUP_ORDER 排序', () => {
    const s = buildSections(list, true)
    const names = s.map((x) => x.name)
    // activation(2) 在 attention(1) 之前，other 最后
    expect(names).toEqual(['activation', 'attention', 'other'])
    expect(names.every((n) => n === '' || GROUP_ORDER.indexOf(n) >= 0)).toBe(true)
    const act = s.find((x) => x.name === 'activation')!
    expect(act.items.map((x) => x.slug)).toEqual(['relu', 'gelu'])
  })
})
