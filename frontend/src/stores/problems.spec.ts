import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useProblemStore } from './problems'
import { api } from '../api'

vi.mock('../api', () => ({
  api: { listProblems: vi.fn() },
}))

describe('problems store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('markSolved / unmarkSolved 更新 solvedCount', () => {
    const s = useProblemStore()
    s.problems = [
      { slug: 'a', title: 'A', difficulty: 'Easy', tags: [], solved: false } as never,
      { slug: 'b', title: 'B', difficulty: 'Easy', tags: [], solved: false } as never,
    ]
    expect(s.solvedCount).toBe(0)
    s.markSolved('a')
    expect(s.solvedCount).toBe(1)
    s.markSolved('a') // 幂等
    expect(s.solvedCount).toBe(1)
    s.unmarkSolved('a')
    expect(s.solvedCount).toBe(0)
    // 未知 slug 静默无副作用
    s.markSolved('nope')
    expect(s.solvedCount).toBe(0)
  })

  it('fetchProblems 成功写入并清 error/loading', async () => {
    ;(api.listProblems as ReturnType<typeof vi.fn>).mockResolvedValue([
      { slug: 'x', title: 'X', difficulty: 'Medium', tags: [], solved: true },
    ])
    const s = useProblemStore()
    await s.fetchProblems()
    expect(s.problems).toHaveLength(1)
    expect(s.solvedCount).toBe(1)
    expect(s.error).toBeNull()
    expect(s.loading).toBe(false)
  })

  it('fetchProblems 失败置 error 并结束 loading', async () => {
    ;(api.listProblems as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('boom'))
    const s = useProblemStore()
    await s.fetchProblems()
    expect(s.error).toBe('boom')
    expect(s.loading).toBe(false)
  })
})
