import { describe, it, expect } from 'vitest'
import { readHomeTab, writeHomeTab, HOME_TAB_KEY } from './uiPrefs'

function fakeStore(initial: Record<string, string> = {}) {
  const data = { ...initial }
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => {
      data[k] = v
    },
  }
}

describe('首页标签记忆', () => {
  it('缺省回落 all', () => {
    expect(readHomeTab(fakeStore())).toBe('all')
  })
  it('脏值也回落 all', () => {
    expect(readHomeTab(fakeStore({ [HOME_TAB_KEY]: 'garbage' }))).toBe('all')
  })
  it('存 collections 则读回 collections', () => {
    const s = fakeStore()
    writeHomeTab('collections', s)
    expect(s.data[HOME_TAB_KEY]).toBe('collections')
    expect(readHomeTab(s)).toBe('collections')
  })
  it('写 all 后读回 all', () => {
    const s = fakeStore({ [HOME_TAB_KEY]: 'collections' })
    writeHomeTab('all', s)
    expect(readHomeTab(s)).toBe('all')
  })
})
