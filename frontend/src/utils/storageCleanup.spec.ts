import { describe, it, expect } from 'vitest'
import {
  removeKeysWithPrefixes,
  cleanupOrphanStorage,
  ORPHAN_PREFIXES,
  CLEANUP_FLAG,
  type KVStore,
} from './storageCleanup'

function memStore(entries: Record<string, string> = {}): KVStore {
  const data = new Map<string, string>(Object.entries(entries))
  return {
    get length() {
      return data.size
    },
    key: (i: number) => [...data.keys()][i] ?? null,
    getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  }
}

describe('removeKeysWithPrefixes', () => {
  it('只删命中前缀的键，其余原样保留', () => {
    const s = memStore({
      'leet-infra:notes:relu': 'hi',
      'leet-infra:notes:softmax': 'x',
      'leet-infra:draft:relu': 'code',
      'leet-infra:home-tab': 'all',
    })
    expect(removeKeysWithPrefixes(ORPHAN_PREFIXES, s)).toBe(2)
    expect(s.getItem('leet-infra:notes:relu')).toBeNull()
    expect(s.getItem('leet-infra:notes:softmax')).toBeNull()
    expect(s.getItem('leet-infra:draft:relu')).toBe('code')
    expect(s.getItem('leet-infra:home-tab')).toBe('all')
  })

  it('无命中返回 0 且不动任何键', () => {
    const s = memStore({ 'leet-infra:draft:relu': 'code' })
    expect(removeKeysWithPrefixes(ORPHAN_PREFIXES, s)).toBe(0)
    expect(s.getItem('leet-infra:draft:relu')).toBe('code')
  })

  it('存储遍历抛异常时静默返回 0', () => {
    const boom: KVStore = {
      get length(): number {
        throw new Error('denied')
      },
      key: () => null,
      getItem: () => null,
      setItem: () => {},
      removeItem: () => {},
    }
    expect(removeKeysWithPrefixes(['x:'], boom)).toBe(0)
  })
})

describe('cleanupOrphanStorage', () => {
  it('首次执行清孤儿并打标记，第二次 O(1) 跳过', () => {
    const s = memStore({ 'leet-infra:notes:relu': 'hi', 'leet-infra:draft:relu': 'code' })
    expect(cleanupOrphanStorage(s)).toBe(1)
    expect(s.getItem('leet-infra:notes:relu')).toBeNull()
    expect(s.getItem('leet-infra:draft:relu')).toBe('code') // 现存键不误伤
    expect(s.getItem(CLEANUP_FLAG)).toBe('1')
    expect(cleanupOrphanStorage(s)).toBe(0) // 已清理过
  })

  it('无孤儿时也置标记（幂等，后续不再扫描）', () => {
    const s = memStore({ 'leet-infra:draft:x': 'c' })
    expect(cleanupOrphanStorage(s)).toBe(0)
    expect(s.getItem(CLEANUP_FLAG)).toBe('1')
  })

  it('localStorage 不可读时不崩溃', () => {
    const denied: KVStore = {
      length: 0,
      key: () => null,
      getItem: () => {
        throw new Error('access denied')
      },
      setItem: () => {},
      removeItem: () => {},
    }
    expect(cleanupOrphanStorage(denied)).toBe(0)
  })
})
