import { describe, it, expect } from 'vitest'
import { draftKey, DRAFT_PREFIX } from './draft'

describe('draftKey 草稿隔离', () => {
  it('pytorch 沿用无前缀旧 key', () => {
    expect(draftKey('relu', 'pytorch')).toBe(DRAFT_PREFIX + 'relu')
    expect(draftKey('relu', '')).toBe(DRAFT_PREFIX + 'relu')
  })
  it('非 pytorch 实现按 :variant 后缀隔离', () => {
    expect(draftKey('relu', 'triton')).toBe(`${DRAFT_PREFIX}relu:triton`)
    expect(draftKey('rope', 'cuda')).toBe(`${DRAFT_PREFIX}rope:cuda`)
  })
  it('同题不同实现互不覆盖', () => {
    expect(draftKey('relu', 'pytorch')).not.toBe(draftKey('relu', 'triton'))
  })
})
