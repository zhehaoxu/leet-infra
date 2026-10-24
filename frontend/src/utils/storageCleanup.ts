// 一次性清理历史遗留的孤儿 localStorage 键。
// 笔记标签改为只读渲染 notes.md 后，早期「可编辑笔记」写入的 `leet-infra:notes:<slug>`
// 已无任何代码引用（见 utils 现存前缀：draft / filter / variant / home-tab / editor-fontsize）。

/** 需要清理的孤儿键前缀（只增不删，供后续遗留键复用此机制）。 */
export const ORPHAN_PREFIXES = ['leet-infra:notes:']

/** localStorage 的最小结构化子集，便于注入假存储做单测。 */
export interface KVStore {
  readonly length: number
  key(index: number): string | null
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

/** 删除所有命中给定前缀的键，返回删除条数；存储异常（隐私模式/配额）静默。 */
export function removeKeysWithPrefixes(prefixes: string[], store: KVStore = localStorage): number {
  try {
    // 先快照待删键名再删，避免遍历中修改存储导致索引错位
    const doomed: string[] = []
    for (let i = 0; i < store.length; i++) {
      const k = store.key(i)
      if (k && prefixes.some((p) => k.startsWith(p))) doomed.push(k)
    }
    for (const k of doomed) store.removeItem(k)
    return doomed.length
  } catch {
    return 0
  }
}

/** 一次性清理已执行的哨兵标记键（导出供单测断言）。 */
export const CLEANUP_FLAG = 'leet-infra:storage-cleanup-v1'

/**
 * 应用启动时的一次性孤儿清理。跑过即打标记，后续启动 O(1) 跳过、不再扫描。
 * 返回本次删除条数（已清理过则 0）。
 */
export function cleanupOrphanStorage(store: KVStore = localStorage): number {
  try {
    if (store.getItem(CLEANUP_FLAG) !== null) return 0
  } catch {
    return 0
  }
  const removed = removeKeysWithPrefixes(ORPHAN_PREFIXES, store)
  try {
    store.setItem(CLEANUP_FLAG, '1')
  } catch {
    // 标记写入失败也接受：下次启动会重扫，仍幂等
  }
  return removed
}
