// 代码草稿持久化的 key 规则（ProblemWorkbench.vue 消费，单测覆盖）。
export const DRAFT_PREFIX = 'leet-infra:draft:'

/** 草稿按 题目×实现 隔离；pytorch 沿用无前缀旧 key（旧题作废后无碰撞） */
export function draftKey(slug: string, variant: string): string {
  return variant && variant !== 'pytorch' ? `${DRAFT_PREFIX}${slug}:${variant}` : DRAFT_PREFIX + slug
}
