<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import type { ProblemSummary } from '../api'

// 题目表格 + 搜索/难度/状态筛选，首页「全部」标签与合集详情页共用
const props = defineProps<{
  problems: ProblemSummary[]
  loading?: boolean
  error?: string | null
  emptyText?: string
  persistKey?: string
}>()

// 行本身是 router-link，徽章用 span+push 避免 a 嵌套 a 的非法 HTML
const router = useRouter()
function goCollection(slug: string) {
  router.push(`/collections/${slug}`)
}

// persistKey 存在时筛选条件存 localStorage，切标签/离开页面后回来不丢
const DIFFS = ['', 'Easy', 'Medium', 'Hard'] as const
const STATUSES = ['', 'solved', 'unsolved'] as const

function filterKey(field: string) {
  return `leet-infra:filter:${props.persistKey}:${field}`
}
function restoreFilter<T extends string>(field: string, allowed: readonly T[], fallback: T): T {
  if (!props.persistKey) return fallback
  const v = localStorage.getItem(filterKey(field)) as T | null
  return v && allowed.includes(v) ? v : fallback
}

const keyword = ref(props.persistKey ? localStorage.getItem(filterKey('kw')) ?? '' : '')
const difficulty = ref<'' | 'Easy' | 'Medium' | 'Hard'>(restoreFilter('diff', DIFFS, ''))
const status = ref<'' | 'solved' | 'unsolved'>(restoreFilter('status', STATUSES, ''))

watch([keyword, difficulty, status], ([kw, d, s]) => {
  if (!props.persistKey) return
  localStorage.setItem(filterKey('kw'), kw)
  localStorage.setItem(filterKey('diff'), d)
  localStorage.setItem(filterKey('status'), s)
})

const filtered = computed(() =>
  props.problems.filter((p) => {
    if (keyword.value && !p.title.toLowerCase().includes(keyword.value.toLowerCase())) return false
    if (difficulty.value && p.difficulty !== difficulty.value) return false
    if (status.value === 'solved' && !p.solved) return false
    if (status.value === 'unsolved' && p.solved) return false
    return true
  })
)

// 列表本身为空用调用方文案；仅在筛选掉全部题目时才说「没有匹配」
const emptyMessage = computed(() =>
  props.problems.length ? '没有匹配的题目' : props.emptyText ?? '暂无题目'
)
</script>

<template>
  <div class="problem-table">
    <div class="toolbar">
      <input v-model="keyword" class="search" placeholder="搜索题目…" />
      <select v-model="difficulty">
        <option value="">全部难度</option>
        <option value="Easy">Easy</option>
        <option value="Medium">Medium</option>
        <option value="Hard">Hard</option>
      </select>
      <select v-model="status">
        <option value="">全部状态</option>
        <option value="solved">已解决</option>
        <option value="unsolved">未解决</option>
      </select>
    </div>

    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="error" class="empty error">加载失败：{{ error }}</div>
    <div v-else-if="!filtered.length" class="empty">{{ emptyMessage }}</div>

    <div class="table" v-else>
      <div class="row head">
        <span>状态</span><span>标题</span><span>难度</span><span>标签</span><span></span>
      </div>
      <router-link
        v-for="p in filtered"
        :key="p.slug"
        :to="`/problems/${p.slug}`"
        class="row"
      >
        <span class="status-dot" :class="{ done: p.solved }">{{ p.solved ? '✓' : '' }}</span>
        <span class="title">{{ p.title }}</span>
        <span><em class="badge" :class="p.difficulty">{{ p.difficulty }}</em></span>
        <span class="tags">
          {{ p.tags.join(' · ') }}
          <span
            v-for="col in p.collections ?? []"
            :key="col.slug"
            class="col-badge"
            :title="`合集：${col.name}`"
            @click.stop.prevent="goCollection(col.slug)"
          >
            ▤ {{ col.name }}
          </span>
        </span>
        <span class="go">解答 →</span>
      </router-link>
    </div>
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 18px;
}
.search {
  flex: 1;
}
.search,
select {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
  padding: 8px 12px;
  font: inherit;
  outline: none;
}
.search:focus {
  border-color: var(--accent);
}
.table {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--bg-panel);
}
.row {
  display: grid;
  grid-template-columns: 56px 1.4fr 90px 1.2fr 80px;
  align-items: center;
  gap: 12px;
  padding: 13px 18px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}
.row:last-child {
  border-bottom: none;
}
.row:hover {
  background: var(--bg-elevated);
}
.head {
  color: var(--text-dim);
  font-size: 12px;
}
.title {
  font-weight: 600;
}
.tags {
  color: var(--text-dim);
  font-size: 12px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.col-badge {
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text-dim);
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 11.5px;
  white-space: nowrap;
  cursor: pointer;
}
.col-badge:hover {
  color: var(--accent);
  border-color: var(--accent);
}
.go {
  color: var(--accent);
  font-size: 13px;
  text-align: right;
}
.status-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 1.5px solid var(--border);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: transparent;
}
.status-dot.done {
  border-color: var(--accent);
  background: var(--accent-dim);
  color: var(--accent);
}
.empty {
  text-align: center;
  color: var(--text-dim);
  padding: 60px 0;
}
.empty.error {
  color: var(--red);
}
</style>
