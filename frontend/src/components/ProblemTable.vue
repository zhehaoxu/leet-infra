<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import type { ProblemSummary } from '../api'
import { filterProblems, buildSections } from '../utils/problemFilter'

// 题目表格 + 搜索/难度/状态筛选，首页「全部」标签与合集详情页共用
const props = defineProps<{
  problems: ProblemSummary[]
  loading?: boolean
  error?: string | null
  emptyText?: string
  persistKey?: string
  /** 允许按 family 折叠分组（数据量大时提升可导航性），工具栏会出现「分组」开关 */
  groupable?: boolean
  /** 合集详情页的实现方式偏好，注入行链接 ?variant= */
  variant?: string
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
// 分组开关：仅当父组件允许分组时生效；默认开启（除非用户显式关过并存了 '0'）
const groupOn = ref<boolean>(
  props.groupable && (props.persistKey ? localStorage.getItem(filterKey('group')) !== '0' : true),
)

watch([keyword, difficulty, status, groupOn], ([kw, d, s, g]) => {
  if (!props.persistKey) return
  localStorage.setItem(filterKey('kw'), kw)
  localStorage.setItem(filterKey('diff'), d)
  localStorage.setItem(filterKey('status'), s)
  localStorage.setItem(filterKey('group'), g ? '1' : '0')
})

const filtered = computed(() =>
  filterProblems(props.problems, {
    keyword: keyword.value,
    difficulty: difficulty.value,
    status: status.value,
  })
)

// 折叠状态（组名集合）；整体替换 Set 以触发响应式
const collapsed = ref<Set<string>>(new Set())
function toggleGroup(name: string) {
  const next = new Set(collapsed.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  collapsed.value = next
}

// sections：分组开启时按 family 切段（按 GROUP_ORDER 排序），否则单段平铺
const sections = computed(() => buildSections(filtered.value, groupOn.value))

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
      <label v-if="groupable" class="group-toggle" title="按 family 分组折叠">
        <input type="checkbox" v-model="groupOn" />
        <span>分组</span>
      </label>
    </div>

    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="error" class="empty error">加载失败：{{ error }}</div>
    <div v-else-if="!filtered.length" class="empty">{{ emptyMessage }}</div>

    <div class="table" v-else>
      <div class="row head">
        <span>状态</span><span>标题</span><span>难度</span><span>标签</span><span></span>
      </div>
      <template v-for="sec in sections" :key="sec.name || '__flat'">
        <button
          v-if="sec.name"
          class="group-head"
          @click="toggleGroup(sec.name)"
        >
          <span class="chev">{{ collapsed.has(sec.name) ? '▸' : '▾' }}</span>
          <span class="gname">{{ sec.name }}</span>
          <span class="gcount">{{ sec.items.length }}</span>
        </button>
        <template v-if="!sec.name || !collapsed.has(sec.name)">
          <router-link
            v-for="p in sec.items"
            :key="p.slug"
            :to="{ path: `/problems/${p.slug}`, query: variant ? { variant } : {} }"
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
        </template>
      </template>
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
.group-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  color: var(--text-dim);
  font-size: 13px;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
.group-toggle:hover {
  color: var(--text);
  border-color: var(--accent);
}
.group-head {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  text-align: left;
  padding: 10px 18px;
  background: var(--bg-elevated);
  border: none;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  font: inherit;
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
}
.group-head:hover {
  color: var(--accent);
}
.group-head .chev {
  color: var(--text-dim);
  width: 12px;
}
.group-head .gcount {
  margin-left: auto;
  color: var(--text-dim);
  font-weight: 400;
  font-size: 12px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0 8px;
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
