<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useProblemStore } from '../stores/problems'
import { api, type CollectionSummary } from '../api'
import ProblemTable from '../components/ProblemTable.vue'

const store = useProblemStore()
const HOME_TAB_KEY = 'leet-infra:home-tab'
const activeTab = ref<'all' | 'collections'>(
  localStorage.getItem(HOME_TAB_KEY) === 'collections' ? 'collections' : 'all'
)

// 合集标签的搜索词（全部标签的题目搜索在 ProblemTable 组件内部）
const collectionKeyword = ref('')

// 合集列表：首次切到合集标签时拉取
const collections = ref<CollectionSummary[]>([])
const collectionsLoading = ref(false)
const collectionsError = ref<string | null>(null)

onMounted(() => {
  if (!store.problems.length) store.fetchProblems()
  if (activeTab.value === 'collections') ensureCollections()
})

async function ensureCollections() {
  if (collections.value.length || collectionsLoading.value) return
  collectionsLoading.value = true
  collectionsError.value = null
  try {
    collections.value = await api.collections()
  } catch (e) {
    collectionsError.value = e instanceof Error ? e.message : String(e)
  } finally {
    collectionsLoading.value = false
  }
}

function setTab(tab: 'all' | 'collections') {
  activeTab.value = tab
  localStorage.setItem(HOME_TAB_KEY, tab)
  if (tab === 'collections') ensureCollections()
}

const filteredCollections = computed(() =>
  collections.value.filter((c) => {
    if (!collectionKeyword.value) return true
    const kw = collectionKeyword.value.toLowerCase()
    return c.name.toLowerCase().includes(kw) || c.description.toLowerCase().includes(kw)
  })
)

// 圆环进度：r=9 的周长 × 完成比例
const RING_C = 2 * Math.PI * 9
function ringDash(c: CollectionSummary) {
  const ratio = c.problem_count ? c.solved_count / c.problem_count : 0
  return `${ratio * RING_C} ${RING_C}`
}
</script>

<template>
  <div class="list-page">
    <div class="hero">
      <h1>AI Infra 从动手写开始</h1>
      <p>PyTorch 题库 · 代码在你本机 Python 虚拟环境中真实运行 · {{ store.solvedCount }}/{{ store.problems.length }} 已解决</p>
    </div>

    <div class="tabs">
      <button class="tab" :class="{ active: activeTab === 'all' }" @click="setTab('all')">全部</button>
      <button
        class="tab"
        :class="{ active: activeTab === 'collections' }"
        @click="setTab('collections')"
      >
        合集
      </button>
    </div>

    <ProblemTable
      v-if="activeTab === 'all'"
      :problems="store.problems"
      :loading="store.loading"
      :error="store.error"
      persist-key="home-all"
    />

    <template v-else>
      <div class="toolbar">
        <input v-model="collectionKeyword" class="search" placeholder="搜索合集…" />
      </div>
      <div v-if="collectionsLoading" class="empty">加载中…</div>
      <div v-else-if="collectionsError" class="empty error">加载失败：{{ collectionsError }}</div>
      <div v-else-if="!filteredCollections.length" class="empty">没有匹配的合集</div>

      <div class="table" v-else>
        <router-link
          v-for="c in filteredCollections"
          :key="c.slug"
          :to="`/collections/${c.slug}`"
          class="row collection-row"
        >
          <span class="collection-mark" :title="`${c.solved_count}/${c.problem_count} 已解决`">
            <svg width="24" height="24" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="9" fill="none" stroke="var(--border)" stroke-width="3" />
              <circle
                v-if="c.solved_count > 0"
                cx="12"
                cy="12"
                r="9"
                fill="none"
                stroke="var(--accent)"
                stroke-width="3"
                :stroke-dasharray="ringDash(c)"
                :stroke-linecap="c.solved_count === c.problem_count ? 'butt' : 'round'"
                transform="rotate(-90 12 12)"
              />
            </svg>
          </span>
          <span class="collection-body">
            <span class="title">{{ c.name }}</span>
            <span class="collection-desc">{{ c.description }}</span>
          </span>
          <span class="collection-progress" :class="{ done: c.solved_count === c.problem_count }">
            {{ c.solved_count }}/{{ c.problem_count }}
          </span>
          <span class="go">进入合集 →</span>
        </router-link>
      </div>
    </template>
  </div>
</template>

<style scoped>
.list-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 24px;
}
.hero h1 {
  font-size: 26px;
  margin: 0 0 8px;
}
.hero p {
  color: var(--text-dim);
  margin: 0 0 28px;
}
.tabs {
  display: flex;
  gap: 18px;
  margin-bottom: 16px;
}
.tab {
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 14px;
  padding: 6px 2px;
  cursor: pointer;
}
.tab:hover {
  color: var(--text);
}
.tab.active {
  color: var(--accent);
}
.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 18px;
}
.search {
  flex: 1;
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
.row.collection-row {
  grid-template-columns: 40px 1.6fr 90px 100px;
}
.collection-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.collection-body {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}
.collection-desc {
  color: var(--text-dim);
  font-size: 12.5px;
}
.collection-progress {
  color: var(--text-dim);
  font-size: 13px;
  font-family: var(--font-mono);
}
.collection-progress.done {
  color: var(--accent);
}
.title {
  font-weight: 600;
}
.go {
  color: var(--accent);
  font-size: 13px;
  text-align: right;
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
