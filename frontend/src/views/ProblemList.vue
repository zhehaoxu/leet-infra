<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useProblemStore } from '../stores/problems'
import { api, type CollectionSummary } from '../api'
import ProblemTable from '../components/ProblemTable.vue'
import ProgressRing from '../components/ProgressRing.vue'
import { readHomeTab, writeHomeTab, type HomeTab } from '../utils/uiPrefs'

const store = useProblemStore()
const activeTab = ref<HomeTab>(readHomeTab())

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

function setTab(tab: HomeTab) {
  activeTab.value = tab
  writeHomeTab(tab)
  if (tab === 'collections') ensureCollections()
}

async function resetProgress() {
  if (!window.confirm('清空全部解题进度？题目代码草稿不受影响。此操作不可撤销。')) return
  try {
    await api.resetProgress()
  } catch (e) {
    window.alert(`重置失败：${e instanceof Error ? e.message : e}`)
    return
  }
  await store.fetchProblems()
  if (activeTab.value === 'collections') {
    collections.value = []
    ensureCollections()
  }
}

// ---- 进度导出/导入（换机/重装不丢记录）----
const importInput = ref<HTMLInputElement | null>(null)

async function exportProgress() {
  try {
    const blob = await api.exportProgress()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `leet-infra-progress-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  } catch (e) {
    window.alert(`导出失败：${e instanceof Error ? e.message : e}`)
  }
}

function triggerImport() {
  importInput.value?.click()
}

async function onImportFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = '' // 允许连续选同一文件
  if (!file) return
  if (!window.confirm('导入将覆盖当前全部解题进度，继续？')) return
  try {
    const data = JSON.parse(await file.text()) as unknown
    const r = await api.importProgress(data)
    await store.fetchProblems()
    if (activeTab.value === 'collections') {
      collections.value = []
      ensureCollections()
    }
    window.alert(`导入完成：恢复 ${r.imported} 题` + (r.skipped ? `，跳过不认识的条目 ${r.skipped} 个` : ''))
  } catch (e) {
    window.alert(`导入失败：${e instanceof Error ? e.message : e}，请确认选择的是 leet-infra 导出的进度文件`)
  }
}

const filteredCollections = computed(() =>
  collections.value.filter((c) => {
    if (!collectionKeyword.value) return true
    const kw = collectionKeyword.value.toLowerCase()
    return c.name.toLowerCase().includes(kw) || c.description.toLowerCase().includes(kw)
  })
)
</script>

<template>
  <div class="list-page">
    <div class="hero">
      <h1>AI Infra 从动手写开始</h1>
      <p>
        PyTorch 题库 · 代码在你本机 Python 虚拟环境中真实运行 · {{ store.solvedCount }}/{{ store.problems.length }} 已解决
        <button
          v-if="store.solvedCount > 0"
          class="reset-progress"
          title="清空全部已解决记录，用于二刷题目"
          @click="resetProgress"
        >
          重置进度
        </button>
        <button class="io-progress" title="下载进度文件 progress.json" @click="exportProgress">导出进度</button>
        <button class="io-progress" title="从导出的 JSON 恢复进度（覆盖当前）" @click="triggerImport">导入进度</button>
        <input ref="importInput" type="file" accept="application/json,.json" class="hidden-file" @change="onImportFile" />
      </p>
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
      groupable
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
            <ProgressRing :value="c.solved_count" :total="c.problem_count" />
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
.reset-progress {
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 12.5px;
  text-decoration: underline;
  text-underline-offset: 3px;
  cursor: pointer;
  margin-left: 10px;
}
.reset-progress:hover {
  color: var(--red);
}
.io-progress {
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 12.5px;
  text-decoration: underline;
  text-underline-offset: 3px;
  cursor: pointer;
  margin-left: 10px;
}
.io-progress:hover {
  color: var(--accent);
}
.hidden-file {
  display: none;
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
