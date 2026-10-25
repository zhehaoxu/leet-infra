<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useProblemStore } from '../stores/problems'
import { api, type CollectionSummary } from '../api'
import ProblemTable from '../components/ProblemTable.vue'
import ProgressRing from '../components/ProgressRing.vue'
import { locale, t } from '../i18n'
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

watch(locale, () => {
  if (activeTab.value !== 'collections') return
  collections.value = []
  ensureCollections()
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
  if (!window.confirm(t('home.resetConfirm'))) return
  try {
    await api.resetProgress()
  } catch (e) {
    window.alert(t('home.resetFailed', { error: e instanceof Error ? e.message : String(e) }))
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
    window.alert(t('home.exportFailed', { error: e instanceof Error ? e.message : String(e) }))
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
  if (!window.confirm(t('home.importConfirm'))) return
  try {
    const data = JSON.parse(await file.text()) as unknown
    const r = await api.importProgress(data)
    await store.fetchProblems()
    if (activeTab.value === 'collections') {
      collections.value = []
      ensureCollections()
    }
    window.alert(
      t('home.importDone', { imported: r.imported }) +
        (r.skipped ? t('home.importSkipped', { skipped: r.skipped }) : '')
    )
  } catch (e) {
    window.alert(
      t('home.importFailed', { error: e instanceof Error ? e.message : String(e) })
    )
  }
}

const filteredCollections = computed(() => {
  const kw = collectionKeyword.value.trim().toLowerCase()
  if (!kw) return collections.value
  return collections.value.filter(
    (c) => c.name.toLowerCase().includes(kw) || c.description.toLowerCase().includes(kw)
  )
})
</script>

<template>
  <div class="list-page">
    <div class="hero">
      <h1>{{ t('home.heroTitle') }}</h1>
      <p>
        {{
          t('home.heroSubtitle', {
            solved: store.solvedCount,
            total: store.problems.length,
          })
        }}
      </p>
      <div class="hero-actions">
        <button
          v-if="store.solvedCount > 0"
          class="link-btn danger"
          :title="t('home.resetTitle')"
          @click="resetProgress"
        >
          {{ t('home.reset') }}
        </button>
        <button class="link-btn" :title="t('home.exportTitle')" @click="exportProgress">
          {{ t('home.export') }}
        </button>
        <button class="link-btn" :title="t('home.importTitle')" @click="triggerImport">
          {{ t('home.import') }}
        </button>
        <input ref="importInput" type="file" accept="application/json,.json" class="hidden-file" @change="onImportFile" />
      </div>
    </div>

    <div class="tabs">
      <button class="tab" :class="{ active: activeTab === 'all' }" @click="setTab('all')">
        {{ t('home.tabAll') }}
      </button>
      <button
        class="tab"
        :class="{ active: activeTab === 'collections' }"
        @click="setTab('collections')"
      >
        {{ t('home.tabCollections') }}
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
        <input
          v-model="collectionKeyword"
          class="search"
          :placeholder="t('home.searchCollections')"
        />
      </div>
      <div v-if="collectionsLoading" class="empty">{{ t('common.loading') }}</div>
      <div v-else-if="collectionsError" class="empty error">
        {{ t('common.loadFailed', { error: collectionsError }) }}
      </div>
      <div v-else-if="!filteredCollections.length" class="empty">
        {{ t('home.noCollections') }}
      </div>

      <div class="table" v-else>
        <router-link
          v-for="c in filteredCollections"
          :key="c.slug"
          :to="`/collections/${c.slug}`"
          class="row collection-row"
        >
          <span
            class="collection-mark"
            :title="t('common.solvedOf', { solved: c.solved_count, total: c.problem_count })"
          >
            <ProgressRing :value="c.solved_count" :total="c.problem_count" />
          </span>
          <span class="collection-body">
            <span class="title">{{ c.name }}</span>
            <span class="collection-desc">{{ c.description }}</span>
          </span>
          <span class="collection-progress" :class="{ done: c.solved_count === c.problem_count }">
            {{ c.solved_count }}/{{ c.problem_count }}
          </span>
          <span class="go">{{ t('home.enterCollection') }}</span>
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
  margin: 0 0 12px;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  margin-bottom: 28px;
}
.link-btn {
  border: none;
  background: none;
  padding: 0;
  color: var(--text-dim);
  font-size: 12.5px;
  text-decoration: underline;
  text-underline-offset: 3px;
  cursor: pointer;
}
.link-btn:hover {
  color: var(--accent);
}
.link-btn.danger:hover {
  color: var(--red);
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
  grid-template-columns: 56px minmax(0, 1.4fr) 90px minmax(0, 1.2fr) 80px;
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
  grid-template-columns: 40px minmax(0, 1.6fr) 90px 100px;
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

@media (max-width: 900px) {
  .list-page {
    padding: 28px 18px;
  }
}
@media (max-width: 720px) {
  .list-page {
    padding: 22px 14px;
  }
  .hero h1 {
    font-size: 22px;
  }
  .hero-actions {
    margin-bottom: 22px;
    gap: 6px 14px;
  }
  .tabs,
  .toolbar {
    margin-bottom: 12px;
  }
  .row,
  .row.collection-row {
    grid-template-columns: 32px minmax(0, 1fr) auto;
    gap: 10px;
    padding: 11px 12px;
  }
  .row.collection-row > .go {
    display: none;
  }
  .collection-desc {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .collection-progress {
    font-size: 12px;
  }
  .empty {
    padding: 40px 0;
  }
}
</style>
