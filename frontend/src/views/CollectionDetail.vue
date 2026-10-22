<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type CollectionDetailInfo } from '../api'
import ProblemTable from '../components/ProblemTable.vue'
import ProgressRing from '../components/ProgressRing.vue'

const props = defineProps<{ slug: string }>()

const collection = ref<CollectionDetailInfo | null>(null)
const loadError = ref<string | null>(null)
const loading = ref(true)

// ---- 实现方式偏好（仅含多 variant 题目时展示；存 localStorage 供做题页继承）----
const VARIANT_PREFIX = 'leet-infra:variant:'
const variantPref = ref('pytorch')

const hasMultiVariant = computed(() =>
  (collection.value?.problems ?? []).some((p) => (p.variants?.length ?? 0) > 1)
)
const variantKeys = computed(() => {
  const seen: string[] = []
  for (const p of collection.value?.problems ?? []) {
    for (const v of p.variants ?? []) {
      if (!seen.includes(v.key)) seen.push(v.key)
    }
  }
  return seen
})
const variantOptions = computed(() => {
  const labels: Record<string, string> = {}
  for (const p of collection.value?.problems ?? []) {
    for (const v of p.variants ?? []) labels[v.key] = v.label
  }
  return variantKeys.value.map((k) => ({ key: k, label: labels[k] ?? k }))
})

function selectVariant(key: string) {
  variantPref.value = key
  localStorage.setItem(VARIANT_PREFIX + props.slug, key)
}

const solvedCount = computed(
  () => (collection.value?.problems ?? []).filter((p) => p.solved).length
)

async function load() {
  loading.value = true
  loadError.value = null
  collection.value = null
  variantPref.value = localStorage.getItem(VARIANT_PREFIX + props.slug) ?? 'pytorch'
  try {
    collection.value = await api.getCollection(props.slug)
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="detail-page">
    <router-link to="/" class="back">← 返回题库</router-link>

    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="loadError" class="empty error">{{ loadError }}</div>

    <template v-else-if="collection">
      <div class="hero">
        <h1>{{ collection.name }}</h1>
        <p>{{ collection.description }}</p>
        <p class="progress">
          <ProgressRing :value="solvedCount" :total="collection.problems.length" :size="16" />
          {{ solvedCount }}/{{ collection.problems.length }} 已解决
        </p>
      </div>

      <div v-if="hasMultiVariant" class="variant-bar">
        <span class="variant-label">实现方式</span>
        <div class="variant-select">
          <select
            :value="variantPref"
            @change="selectVariant(($event.target as HTMLSelectElement).value)"
          >
            <option v-for="v in variantOptions" :key="v.key" :value="v.key">
              {{ v.label }}
            </option>
          </select>
          <span class="variant-caret" aria-hidden="true">▾</span>
        </div>
        <span class="variant-hint">题目入口将沿用该选择；未实现的实现做题时自动回退</span>
      </div>

      <ProblemTable
        :problems="collection.problems"
        :persist-key="`collection-${slug}`"
        :variant="variantPref"
        empty-text="该合集暂无题目"
      />
    </template>
  </div>
</template>

<style scoped>
.detail-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 32px 24px;
}
.back {
  color: var(--text-dim);
  font-size: 13px;
}
.back:hover {
  color: var(--accent);
}
.hero {
  margin-top: 18px;
  margin-bottom: 24px;
}
.hero h1 {
  font-size: 24px;
  margin: 0 0 8px;
}
.hero p {
  color: var(--text-dim);
  margin: 0;
}
.hero .progress {
  margin-top: 6px;
  color: var(--accent);
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.variant-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.variant-label {
  font-size: 13px;
  color: var(--text-dim);
}
.variant-select {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.variant-select select {
  appearance: none;
  -webkit-appearance: none;
  font: inherit;
  font-size: 12.5px;
  color: var(--accent);
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 26px 4px 14px;
  cursor: pointer;
}
.variant-select select:hover,
.variant-select select:focus {
  outline: none;
  border-color: var(--accent);
}
.variant-caret {
  position: absolute;
  right: 12px;
  pointer-events: none;
  color: var(--accent);
  font-size: 10px;
}
.variant-hint {
  font-size: 11.5px;
  color: var(--text-dim);
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
