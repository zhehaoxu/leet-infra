<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type CollectionDetailInfo } from '../api'
import ProblemTable from '../components/ProblemTable.vue'
import ProgressRing from '../components/ProgressRing.vue'

const props = defineProps<{ slug: string }>()

const collection = ref<CollectionDetailInfo | null>(null)
const loadError = ref<string | null>(null)
const loading = ref(true)

const solvedCount = computed(
  () => (collection.value?.problems ?? []).filter((p) => p.solved).length
)

async function load() {
  loading.value = true
  loadError.value = null
  collection.value = null
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

      <ProblemTable
        :problems="collection.problems"
        :persist-key="`collection-${slug}`"
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
.empty {
  text-align: center;
  color: var(--text-dim);
  padding: 60px 0;
}
.empty.error {
  color: var(--red);
}
</style>
