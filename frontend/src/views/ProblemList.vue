<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useProblemStore } from '../stores/problems'

const store = useProblemStore()
const keyword = ref('')
const difficulty = ref<'' | 'Easy' | 'Medium' | 'Hard'>('')
const status = ref<'' | 'solved' | 'unsolved'>('')

onMounted(() => {
  if (!store.problems.length) store.fetchProblems()
})

const filtered = computed(() =>
  store.problems.filter((p) => {
    if (keyword.value && !p.title.toLowerCase().includes(keyword.value.toLowerCase())) return false
    if (difficulty.value && p.difficulty !== difficulty.value) return false
    if (status.value === 'solved' && !p.solved) return false
    if (status.value === 'unsolved' && p.solved) return false
    return true
  })
)
</script>

<template>
  <div class="list-page">
    <div class="hero">
      <h1>AI Infra 从动手写开始</h1>
      <p>PyTorch 题库 · 代码在你本机 Python 虚拟环境中真实运行 · {{ store.solvedCount }}/{{ store.problems.length }} 已解决</p>
    </div>

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

    <div v-if="store.loading" class="empty">加载中…</div>
    <div v-else-if="store.error" class="empty error">加载失败：{{ store.error }}</div>
    <div v-else-if="!filtered.length" class="empty">没有匹配的题目</div>

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
        <span class="tags">{{ p.tags.join(' · ') }}</span>
        <span class="go">解答 →</span>
      </router-link>
    </div>
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
router-link.row:hover,
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
