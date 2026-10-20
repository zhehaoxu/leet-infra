<script setup lang="ts">
import { useProblemStore } from './stores/problems'

const store = useProblemStore()
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <router-link to="/" class="logo">
        <span class="logo-mark">◈</span> leet<span class="logo-thin">-infra</span>
      </router-link>
      <div class="topbar-right">
        <span class="progress-chip" v-if="store.problems.length">
          {{ store.solvedCount }} / {{ store.problems.length }} 已解决
        </span>
        <span class="env-chip">PyTorch · 本地 CPU</span>
      </div>
    </header>
    <main class="content">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.topbar {
  height: 52px;
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
}
.logo {
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.5px;
}
.logo-mark {
  color: var(--accent);
}
.logo-thin {
  font-weight: 300;
}
.topbar-right {
  display: flex;
  gap: 10px;
}
.progress-chip,
.env-chip {
  font-size: 12px;
  color: var(--text-dim);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px 12px;
  background: var(--bg-elevated);
}
.progress-chip {
  color: var(--accent);
}
.content {
  flex: 1;
  min-height: 0;
  overflow: auto;
}
</style>
