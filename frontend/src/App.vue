<script setup lang="ts">
import { ref } from 'vue'
import { useProblemStore } from './stores/problems'
import HelpModal from './components/HelpModal.vue'

const store = useProblemStore()
const showHelp = ref(false)
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <router-link to="/" class="logo">
        <svg class="logo-mark" width="26" height="26" viewBox="6 12 52 42" aria-hidden="true">
          <g stroke-linejoin="round" stroke-width="3">
            <path d="M32 34 L54 42 L32 50 L10 42 Z" fill="#2a3542" stroke="#2a3542"/>
            <path d="M32 25 L54 33 L32 41 L10 33 Z" fill="#3d4a5c" stroke="#3d4a5c"/>
            <path d="M32 16 L54 24 L32 32 L10 24 Z" fill="#2dd4a7" stroke="#2dd4a7"/>
          </g>
        </svg>
        <span>leet-infra</span>
      </router-link>
      <div class="topbar-right">
        <span class="progress-chip" v-if="store.problems.length">
          {{ store.solvedCount }} / {{ store.problems.length }} 已解决
        </span>
        <span class="env-chip">PyTorch · 本地 CPU</span>
        <button class="help-btn" title="帮助（快捷键与评测规则）" @click="showHelp = true">?</button>
      </div>
    </header>
    <main class="content">
      <router-view />
    </main>
    <HelpModal v-if="showHelp" @close="showHelp = false" />
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
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.5px;
}
.logo-mark {
  display: block;
}
.topbar-right {
  display: flex;
  gap: 10px;
  align-items: center;
}
.help-btn {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text-dim);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}
.help-btn:hover {
  color: var(--accent);
  border-color: var(--accent);
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
