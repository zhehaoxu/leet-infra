<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import { t } from '../i18n'

const emit = defineEmits<{ (e: 'close'): void }>()

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="overlay" @click.self="emit('close')">
    <div class="modal" role="dialog" :aria-label="t('help.title')">
      <button class="close" :title="t('help.close')" @click="emit('close')">✕</button>
      <h2>{{ t('help.title') }}</h2>

      <h3>{{ t('help.shortcuts') }}</h3>
      <ul>
        <li><kbd>⌘/Ctrl + Enter</kbd> {{ t('help.shortcutRun') }}</li>
        <li><kbd>⌘/Ctrl + Shift + Enter</kbd> {{ t('help.shortcutSubmit') }}</li>
        <li><kbd>Esc</kbd> {{ t('help.shortcutClose') }}</li>
      </ul>

      <h3>{{ t('help.runVsSubmit') }}</h3>
      <ul>
        <li>{{ t('help.runRule') }}</li>
        <li>{{ t('help.submitRule') }}</li>
      </ul>

      <h3>{{ t('help.env') }}</h3>
      <ul>
        <li>{{ t('help.envLocal') }}</li>
        <li>{{ t('help.envPackages') }}</li>
        <li>{{ t('help.envSeed') }}</li>
        <li>{{ t('help.envTensors') }}</li>
      </ul>

      <h3>{{ t('help.extending') }}</h3>
      <ul>
        <li>{{ t('help.extendProblems') }}</li>
        <li>{{ t('help.extendCollections') }}</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(5, 8, 12, 0.62);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}
.modal {
  position: relative;
  width: min(560px, 92vw);
  max-height: 82vh;
  overflow-y: auto;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 26px 30px;
}
.close {
  position: absolute;
  top: 14px;
  right: 14px;
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 14px;
  cursor: pointer;
}
.close:hover {
  color: var(--text);
}
h2 {
  margin: 0 0 14px;
  font-size: 19px;
}
h3 {
  margin: 18px 0 8px;
  font-size: 14px;
  color: var(--accent);
}
ul {
  margin: 0;
  padding-left: 18px;
  color: var(--text-dim);
  font-size: 13.5px;
  line-height: 1.9;
}
kbd,
code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1px 6px;
}

@media (max-width: 640px) {
  .overlay {
    padding: 12px 0;
    align-items: flex-start;
  }
  .modal {
    width: 94vw;
    max-height: 88vh;
    border-radius: 12px;
    padding: 22px 18px;
  }
  ul {
    font-size: 13px;
    line-height: 1.75;
  }
}
</style>
