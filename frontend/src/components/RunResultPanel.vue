<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { RunResult } from '../api'

const props = defineProps<{ result: RunResult | null; running: boolean; error: string | null }>()

const isRunMode = computed(() => props.result?.mode === 'run')

const statusText = computed(() => {
  const r = props.result
  if (!r) return ''
  switch (r.status) {
    case 'accepted':
      return r.mode === 'run' ? '样例通过 · 可提交全量评测' : 'Accepted · 全部通过'
    case 'wrong_answer':
      return r.mode === 'run' ? '样例未通过' : 'Wrong Answer'
    case 'timeout':
      return 'Time Limit Exceeded'
    case 'syntax':
      return 'Syntax Error'
    case 'no_solve':
      return '未实现 solve()'
    case 'env_missing':
      return '评测环境缺失'
    default:
      return r.status
  }
})

const statusClass = computed(() => {
  const r = props.result
  if (!r) return ''
  return r.accepted ? 'ok' : 'fail'
})

// 失败用例置顶展开，通过用例默认折叠；全通过时正常展开
const failedCases = computed(() => props.result?.cases.filter((c) => !c.passed) ?? [])
const passedCases = computed(() => props.result?.cases.filter((c) => c.passed) ?? [])
const showPassed = ref(false)
const panelEl = ref<HTMLDivElement | null>(null)

watch(
  () => props.result,
  async (r) => {
    if (!r) return
    showPassed.value = failedCases.value.length === 0
    if (failedCases.value.length) {
      await nextTick()
      panelEl.value?.querySelector('.case.fail')?.scrollIntoView({ block: 'nearest' })
    }
  }
)
</script>

<template>
  <div class="result-panel" ref="panelEl">
    <div v-if="running" class="hint">
      <span class="spinner"></span> 正在本地评测环境运行…（子进程执行 PyTorch 代码）
    </div>
    <div v-else-if="error" class="hint fail-text">请求失败：{{ error }}</div>
    <div v-else-if="!result" class="hint">
      尚未运行。「运行」只跑样例用例（⌘/Ctrl+Enter），「提交」全量评测并记录进度（⌘/Ctrl+Shift+Enter）。
    </div>

    <template v-else>
      <div class="summary">
        <span class="status" :class="statusClass">{{ statusText }}</span>
        <span class="metric">{{ result.passed }}/{{ result.total }} {{ isRunMode ? '样例用例通过' : '用例通过' }}</span>
        <span class="metric" v-if="result.elapsed_ms != null"
          >总耗时 {{ result.elapsed_ms.toFixed(0) }} ms</span
        >
      </div>

      <div v-if="result.error" class="error-box">{{ result.error }}</div>

      <div class="cases" v-if="result.cases.length">
        <div v-for="c in failedCases" :key="'f-' + c.name" class="case fail">
          <span class="case-icon">✗</span>
          <div class="case-body">
            <div class="case-name">{{ c.name }}</div>
            <div class="case-msg" v-if="c.message">{{ c.message }}</div>
          </div>
          <span class="case-time">{{ c.elapsed_ms.toFixed(1) }}ms</span>
        </div>

        <template v-if="passedCases.length">
          <button
            v-if="failedCases.length"
            class="passed-toggle"
            @click="showPassed = !showPassed"
          >
            {{ showPassed ? '▾' : '▸' }} ✓ {{ passedCases.length }} 个用例通过
          </button>
          <div v-if="showPassed" class="passed-list">
            <div v-for="c in passedCases" :key="'p-' + c.name" class="case">
              <span class="case-icon">✓</span>
              <div class="case-body">
                <div class="case-name">{{ c.name }}</div>
              </div>
              <span class="case-time">{{ c.elapsed_ms.toFixed(1) }}ms</span>
            </div>
          </div>
        </template>
      </div>

      <div class="stdout" v-if="result.stdout">
        <div class="stdout-title">stdout</div>
        <pre>{{ result.stdout }}</pre>
      </div>
    </template>
  </div>
</template>

<style scoped>
.result-panel {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  background: var(--bg-panel);
  max-height: 320px;
  overflow-y: auto;
}
.hint {
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 8px;
}
.fail-text {
  color: var(--red);
}
.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid var(--accent);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.summary {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 10px;
}
.status {
  font-weight: 700;
}
.status.ok {
  color: var(--accent);
}
.status.fail {
  color: var(--red);
}
.metric {
  color: var(--text-dim);
  font-size: 13px;
}
.error-box {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--red);
  background: rgba(241, 106, 106, 0.08);
  border: 1px solid rgba(241, 106, 106, 0.3);
  border-radius: 8px;
  padding: 10px 12px;
  white-space: pre-wrap;
  margin-bottom: 10px;
  max-height: 180px;
  overflow-y: auto;
}
.cases {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 10px;
}
.case {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
}
.case.fail {
  border-color: rgba(241, 106, 106, 0.45);
}
.passed-toggle {
  align-self: flex-start;
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 12.5px;
  cursor: pointer;
  padding: 4px 2px;
}
.passed-toggle:hover {
  color: var(--accent);
}
.passed-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.case-icon {
  font-weight: 700;
  color: var(--accent);
}
.case.fail .case-icon {
  color: var(--red);
}
.case-body {
  flex: 1;
  min-width: 0;
}
.case-name {
  font-family: var(--font-mono);
  font-size: 12.5px;
}
.case-msg {
  color: var(--red);
  font-size: 12px;
  margin-top: 3px;
  word-break: break-all;
}
.case-time {
  color: var(--text-dim);
  font-size: 12px;
  flex: none;
}
.stdout pre {
  margin: 6px 0 0;
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  max-height: 140px;
  overflow: auto;
  white-space: pre-wrap;
}
.stdout-title {
  color: var(--text-dim);
  font-size: 12px;
}
</style>
