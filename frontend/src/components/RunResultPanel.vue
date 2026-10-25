<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { CaseResult, RunResult } from '../api'
import { t } from '../i18n'

const props = defineProps<{ result: RunResult | null; running: boolean; error: string | null }>()

const isRunMode = computed(() => props.result?.mode === 'run')

// ---- 参考解耗时基准（首次 AC 后后台测量，仅展示不设门槛）----
const benchmark = computed(() => props.result?.benchmark ?? null)

function benchLabel(c: CaseResult): string {
  const b = benchmark.value?.[c.name]
  if (b == null || !c.passed) return ''
  const r = c.elapsed_ms / b
  const ratio = r >= 10 ? r.toFixed(0) : r.toFixed(1)
  return t('result.bench', { ms: b.toFixed(1), ratio })
}

function isFast(c: CaseResult): boolean {
  const b = benchmark.value?.[c.name]
  return b != null && c.elapsed_ms <= b
}

const statusText = computed(() => {
  const r = props.result
  if (!r) return ''
  switch (r.status) {
    case 'accepted':
      return r.mode === 'run' ? t('result.acceptedRun') : t('result.acceptedSubmit')
    case 'wrong_answer':
      return r.mode === 'run' ? t('result.wrongRun') : t('result.wrongSubmit')
    case 'timeout':
      return t('result.timeout')
    case 'syntax':
      return t('result.syntax')
    case 'no_solve':
      return t('result.noSolve')
    case 'cancelled':
      return t('result.cancelled')
    case 'env_missing':
      return t('result.envMissing')
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
const passedRows = computed(() =>
  (props.result?.cases ?? [])
    .filter((c) => c.passed)
    .map((c) => ({ ...c, bench: benchLabel(c), fast: isFast(c) }))
)
const showPassed = ref(false)
const panelEl = ref<HTMLDivElement | null>(null)

watch(
  () => props.result,
  async (r) => {
    if (!r) return
    showPassed.value = r.cases.every((c) => c.passed)
    if (!showPassed.value) {
      await nextTick()
      panelEl.value?.querySelector('.case.fail')?.scrollIntoView({ block: 'nearest' })
    }
  }
)
</script>

<template>
  <div class="result-panel" ref="panelEl">
    <div v-if="running" class="hint">
      <span class="spinner"></span> {{ t('result.running') }}
    </div>
    <div v-else-if="error" class="hint fail-text">
      {{ t('result.requestFailed', { error }) }}
    </div>
    <div v-else-if="!result" class="hint">{{ t('result.idle') }}</div>

    <template v-else>
      <div class="summary">
        <span class="status" :class="statusClass">{{ statusText }}</span>
        <span class="metric">{{
          isRunMode
            ? t('result.sampleMetric', { passed: result.passed, total: result.total })
            : t('result.caseMetric', { passed: result.passed, total: result.total })
        }}</span>
        <span class="metric" v-if="result.elapsed_ms != null">{{
          t('result.totalTime', { ms: result.elapsed_ms.toFixed(0) })
        }}</span>
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

        <template v-if="passedRows.length">
          <button
            v-if="failedCases.length"
            class="passed-toggle"
            @click="showPassed = !showPassed"
          >
            {{ showPassed ? '▾' : '▸' }} ✓
            {{ t('result.passedToggle', { count: passedRows.length }) }}
          </button>
          <div v-if="showPassed" class="passed-list">
            <div v-for="c in passedRows" :key="'p-' + c.name" class="case">
              <span class="case-icon">✓</span>
              <div class="case-body">
                <div class="case-name">{{ c.name }}</div>
              </div>
              <div class="case-times">
                <span class="case-time">{{ c.elapsed_ms.toFixed(1) }}ms</span>
                <span
                  v-if="c.bench"
                  class="case-bench"
                  :class="{ fast: c.fast }"
                  :title="t('result.benchTitle')"
                  >{{ c.bench }}</span
                >
              </div>
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
.case-times {
  flex: none;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}
.case-bench {
  font-size: 11px;
  color: var(--text-dim);
  white-space: nowrap;
}
.case-bench.fast {
  color: var(--accent);
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

@media (max-width: 720px), (max-height: 700px) {
  .result-panel {
    padding: 10px 12px;
    max-height: 45vh;
  }
  .summary {
    flex-wrap: wrap;
    gap: 4px 12px;
  }
  .case {
    padding: 7px 10px;
  }
}
</style>
