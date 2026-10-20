<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api, type ProblemDetail, type RunResult } from '../api'
import { useProblemStore } from '../stores/problems'
import CodeEditor from '../components/CodeEditor.vue'
import MarkdownView from '../components/MarkdownView.vue'
import RunResultPanel from '../components/RunResultPanel.vue'
import { burstConfetti } from '../utils/confetti'

const props = defineProps<{ slug: string }>()
const store = useProblemStore()

const problem = ref<ProblemDetail | null>(null)
const code = ref('')
const result = ref<RunResult | null>(null)
const running = ref(false)
const runningKind = ref<'run' | 'submit'>('run')
const runError = ref<string | null>(null)
const loadError = ref<string | null>(null)

const solutionCode = ref<string | null>(null)
const showSolution = ref(false)
const accepted = ref(false)

// ---- 草稿持久化（localStorage，每题一份）----
const DRAFT_PREFIX = 'leet-infra:draft:'
const draftRestored = ref(false)
let draftTimer: ReturnType<typeof setTimeout> | null = null

function draftKey(slug: string) {
  return DRAFT_PREFIX + slug
}

function flushDraft() {
  if (draftTimer) {
    clearTimeout(draftTimer)
    draftTimer = null
  }
  if (problem.value && code.value !== problem.value.starter_code) {
    localStorage.setItem(draftKey(problem.value.slug), code.value)
  }
}

watch(code, () => {
  // 输入防抖，避免每次击键同步写 localStorage
  if (draftTimer) clearTimeout(draftTimer)
  draftTimer = setTimeout(flushDraft, 400)
})

// 左右分栏拖拽
const splitRatio = ref(0.45)
const bodyEl = ref<HTMLDivElement | null>(null)

function onDragStart(e: MouseEvent) {
  e.preventDefault()
  const move = (ev: MouseEvent) => {
    const rect = bodyEl.value?.getBoundingClientRect()
    if (!rect) return
    const ratio = Math.min(0.75, Math.max(0.25, (ev.clientX - rect.left) / rect.width))
    splitRatio.value = ratio
  }
  const up = () => {
    window.removeEventListener('mousemove', move)
    window.removeEventListener('mouseup', up)
  }
  window.addEventListener('mousemove', move)
  window.addEventListener('mouseup', up)
}

async function load() {
  flushDraft() // 先把上一页的编辑存回对应草稿再切走
  problem.value = null
  result.value = null
  runError.value = null
  loadError.value = null
  solutionCode.value = null
  showSolution.value = false
  accepted.value = false
  draftRestored.value = false
  try {
    const p = await api.getProblem(props.slug)
    problem.value = p
    const draft = localStorage.getItem(draftKey(p.slug))
    draftRestored.value = draft != null && draft !== p.starter_code
    code.value = draft ?? p.starter_code
    accepted.value = p.solved
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e)
  }
}

async function execute(kind: 'run' | 'submit') {
  if (running.value) return
  running.value = true
  runningKind.value = kind
  runError.value = null
  try {
    const r =
      kind === 'run'
        ? await api.run(props.slug, code.value)
        : await api.submit(props.slug, code.value)
    result.value = r
    // 只有提交才记录进度并喷彩花
    if (kind === 'submit' && r.accepted) {
      if (!accepted.value) store.markSolved(props.slug)
      accepted.value = true
      burstConfetti()
    }
  } catch (e) {
    runError.value = e instanceof Error ? e.message : String(e)
    result.value = null
  } finally {
    running.value = false
  }
}

const runCode = () => execute('run')
const submitCode = () => execute('submit')

function resetCode() {
  if (problem.value) {
    localStorage.removeItem(draftKey(problem.value.slug))
    draftRestored.value = false
    code.value = problem.value.starter_code
    result.value = null
  }
}

async function toggleSolution() {
  if (showSolution.value) {
    showSolution.value = false
    return
  }
  if (!solutionCode.value) {
    try {
      const r = await api.getSolution(props.slug)
      solutionCode.value = r.solution_code
    } catch (e) {
      runError.value = e instanceof Error ? e.message : String(e)
      return
    }
  }
  showSolution.value = true
}

watch(() => props.slug, load) // 题目间直接跳转时重新加载
onMounted(load)
onBeforeUnmount(flushDraft)
</script>

<template>
  <div class="workbench">
    <div v-if="loadError" class="empty">{{ loadError }}</div>
    <div v-else-if="!problem" class="empty">加载题目中…</div>
    <template v-else>
      <div class="subheader">
        <router-link to="/" class="back">← 题库</router-link>
        <span class="title">{{ problem.title }}</span>
        <em class="badge" :class="problem.difficulty">{{ problem.difficulty }}</em>
        <span v-if="accepted" class="ac-tag">✓ 已解决</span>
        <span v-if="draftRestored" class="draft-hint" title="已从 localStorage 恢复上次编辑的草稿">草稿已恢复</span>
        <div class="actions">
          <button v-if="accepted" class="ghost" @click="toggleSolution">
            {{ showSolution ? '收起参考答案' : '参考答案' }}
          </button>
          <button class="ghost" :disabled="running" @click="resetCode"><span class="btn-icon">↺</span>重置</button>
          <button class="ghost" :disabled="running" @click="runCode"><span class="btn-icon play">▶</span>运行</button>
          <button class="run" :disabled="running" @click="submitCode">
            <span v-if="running" class="loading">{{ runningKind === 'submit' ? '提交中…' : '运行中…' }}</span>
            <template v-else><span class="btn-icon">✓</span>提交</template>
          </button>
        </div>
      </div>

      <div class="split-body" ref="bodyEl">
        <section class="pane desc" :style="{ width: `${splitRatio * 100}%` }">
          <MarkdownView v-if="!showSolution" :key="problem.slug + '-desc'" :source="problem.description" />
          <div v-else class="solution-view">
            <h2>参考答案（AC 后解锁）</h2>
            <pre class="solution-code">{{ solutionCode }}</pre>
          </div>
        </section>

        <div class="divider" @mousedown="onDragStart"></div>

        <section class="pane code-pane" :style="{ width: `${(1 - splitRatio) * 100 - 0.5}%` }">
          <div class="editor-wrap">
            <CodeEditor v-model="code" @run="runCode" />
          </div>
          <RunResultPanel :result="result" :running="running" :error="runError" />
        </section>
      </div>
    </template>
  </div>
</template>

<style scoped>
.workbench {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.empty {
  color: var(--text-dim);
  text-align: center;
  padding: 80px;
}
.subheader {
  flex: none;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
}
.back {
  color: var(--text-dim);
  font-size: 13px;
}
.back:hover {
  color: var(--accent);
}
.title {
  font-size: 16px;
  font-weight: 700;
}
.ac-tag {
  color: var(--accent);
  font-size: 12px;
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 1px 10px;
}
.draft-hint {
  color: var(--blue);
  font-size: 12px;
  border: 1px dashed rgba(106, 169, 241, 0.5);
  border-radius: 999px;
  padding: 1px 10px;
}
.actions {
  margin-left: auto;
  display: flex;
  gap: 8px;
  align-items: center;
}
button {
  border-radius: 8px;
  padding: 7px 16px;
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text);
}
button.ghost:hover:not(:disabled) {
  border-color: var(--text-dim);
}
button.ghost:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
button.run {
  background: var(--accent);
  border-color: var(--accent);
  color: #06231a;
  font-weight: 700;
}
button.run:disabled {
  opacity: 0.6;
  cursor: wait;
}
.btn-icon {
  font-size: 15px;
  line-height: 1;
  margin-right: 4px;
}
.btn-icon.play {
  font-size: 11px;
}
.loading {
  font-weight: 700;
}
.split-body {
  flex: 1;
  min-height: 0;
  display: flex;
}
.pane {
  min-width: 0;
  overflow-y: auto;
  padding: 18px 20px;
}
.desc {
  border-right: 1px solid var(--border);
}
.divider {
  flex: none;
  width: 5px;
  cursor: col-resize;
  background: transparent;
}
.divider:hover {
  background: var(--accent);
  opacity: 0.35;
}
.code-pane {
  display: flex;
  flex-direction: column;
  padding: 0;
}
.editor-wrap {
  flex: 1;
  min-height: 260px;
}
.solution-view h2 {
  margin-top: 0;
}
.solution-code {
  font-family: var(--font-mono);
  font-size: 12.5px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  white-space: pre-wrap;
}
</style>
