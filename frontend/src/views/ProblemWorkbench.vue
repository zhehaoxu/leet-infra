<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
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
const activeTab = ref<'desc' | 'solution'>('desc')
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
  if (!store.problems.length) void store.fetchProblems() // 导航需要全量题序
  problem.value = null
  result.value = null
  runError.value = null
  loadError.value = null
  solutionCode.value = null
  activeTab.value = 'desc'
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
  draftRestored.value = false // 已开始执行，草稿恢复提示完成使命
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

// ---- 上一题/下一题导航 ----
const problemIndex = computed(() => store.problems.findIndex((p) => p.slug === props.slug))
const prevProblem = computed(() =>
  problemIndex.value > 0 ? store.problems[problemIndex.value - 1] : null
)
const nextProblem = computed(() => {
  const i = problemIndex.value
  return i >= 0 && i < store.problems.length - 1 ? store.problems[i + 1] : null
})
// 从当前题往后循环找第一道未解题（仅 AC 后展示入口）
const nextUnsolved = computed(() => {
  const list = store.problems
  if (!list.length) return null
  for (let step = 1; step <= list.length; step++) {
    const p = list[(problemIndex.value + step) % list.length]
    if (!p.solved) return p
  }
  return null
})

// ---- 参考答案：复制 ----
const copied = ref(false)
let copiedTimer: ReturnType<typeof setTimeout> | null = null

async function copySolution() {
  if (!solutionCode.value) return
  try {
    await navigator.clipboard.writeText(solutionCode.value)
  } catch {
    // 剪贴板 API 不可用时的退化路径
    const ta = document.createElement('textarea')
    ta.value = solutionCode.value
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    ta.remove()
  }
  copied.value = true
  if (copiedTimer) clearTimeout(copiedTimer)
  copiedTimer = setTimeout(() => (copied.value = false), 1500)
}

function resetCode() {
  if (problem.value) {
    localStorage.removeItem(draftKey(problem.value.slug))
    draftRestored.value = false
    code.value = problem.value.starter_code
    result.value = null
  }
}

async function openSolution() {
  if (!accepted.value) return // 未解锁时标签禁用
  activeTab.value = 'solution'
  if (!solutionCode.value) {
    try {
      const r = await api.getSolution(props.slug)
      solutionCode.value = r.solution_code
    } catch (e) {
      runError.value = e instanceof Error ? e.message : String(e)
      activeTab.value = 'desc'
    }
  }
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
        <div class="pager" v-if="problemIndex >= 0">
          <router-link
            v-if="prevProblem"
            :to="`/problems/${prevProblem.slug}`"
            class="pager-link"
            :title="prevProblem.title"
          >
            ← 上一题
          </router-link>
          <span class="pager-pos">{{ problemIndex + 1 }}/{{ store.problems.length }}</span>
          <router-link
            v-if="nextProblem"
            :to="`/problems/${nextProblem.slug}`"
            class="pager-link"
            :title="nextProblem.title"
          >
            下一题 →
          </router-link>
        </div>
        <router-link
          v-if="accepted && nextUnsolved"
          :to="`/problems/${nextUnsolved.slug}`"
          class="pager-link next-unsolved"
          :title="`去做：${nextUnsolved.title}`"
        >
          下一道未解 →
        </router-link>
        <div class="actions">
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
          <div class="tabbar">
            <button class="tab" :class="{ active: activeTab === 'desc' }" @click="activeTab = 'desc'">
              题目说明
            </button>
            <button
              class="tab"
              :class="{ active: activeTab === 'solution' }"
              :disabled="!accepted"
              :title="accepted ? '查看参考答案' : '提交通过后解锁参考答案'"
              @click="openSolution"
            >
              参考答案
            </button>
          </div>
          <div class="tab-content" :class="{ 'solution-mode': activeTab === 'solution' }">
            <MarkdownView v-if="activeTab === 'desc'" :key="problem.slug + '-desc'" :source="problem.description" />
            <div v-else class="solution-view">
              <div class="solution-toolbar">
                <button class="copy-btn" @click="copySolution">
                  <span class="btn-icon">⧉</span>{{ copied ? '已复制' : '复制代码' }}
                </button>
              </div>
              <div class="solution-editor">
                <CodeEditor :model-value="solutionCode ?? ''" read-only />
              </div>
            </div>
          </div>
        </section>

        <div class="divider" @mousedown="onDragStart"></div>

        <section class="pane code-pane" :style="{ width: `${(1 - splitRatio) * 100 - 0.5}%` }">
          <div class="editor-wrap">
            <CodeEditor v-model="code" @run="runCode" @submit="submitCode" />
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
.pager {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-left: 6px;
}
.pager-link {
  color: var(--text-dim);
  font-size: 12.5px;
  white-space: nowrap;
}
.pager-link:hover {
  color: var(--accent);
}
.pager-pos {
  color: var(--text-dim);
  font-size: 12px;
  font-family: var(--font-mono);
}
.pager-link.next-unsolved {
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 2px 12px;
}
.pager-link.next-unsolved:hover {
  background: var(--accent-dim);
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
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
  border-right: 1px solid var(--border);
}
.tabbar {
  flex: none;
  display: flex;
  gap: 4px;
  padding: 0 12px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
}
.tab {
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 13.5px;
  padding: 12px 14px;
  cursor: pointer;
}
.tab:hover:not(:disabled) {
  color: var(--text);
}
.tab.active {
  color: var(--accent);
}
.tab:disabled {
  color: var(--text-dim);
  opacity: 0.45;
  cursor: not-allowed;
}
.tab-content {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 18px 20px;
}
.tab-content.solution-mode {
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.solution-view {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.solution-toolbar {
  flex: none;
  display: flex;
  justify-content: flex-end;
}
.copy-btn {
  border-radius: 8px;
  padding: 5px 12px;
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text);
  font-size: 12.5px;
}
.copy-btn:hover {
  border-color: var(--text-dim);
}
.solution-editor {
  flex: 1;
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
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
</style>
