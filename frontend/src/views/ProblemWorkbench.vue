<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type ProblemDetail, type RunResult } from '../api'
import { useProblemStore } from '../stores/problems'
import CodeEditor from '../components/CodeEditor.vue'
import MarkdownView from '../components/MarkdownView.vue'
import RunResultPanel from '../components/RunResultPanel.vue'
import { locale, t, type MessageKey } from '../i18n'
import { burstConfetti } from '../utils/confetti'
import { draftKey } from '../utils/draft'

const props = defineProps<{ slug: string }>()
const store = useProblemStore()
const route = useRoute()
const router = useRouter()

const problem = ref<ProblemDetail | null>(null)
const code = ref('')
const result = ref<RunResult | null>(null)
const running = ref(false)
const runningKind = ref<'run' | 'submit'>('run')
const runError = ref<string | null>(null)
const loadError = ref<string | null>(null)

const solutionCode = ref<string | null>(null)
const activeTab = ref<'desc' | 'solution' | 'notes'>('desc')
const accepted = ref(false)

// 参考答案以 Markdown 代码块渲染，复用题面同款 MarkdownView
const solutionMarkdown = computed(() =>
  solutionCode.value ? '```python\n' + solutionCode.value.trim() + '\n```' : ''
)

// ---- 实现变体（pytorch/triton/cuda）----
// 用户期望的 variant：URL query > 合集偏好 > 题目默认；后端对不可用的回退默认
function difficultyLabel(d: 'Easy' | 'Medium' | 'Hard') {
  return t(`difficulty.${d.toLowerCase()}` as MessageKey)
}

const preferredVariant = ref('')
const variantList = computed(() => problem.value?.variants ?? [])
// 仅算子合集（operators）题目展示实现方式选择器，其他合集一律隐藏
function isOperatorsProblem(p: ProblemDetail): boolean {
  return (p.collections ?? []).some((c) => c.slug === 'operators')
}
const showVariantToggle = computed(
  () => variantList.value.length > 1 && problem.value !== null && isOperatorsProblem(problem.value)
)
const activeVariant = computed(() => problem.value?.active_variant ?? 'pytorch')
const variantFallbackNotice = computed(
  () =>
    preferredVariant.value !== '' &&
    problem.value !== null &&
    preferredVariant.value !== activeVariant.value
)

function collectionPreferenceKey(p: ProblemDetail): string | null {
  if (!isOperatorsProblem(p)) return null
  for (const c of p.collections ?? []) {
    const v = localStorage.getItem(`leet-infra:variant:${c.slug}`)
    if (v) return v
  }
  return null
}

function resolvePreferred(p: ProblemDetail): string {
  const fallback = p.variants?.find((v) => v.available)?.key ?? p.variants?.[0]?.key ?? 'pytorch'
  if (!isOperatorsProblem(p)) return fallback // 非算子合集固定默认实现，不吃 URL/合集偏好
  const q = typeof route.query.variant === 'string' ? route.query.variant : ''
  if (q && p.variants?.some((v) => v.key === q)) return q
  const col = collectionPreferenceKey(p)
  if (col && p.variants?.some((v) => v.key === col)) return col
  return fallback
}

function selectVariant(key: string) {
  preferredVariant.value = key
  router.replace({ query: { ...route.query, variant: key } })
}

// ---- 草稿持久化（localStorage，每题一份；key 规则见 utils/draft.ts）----
const draftRestored = ref(false)
let draftTimer: ReturnType<typeof setTimeout> | null = null

function flushDraft() {
  if (draftTimer) {
    clearTimeout(draftTimer)
    draftTimer = null
  }
  if (problem.value && code.value !== problem.value.starter_code) {
    localStorage.setItem(draftKey(problem.value.slug, activeVariant.value), code.value)
  }
}

watch(code, () => {
  // 输入防抖，避免每次击键同步写 localStorage
  if (draftTimer) clearTimeout(draftTimer)
  draftTimer = setTimeout(flushDraft, 400)
})

// ---- 题目笔记：来自后端 notes.md，缺失时该标签置灰 ----
const notes = ref('')
const hasNotes = computed(() => notes.value.trim().length > 0)

// 左右分栏拖拽（仅桌面双栏布局可见分隔条）
const splitRatio = ref(0.45)
const splitStyle = computed(() => ({ '--split': String(splitRatio.value) }))
const bodyEl = ref<HTMLDivElement | null>(null)
let detachDrag: (() => void) | null = null

function onDragStart(e: MouseEvent) {
  e.preventDefault()
  const move = (ev: MouseEvent) => {
    const rect = bodyEl.value?.getBoundingClientRect()
    if (!rect) return
    splitRatio.value = Math.min(0.75, Math.max(0.25, (ev.clientX - rect.left) / rect.width))
  }
  const up = () => {
    window.removeEventListener('mousemove', move)
    window.removeEventListener('mouseup', up)
    detachDrag = null
  }
  window.addEventListener('mousemove', move)
  window.addEventListener('mouseup', up)
  detachDrag = up
}

// 语言切换只重拉文案（题面/笔记/标题），保留编辑器内容、评测结果与当前标签页
async function reloadForLocale() {
  if (!problem.value) return
  try {
    const p = await api.getProblem(props.slug, activeVariant.value)
    problem.value = p
    notes.value = p.notes ?? ''
  } catch (e) {
    runError.value = e instanceof Error ? e.message : String(e)
  }
}

async function load() {
  flushDraft() // 先把上一页的编辑存回对应草稿再切走
  if (!store.problems.length) void store.fetchProblems() // 导航需要全量题序
  const q = typeof route.query.variant === 'string' ? route.query.variant : ''
  preferredVariant.value = q // 首次进入先取 URL；无/非算子合集则由 resolvePreferred 定默认
  problem.value = null
  result.value = null
  runError.value = null
  loadError.value = null
  solutionCode.value = null
  activeTab.value = 'desc'
  accepted.value = false
  draftRestored.value = false
  notes.value = ''
  try {
    let p = await api.getProblem(props.slug, preferredVariant.value || undefined)
    // URL 未携带 variant 时按合集偏好/题目默认解析期望实现
    preferredVariant.value = resolvePreferred(p)
    if (preferredVariant.value !== p.active_variant) {
      // 期望的是可用但非默认的实现（如未来的 triton）：重拉对应三件套
      p = await api.getProblem(props.slug, preferredVariant.value)
    }
    problem.value = p
    const draft = localStorage.getItem(draftKey(p.slug, p.active_variant))
    draftRestored.value = draft != null && draft !== p.starter_code
    code.value = draft ?? p.starter_code
    notes.value = p.notes ?? ''
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
        ? await api.run(props.slug, code.value, activeVariant.value)
        : await api.submit(props.slug, code.value, activeVariant.value)
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

async function cancelRun() {
  // 后端 kill 评测子进程；当前 fetch 会以 cancelled 结果自然结束
  try {
    await api.cancelJudge()
  } catch {
    /* 忽略：进程已结束时无可取消 */
  }
}

function resetCode() {
  if (problem.value) {
    localStorage.removeItem(draftKey(problem.value.slug, activeVariant.value))
    draftRestored.value = false
    code.value = problem.value.starter_code
    result.value = null
  }
}

async function redoProblem() {
  // 单题重刷：只清本题 solved + 本题草稿，笔记保留
  if (!problem.value || !accepted.value || running.value) return
  const slug = problem.value.slug
  if (!confirm(t('workbench.redoConfirm'))) return
  try {
    await api.unmarkProblem(slug)
  } catch (e) {
    runError.value = e instanceof Error ? e.message : String(e)
    return
  }
  store.unmarkSolved(slug)
  accepted.value = false
  solutionCode.value = null
  activeTab.value = 'desc'
  result.value = null
  runError.value = null
  localStorage.removeItem(draftKey(slug, activeVariant.value))
  draftRestored.value = false
  code.value = problem.value.starter_code // 先回到模板，防 load 前防抖把旧代码写回草稿
}

async function openSolution() {
  if (!accepted.value) return // 未解锁时标签禁用
  activeTab.value = 'solution'
  if (!solutionCode.value) {
    try {
      const r = await api.getSolution(props.slug, activeVariant.value)
      solutionCode.value = r.solution_code
    } catch (e) {
      runError.value = e instanceof Error ? e.message : String(e)
      activeTab.value = 'desc'
    }
  }
}

// 题目切换或 URL variant 变化时重新加载（选择器切换经由 router.replace 统一走这里）
watch([() => props.slug, () => route.query.variant], ([slug, q], [oldSlug, oldQ]) => {
  if (slug === oldSlug && typeof q === 'string' && problem.value && q !== oldQ) {
    // 同题内切换实现：先按当前实现的 key 存回草稿，再重新加载
    flushDraft()
  }
  if (slug !== oldSlug) {
    void load()
    return
  }
  // 同题但 URL 请求了本题不存在的 variant（如后端回退后 URL 仍停在 triton）：
  // 把 URL 归位到实际实现，避免加载-回退死循环
  if (typeof q === 'string' && problem.value?.slug === slug && q !== problem.value.active_variant) {
    if (!problem.value.variants?.some((v) => v.key === q && v.available)) {
      void router.replace({ query: { ...route.query, variant: problem.value.active_variant } })
      return
    }
  }
  void load()
})
watch(locale, reloadForLocale)
onMounted(load)
onBeforeUnmount(() => {
  detachDrag?.()
  flushDraft()
})
</script>

<template>
  <div class="workbench">
    <div v-if="loadError" class="empty">{{ loadError }}</div>
    <div v-else-if="!problem" class="empty">{{ t('workbench.loadingProblem') }}</div>
    <template v-else>
      <div class="subheader">
        <router-link to="/" class="back">{{ t('workbench.back') }}</router-link>
        <span class="title" :title="problem.title">{{ problem.title }}</span>
        <em class="badge" :class="problem.difficulty">{{
          difficultyLabel(problem.difficulty)
        }}</em>
        <span v-if="accepted" class="ac-tag">{{ t('workbench.solved') }}</span>
        <span
          v-if="draftRestored"
          class="draft-hint"
          :title="t('workbench.draftRestoredTitle')"
          >{{ t('workbench.draftRestored') }}</span
        >
        <div v-if="showVariantToggle" class="variant-select" :title="t('workbench.implementation')">
          <select
            :value="activeVariant"
            @change="selectVariant(($event.target as HTMLSelectElement).value)"
          >
            <option
              v-for="v in variantList"
              :key="v.key"
              :value="v.key"
              :disabled="!v.available"
            >
              {{ v.label }}{{ v.available ? '' : t('workbench.comingSoon') }}
            </option>
          </select>
          <span class="variant-caret" aria-hidden="true">▾</span>
        </div>
        <span v-if="variantFallbackNotice" class="variant-notice">
          {{
            t('workbench.variantFallback', {
              variant: variantList.find((v) => v.key === preferredVariant)?.label ?? '',
              active: activeVariant,
            })
          }}
        </span>
        <div class="pager" v-if="problemIndex >= 0">
          <router-link
            v-if="prevProblem"
            :to="{ path: `/problems/${prevProblem.slug}`, query: { variant: activeVariant } }"
            class="pager-link"
            :title="prevProblem.title"
          >
            {{ t('workbench.prev') }}
          </router-link>
          <span class="pager-pos">{{ problemIndex + 1 }}/{{ store.problems.length }}</span>
          <router-link
            v-if="nextProblem"
            :to="{ path: `/problems/${nextProblem.slug}`, query: { variant: activeVariant } }"
            class="pager-link"
            :title="nextProblem.title"
          >
            {{ t('workbench.next') }}
          </router-link>
        </div>
        <router-link
          v-if="accepted && nextUnsolved"
          :to="{ path: `/problems/${nextUnsolved.slug}`, query: { variant: activeVariant } }"
          class="pager-link next-unsolved"
          :title="t('workbench.goTo', { title: nextUnsolved.title })"
        >
          {{ t('workbench.nextUnsolved') }}
        </router-link>
        <div class="actions">
          <button
            v-if="accepted && !running"
            class="ghost"
            :title="t('workbench.redoTitle')"
            @click="redoProblem"
          >
            <span class="btn-icon">⟳</span>{{ t('workbench.redo') }}
          </button>
          <button v-if="!running" class="ghost" @click="resetCode">
            <span class="btn-icon">↺</span>{{ t('workbench.reset') }}
          </button>
          <button v-if="!running" class="ghost" @click="runCode">
            <span class="btn-icon play">▶</span>{{ t('workbench.run') }}
          </button>
          <button v-if="running" class="ghost danger" @click="cancelRun">
            <span class="btn-icon">✕</span>{{ t('workbench.cancel') }}
          </button>
          <button class="run" :disabled="running" @click="submitCode">
            <span v-if="running" class="loading">{{
              runningKind === 'submit' ? t('workbench.submitting') : t('workbench.running')
            }}</span>
            <template v-else>
              <span class="btn-icon">✓</span>{{ t('workbench.submit') }}
            </template>
          </button>
        </div>
      </div>

      <div class="split-body" ref="bodyEl" :style="splitStyle">
        <section class="pane desc">
          <div class="tabbar">
            <button
              class="tab"
              :class="{ active: activeTab === 'desc' }"
              @click="activeTab = 'desc'"
            >
              {{ t('workbench.tabDescription') }}
            </button>
            <button
              class="tab"
              :class="{ active: activeTab === 'notes' }"
              :disabled="!hasNotes"
              :title="hasNotes ? t('workbench.notesTitle') : t('workbench.notesDisabled')"
              @click="hasNotes && (activeTab = 'notes')"
            >
              {{ t('workbench.tabNotes') }}
            </button>
            <button
              class="tab"
              :class="{ active: activeTab === 'solution' }"
              :disabled="!accepted"
              :title="
                accepted ? t('workbench.solutionUnlocked') : t('workbench.solutionLocked')
              "
              @click="openSolution"
            >
              {{ t('workbench.tabSolution') }}
            </button>
          </div>
          <div class="tab-content" :class="{ 'solution-mode': activeTab !== 'desc' }">
            <MarkdownView v-if="activeTab === 'desc'" :key="problem.slug + '-desc'" :source="problem.description" />
            <div v-else-if="activeTab === 'notes'" class="notes-view">
              <div class="notes-toolbar">
                <span class="notes-hint">{{ t('workbench.notesTitle') }}</span>
              </div>
              <div class="notes-preview">
                <MarkdownView :key="problem.slug + '-notes'" :source="notes" />
              </div>
            </div>
            <div v-else class="solution-view">
              <div class="solution-toolbar">
                <button class="copy-btn" @click="copySolution">
                  <span class="btn-icon">⧉</span>{{
                    copied ? t('workbench.copied') : t('workbench.copyCode')
                  }}
                </button>
              </div>
              <div class="solution-preview">
                <MarkdownView :key="problem.slug + '-solution'" :source="solutionMarkdown" />
              </div>
            </div>
          </div>
        </section>

        <div class="divider" @mousedown="onDragStart"></div>

        <section class="pane code-pane">
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
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 12px;
  padding: 10px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
}
.back {
  color: var(--text-dim);
  font-size: 13px;
  white-space: nowrap;
}
.back:hover {
  color: var(--accent);
}
.title {
  font-size: 16px;
  font-weight: 700;
  min-width: 0;
  max-width: 40vw;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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
.variant-select {
  position: relative;
  display: inline-flex;
  align-items: center;
  margin-left: 4px;
}
.variant-select select {
  appearance: none;
  -webkit-appearance: none;
  font: inherit;
  font-size: 12px;
  color: var(--accent);
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px 26px 3px 12px;
  cursor: pointer;
}
.variant-select select:hover {
  border-color: var(--accent);
}
.variant-select select:focus {
  outline: none;
  border-color: var(--accent);
}
.variant-select option:disabled {
  color: var(--text-dim);
  opacity: 0.5;
}
.variant-caret {
  position: absolute;
  right: 11px;
  pointer-events: none;
  color: var(--accent);
  font-size: 10px;
}
.variant-notice {
  color: var(--blue);
  font-size: 11.5px;
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
.danger {
  color: var(--red);
  border-color: rgba(241, 106, 106, 0.45);
}
.danger:hover {
  border-color: var(--red);
  background: rgba(241, 106, 106, 0.08);
}
.split-body {
  --split: 0.45;
  flex: 1;
  min-height: 0;
  display: flex;
}
.pane {
  min-width: 0;
}
.desc {
  width: calc(var(--split) * 100%);
  display: flex;
  flex-direction: column;
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
  overflow-x: auto;
}
.tab {
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 13.5px;
  padding: 12px 14px;
  cursor: pointer;
  white-space: nowrap;
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
.solution-preview {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}
.notes-view {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.notes-toolbar {
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.notes-hint {
  color: var(--text-dim);
  font-size: 12px;
}
.notes-preview {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
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
  width: calc((1 - var(--split)) * 100% - 5px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.editor-wrap {
  flex: 1;
  min-height: 260px;
}

@media (max-width: 1100px) {
  .title {
    max-width: 30vw;
  }
}
@media (max-width: 900px) {
  .split-body {
    flex-direction: column;
    overflow-y: auto;
  }
  .divider {
    display: none;
  }
  .desc,
  .code-pane {
    width: 100%;
  }
  .desc {
    border-right: none;
    border-bottom: 1px solid var(--border);
    overflow: visible;
  }
  .tab-content,
  .tab-content.solution-mode,
  .solution-preview,
  .notes-preview {
    overflow: visible;
  }
  .code-pane {
    flex: none;
  }
  .editor-wrap {
    flex: none;
    height: 50vh;
    min-height: 300px;
  }
}
@media (max-width: 720px) {
  .subheader {
    gap: 8px;
    padding: 8px 12px;
  }
  .title {
    max-width: none;
    flex: 1 1 60%;
    font-size: 14px;
  }
  .pager {
    order: 5;
    width: 100%;
    justify-content: space-between;
    margin-left: 0;
    gap: 8px;
  }
  .actions {
    order: 10;
    width: 100%;
    margin-left: 0;
    gap: 6px;
  }
  .actions button {
    flex: 1;
    padding: 8px 6px;
    white-space: nowrap;
  }
  .tab-content {
    padding: 14px;
  }
  .tab {
    padding: 10px 10px;
    font-size: 13px;
  }
}
</style>
