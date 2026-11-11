<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api, type ProjectStepDetail, type RunResult } from '../api'
import CodeEditor from '../components/CodeEditor.vue'
import MarkdownView from '../components/MarkdownView.vue'
import RunResultPanel from '../components/RunResultPanel.vue'
import { locale, t, type MessageKey } from '../i18n'
import { burstConfetti } from '../utils/confetti'

const props = defineProps<{ slug: string; step: number }>()

const detail = ref<ProjectStepDetail | null>(null)
const code = ref('')
const result = ref<RunResult | null>(null)
const running = ref(false)
const runningKind = ref<'run' | 'submit'>('run')
const runError = ref<string | null>(null)
const loadError = ref<string | null>(null)
const solutionCode = ref<string | null>(null)
const activeTab = ref<'desc' | 'notes' | 'solution'>('desc')
const accepted = ref(false)
const unlockNotice = ref(false)

// 项目草稿与普通题草稿分开存，key 带项目 slug + step
let currentDraftKey = ''
let draftTimer: ReturnType<typeof setTimeout> | null = null

function flushDraft() {
  if (draftTimer) {
    clearTimeout(draftTimer)
    draftTimer = null
  }
  if (currentDraftKey && detail.value && code.value !== detail.value.starter_code) {
    localStorage.setItem(currentDraftKey, code.value)
  }
}

watch(code, () => {
  if (draftTimer) clearTimeout(draftTimer)
  draftTimer = setTimeout(flushDraft, 400)
})

function difficultyLabel(d: string) {
  return t(`difficulty.${d.toLowerCase()}` as MessageKey)
}

const solutionMarkdown = computed(() =>
  solutionCode.value ? '```python\n' + solutionCode.value.trim() + '\n```' : ''
)
const scaffoldNote = computed(() => {
  const imports = detail.value?.step.scaffold_imports ?? []
  return imports.length
    ? t('step.scaffoldNote', { modules: imports.join(', ') })
    : t('step.noScaffold')
})

async function load() {
  flushDraft() // 先把上一步的草稿写回对应的 key
  detail.value = null
  result.value = null
  runError.value = null
  loadError.value = null
  solutionCode.value = null
  activeTab.value = 'desc'
  accepted.value = false
  unlockNotice.value = false
  try {
    const d = await api.getProjectStep(props.slug, props.step)
    detail.value = d
    currentDraftKey = `leet-infra:project:${props.slug}:${props.step}`
    code.value = localStorage.getItem(currentDraftKey) ?? d.starter_code
    accepted.value = d.step.completed
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e)
  }
}

// 语言切换只重拉文案，保留编辑器内容、结果与当前标签页
async function reloadForLocale() {
  if (!detail.value) return
  try {
    detail.value = await api.getProjectStep(props.slug, props.step)
  } catch (e) {
    runError.value = e instanceof Error ? e.message : String(e)
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
        ? await api.runProjectStep(props.slug, props.step, code.value)
        : await api.submitProjectStep(props.slug, props.step, code.value)
    result.value = r
    if (kind === 'submit' && r.accepted) {
      const wasAccepted = accepted.value
      accepted.value = true
      // 进度记录与判题解耦：写失败只影响徽章，不改变判题结果
      try {
        await api.markProjectStep(props.slug, props.step)
        if (detail.value?.step.final_step) unlockNotice.value = true
      } catch {
        /* 进度台账写入失败不影响本题 AC */
      }
      if (!wasAccepted) burstConfetti()
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

function stepLink(step: number) {
  return `/projects/${props.slug}/steps/${step}`
}

function resetCode() {
  if (detail.value) {
    localStorage.removeItem(currentDraftKey)
    code.value = detail.value.starter_code
    result.value = null
  }
}

async function cancelRun() {
  try {
    await api.cancelJudge()
  } catch {
    /* 进程已结束时无可取消 */
  }
}

async function redoStep() {
  if (!accepted.value || running.value) return
  if (!confirm(t('step.redoConfirm'))) return
  try {
    await api.unmarkProjectStep(props.slug, props.step)
  } catch (e) {
    runError.value = e instanceof Error ? e.message : String(e)
    return
  }
  accepted.value = false
  solutionCode.value = null
  activeTab.value = 'desc'
  result.value = null
  unlockNotice.value = false
  code.value = detail.value?.starter_code ?? ''
  localStorage.removeItem(currentDraftKey)
}

async function openSolution() {
  if (!accepted.value) return
  activeTab.value = 'solution'
  if (!solutionCode.value) {
    try {
      const r = await api.getProjectSolution(props.slug, props.step)
      solutionCode.value = r.solution_code
    } catch (e) {
      runError.value = e instanceof Error ? e.message : String(e)
      activeTab.value = 'desc'
    }
  }
}

const copied = ref(false)
let copiedTimer: ReturnType<typeof setTimeout> | null = null

async function copySolution() {
  if (!solutionCode.value) return
  try {
    await navigator.clipboard.writeText(solutionCode.value)
  } catch {
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

watch([() => props.slug, () => props.step], () => void load())
watch(locale, reloadForLocale)
onMounted(load)
onBeforeUnmount(() => {
  if (draftTimer) clearTimeout(draftTimer)
  flushDraft()
})
</script>

<template>
  <div class="project-step">
    <div v-if="loadError" class="empty">{{ t('common.loadFailed', { error: loadError }) }}</div>
    <div v-else-if="!detail" class="empty">{{ t('step.loading') }}</div>
    <template v-else>
      <div class="subheader">
        <router-link :to="`/projects/${slug}`" class="back">{{ t('step.back') }}</router-link>
        <span class="step-badge">
          {{ t('step.stepLabel', { step: detail.step.step, total: detail.total_steps }) }}
        </span>
        <span class="title" :title="detail.step.title">{{ detail.step.title }}</span>
        <em class="badge" :class="detail.step.difficulty">
          {{ difficultyLabel(detail.step.difficulty) }}
        </em>
        <span v-if="accepted" class="ac-tag">{{ t('step.completed') }}</span>
        <div class="pager">
          <router-link
            v-if="detail.prev_step"
            :to="stepLink(detail.prev_step)"
            class="pager-link"
          >
            {{ t('step.prevStep') }}
          </router-link>
          <router-link
            v-if="detail.next_step"
            :to="stepLink(detail.next_step)"
            class="pager-link"
          >
            {{ t('step.nextStep') }}
          </router-link>
        </div>
        <div class="actions">
          <button v-if="accepted && !running" class="ghost" @click="redoStep">
            <span class="btn-icon">⟳</span>{{ t('step.redo') }}
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
            <span v-if="running" class="loading">
              {{ runningKind === 'submit' ? t('workbench.submitting') : t('workbench.running') }}
            </span>
            <template v-else>
              <span class="btn-icon">✓</span>{{ t('workbench.submit') }}
            </template>
          </button>
        </div>
      </div>

      <div class="scaffold-bar">
        <span class="scaffold-note">{{ scaffoldNote }}</span>
        <span class="verify">{{ detail.step.verification.join(' · ') }}</span>
        <span v-for="dep in detail.step.depends_on" :key="dep" class="dep">
          {{ t('project.dependsOn', { steps: dep }) }}
        </span>
        <span v-if="detail.step.final_step" class="tag-final">{{ t('project.finalStep') }}</span>
        <span v-if="unlockNotice" class="unlock">{{ t('step.unlockOverview') }}</span>
      </div>

      <div class="split-body">
        <section class="pane desc">
          <div class="tabbar">
            <button class="tab" :class="{ active: activeTab === 'desc' }" @click="activeTab = 'desc'">
              {{ t('workbench.tabDescription') }}
            </button>
            <button class="tab" :class="{ active: activeTab === 'notes' }" @click="activeTab = 'notes'">
              {{ t('workbench.tabNotes') }}
            </button>
            <button
              class="tab"
              :class="{ active: activeTab === 'solution' }"
              :disabled="!accepted"
              :title="accepted ? t('workbench.solutionUnlocked') : t('workbench.solutionLocked')"
              @click="openSolution"
            >
              {{ t('workbench.tabSolution') }}
            </button>
          </div>
          <div class="tab-content" :class="{ 'solution-mode': activeTab !== 'desc' }">
            <MarkdownView
              v-if="activeTab === 'desc'"
              :key="`${slug}-${step}-desc`"
              :source="detail.description"
            />
            <div v-else-if="activeTab === 'notes'" class="notes-view">
              <div class="notes-hint">{{ t('step.notesTitle') }}</div>
              <MarkdownView :key="`${slug}-${step}-notes`" :source="detail.notes" />
            </div>
            <div v-else class="solution-view">
              <div class="solution-toolbar">
                <button class="copy-btn" @click="copySolution">
                  <span class="btn-icon">⧉</span>
                  {{ copied ? t('workbench.copied') : t('workbench.copyCode') }}
                </button>
              </div>
              <MarkdownView :source="solutionMarkdown" />
            </div>
          </div>
        </section>

        <div class="divider"></div>

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
.project-step {
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
.step-badge {
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 11.5px;
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 2px 9px;
  white-space: nowrap;
}
.title {
  font-size: 16px;
  font-weight: 700;
  min-width: 0;
  max-width: 34vw;
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
.pager {
  display: flex;
  gap: 12px;
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
.danger {
  color: var(--red);
  border-color: rgba(241, 106, 106, 0.45);
}
.scaffold-bar {
  flex: none;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 14px;
  padding: 7px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elevated);
  font-size: 11.5px;
  color: var(--text-dim);
}
.scaffold-note {
  font-family: var(--font-mono);
  color: var(--blue);
}
.verify,
.dep {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 1px 9px;
}
.tag-final {
  color: var(--yellow);
  border: 1px solid rgba(230, 194, 96, 0.5);
  border-radius: 999px;
  padding: 1px 9px;
}
.unlock {
  margin-left: auto;
  color: var(--accent);
  font-weight: 700;
}
.split-body {
  flex: 1;
  min-height: 0;
  display: flex;
}
.pane {
  min-width: 0;
}
.desc {
  width: 45%;
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
.notes-view,
.solution-view {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
}
.notes-hint {
  color: var(--text-dim);
  font-size: 12px;
  flex: none;
}
.solution-toolbar {
  display: flex;
  justify-content: flex-end;
  flex: none;
}
.copy-btn {
  border-radius: 8px;
  padding: 5px 12px;
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text);
  font-size: 12.5px;
}
.divider {
  flex: none;
  width: 5px;
}
.code-pane {
  width: calc(55% - 5px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.editor-wrap {
  flex: 1;
  min-height: 260px;
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
  .tab-content.solution-mode {
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
  .title {
    max-width: none;
    flex: 1 1 60%;
    font-size: 14px;
  }
  .pager,
  .actions {
    order: 5;
    width: 100%;
    margin-left: 0;
  }
  .actions button {
    flex: 1;
    padding: 8px 6px;
    white-space: nowrap;
  }
  .unlock {
    margin-left: 0;
  }
}
</style>
