<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, type ProjectDetailInfo, type ProjectStep } from '../api'
import MarkdownView from '../components/MarkdownView.vue'
import ProgressRing from '../components/ProgressRing.vue'
import { locale, t, type MessageKey } from '../i18n'

const props = defineProps<{ slug: string }>()

type PageTab = 'milestones' | 'architecture' | 'overview'
const tab = ref<PageTab>('milestones')
const detail = ref<ProjectDetailInfo | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    detail.value = await api.getProject(props.slug)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    detail.value = null
  } finally {
    loading.value = false
  }
}

onMounted(load)
watch(locale, load)
watch(() => props.slug, load)

const completedCount = computed(() => detail.value?.progress.completed_steps.length ?? 0)
const done = computed(() => new Set(detail.value?.progress.completed_steps ?? []))

function difficultyLabel(d: string) {
  return t(`difficulty.${d.toLowerCase()}` as MessageKey)
}

function pendingDeps(step: ProjectStep): number[] {
  return step.depends_on.filter((n) => !done.value.has(n))
}

function stepLink(step: number) {
  return `/projects/${props.slug}/steps/${step}`
}

function statusLabel(status: string) {
  return status === 'live' ? t('project.statusLive') : t('project.statusDraft')
}
</script>

<template>
  <div class="project-page">
    <div v-if="error" class="empty error">{{ t('common.loadFailed', { error: error ?? '' }) }}</div>
    <div v-else-if="loading || !detail" class="empty">{{ t('common.loading') }}</div>
    <template v-else>
      <router-link to="/" class="back">{{ t('project.back') }}</router-link>

      <header class="project-hero">
        <div class="hero-main">
          <div class="hero-title-row">
            <h1>{{ detail.name }}</h1>
            <em class="badge" :class="detail.difficulty">{{ difficultyLabel(detail.difficulty) }}</em>
            <span class="status-chip" :class="detail.status">{{ statusLabel(detail.status) }}</span>
          </div>
          <p class="hero-summary">{{ detail.summary }}</p>
          <div class="hero-tags">
            <span class="chip">{{ detail.framework }}</span>
            <span class="chip">{{ t('project.stepsUnit', { count: detail.step_count }) }}</span>
            <span class="chip">scaffold {{ detail.scaffold_version }}</span>
            <span class="chip hint">{{ t('project.dependencyHint') }}</span>
          </div>
        </div>
        <div class="hero-progress">
          <ProgressRing :value="completedCount" :total="detail.step_count" :size="76" />
          <span class="hero-progress-text">
            {{ t('project.progress', { done: completedCount, total: detail.step_count }) }}
          </span>
        </div>
      </header>

      <div class="page-tabs">
        <button class="tab" :class="{ active: tab === 'milestones' }" @click="tab = 'milestones'">
          {{ t('project.tabMilestones') }}
        </button>
        <button class="tab" :class="{ active: tab === 'architecture' }" @click="tab = 'architecture'">
          {{ t('project.tabArchitecture') }}
        </button>
        <button class="tab" :class="{ active: tab === 'overview' }" @click="tab = 'overview'">
          {{ t('project.tabOverview') }}
          <span v-if="!detail.overview_unlocked" class="tab-lock" aria-hidden="true">·</span>
        </button>
      </div>

      <div v-if="tab === 'milestones'" class="milestone-list">
        <div
          v-for="(step, index) in detail.steps"
          :key="step.step"
          class="milestone"
          :class="{ done: step.completed, final: step.final_step }"
        >
          <div class="rail">
            <span class="node" :class="{ done: step.completed, final: step.final_step }">
              {{ step.completed ? '✓' : step.step }}
            </span>
            <span v-if="index < detail.steps.length - 1" class="rail-line"></span>
          </div>

          <div class="milestone-body">
            <div class="milestone-head">
              <span class="step-no">Step {{ step.step }}</span>
              <h3>{{ step.title }}</h3>
              <em class="badge" :class="step.difficulty">{{ difficultyLabel(step.difficulty) }}</em>
              <span v-if="step.final_step" class="tag final">{{ t('project.finalStep') }}</span>
              <span v-if="detail.review_step === step.step" class="tag review">
                {{ t('project.reviewStep') }}
              </span>
              <router-link :to="stepLink(step.step)" class="cta">
                {{ step.completed ? t('project.redo') : t('project.start') }}
              </router-link>
            </div>

            <p class="why">
              <span class="why-label">{{ t('project.why') }}</span>
              {{ step.why }}
            </p>

            <div class="milestone-meta">
              <span v-if="step.depends_on.length" class="chip">
                {{ t('project.dependsOn', { steps: step.depends_on.join(', ') }) }}
              </span>
              <span v-else class="chip">{{ t('step.noScaffold') }}</span>
              <span v-for="v in step.verification" :key="v" class="chip layer">{{ v }}</span>
              <span class="chip">
                {{ t('project.scaffold', { count: step.scaffold_imports.length }) }}
              </span>
              <span
                v-if="!step.completed && pendingDeps(step).length"
                class="chip recommend"
              >
                {{ t('project.recommend', { steps: pendingDeps(step).join(', ') }) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div v-else-if="tab === 'architecture'" class="markdown-panel">
        <MarkdownView :source="detail.architecture" />
      </div>

      <div v-else class="markdown-panel">
        <template v-if="detail.overview">
          <div class="unlocked-banner">{{ t('project.unlocked') }}</div>
          <MarkdownView :source="detail.overview || ''" />
        </template>
        <div v-else class="locked">
          <div class="lock-mark" aria-hidden="true">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none">
              <rect x="4" y="10" width="16" height="10" rx="2" stroke="var(--text-dim)" stroke-width="1.6" />
              <path d="M8 10V7a4 4 0 0 1 8 0v3" stroke="var(--text-dim)" stroke-width="1.6" />
              <circle cx="12" cy="15" r="1.4" fill="var(--accent)" />
            </svg>
          </div>
          <h3>{{ t('project.overviewLocked') }}</h3>
          <p>
            {{ t('project.overviewLockedHint', { step: detail.overview_unlock_step ?? detail.step_count }) }}
          </p>
          <router-link
            v-if="detail.overview_unlock_step"
            :to="stepLink(detail.overview_unlock_step)"
            class="lock-cta"
          >
            {{ t('project.goFinal') }}
          </router-link>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.project-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 32px 24px 60px;
}
.empty {
  text-align: center;
  color: var(--text-dim);
  padding: 80px 0;
}
.empty.error {
  color: var(--red);
}
.back {
  display: inline-block;
  color: var(--text-dim);
  font-size: 13px;
  margin-bottom: 22px;
}
.back:hover {
  color: var(--accent);
}
.project-hero {
  display: flex;
  gap: 24px;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}
.hero-main {
  min-width: 0;
}
.hero-title-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.hero-title-row h1 {
  font-size: 25px;
  margin: 0;
}
.status-chip {
  font-size: 11.5px;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--text-dim);
}
.status-chip.live {
  color: var(--accent);
  border-color: var(--accent);
}
.hero-summary {
  color: var(--text-dim);
  font-size: 13.5px;
  line-height: 1.7;
  margin: 12px 0;
  max-width: 640px;
}
.hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip {
  display: inline-flex;
  align-items: center;
  font-size: 11.5px;
  color: var(--text-dim);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 10px;
  background: var(--bg-elevated);
  white-space: nowrap;
}
.chip.hint {
  border-style: dashed;
}
.chip.layer {
  color: var(--blue);
  border-color: rgba(106, 169, 241, 0.4);
  font-family: var(--font-mono);
}
.chip.recommend {
  color: var(--yellow);
  border-color: rgba(230, 194, 96, 0.45);
}
.hero-progress {
  flex: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.hero-progress-text {
  color: var(--text-dim);
  font-size: 12px;
  font-family: var(--font-mono);
}
.page-tabs {
  display: flex;
  gap: 18px;
  margin: 20px 0 22px;
  border-bottom: 1px solid var(--border);
}
.tab {
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 14px;
  padding: 8px 2px 12px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}
.tab:hover {
  color: var(--text);
}
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
.tab-lock {
  color: var(--yellow);
  font-weight: 700;
}

/* 里程碑链：左侧轨道 + 右侧卡片 */
.milestone-list {
  display: flex;
  flex-direction: column;
}
.milestone {
  display: grid;
  grid-template-columns: 46px minmax(0, 1fr);
  gap: 14px;
}
.rail {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.node {
  flex: none;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-panel);
  color: var(--text-dim);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 12.5px;
  font-weight: 700;
  margin-top: 16px;
}
.node.done {
  color: #06231a;
  background: var(--accent);
  border-color: var(--accent);
}
.node.final:not(.done) {
  color: var(--yellow);
  border-color: rgba(230, 194, 96, 0.6);
}
.rail-line {
  flex: 1;
  width: 1px;
  min-height: 24px;
  background: var(--border);
  margin-bottom: 4px;
}
.milestone-body {
  margin-bottom: 14px;
  padding: 16px 18px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-panel);
  min-width: 0;
}
.milestone.done .milestone-body {
  border-color: rgba(45, 212, 167, 0.35);
}
.milestone.final .milestone-body {
  border-left: 3px solid var(--yellow);
}
.milestone-body:hover {
  background: var(--bg-elevated);
}
.milestone-head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.step-no {
  color: var(--accent);
  font-size: 11.5px;
  font-family: var(--font-mono);
  letter-spacing: 0.3px;
}
.milestone-head h3 {
  font-size: 15px;
  margin: 0 4px 0 0;
}
.tag {
  font-size: 11px;
  padding: 2px 9px;
  border-radius: 999px;
}
.tag.final {
  color: var(--yellow);
  background: rgba(230, 194, 96, 0.12);
}
.tag.review {
  color: var(--blue);
  background: rgba(106, 169, 241, 0.12);
}
.cta {
  margin-left: auto;
  color: var(--accent);
  font-size: 12.5px;
  white-space: nowrap;
}
.cta:hover {
  text-decoration: underline;
}
.why {
  color: var(--text-dim);
  font-size: 12.5px;
  line-height: 1.7;
  margin: 10px 0 12px;
}
.why-label {
  color: var(--text);
  font-weight: 600;
  margin-right: 8px;
}
.milestone-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}
.markdown-panel {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-panel);
  padding: 22px 26px;
}
.unlocked-banner {
  color: var(--accent);
  font-size: 12.5px;
  font-weight: 700;
  padding-bottom: 14px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.locked {
  text-align: center;
  padding: 44px 20px 52px;
}
.lock-mark {
  display: flex;
  justify-content: center;
  margin-bottom: 14px;
}
.locked h3 {
  font-size: 16px;
  margin: 0 0 10px;
}
.locked p {
  color: var(--text-dim);
  font-size: 13px;
  line-height: 1.7;
  max-width: 480px;
  margin: 0 auto 20px;
}
.lock-cta {
  display: inline-block;
  color: #06231a;
  background: var(--accent);
  border-radius: 8px;
  padding: 8px 18px;
  font-size: 13px;
  font-weight: 700;
}

@media (max-width: 720px) {
  .project-page {
    padding: 22px 14px 40px;
  }
  .project-hero {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
  .hero-progress {
    flex-direction: row;
    gap: 12px;
  }
  .hero-title-row h1 {
    font-size: 21px;
  }
  .milestone {
    grid-template-columns: 30px minmax(0, 1fr);
    gap: 8px;
  }
  .node {
    width: 26px;
    height: 26px;
    font-size: 11.5px;
  }
  .milestone-body {
    padding: 13px;
  }
  .cta {
    margin-left: 0;
  }
  .markdown-panel {
    padding: 16px;
  }
}
</style>
