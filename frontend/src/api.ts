import { locale } from './i18n'

function withLang(path: string): string {
  return `${path}${path.includes('?') ? '&' : '?'}lang=${locale.value}`
}

export interface CollectionRef {
  slug: string
  name: string
}

export interface VariantInfo {
  key: string
  label: string
  available: boolean
}

export interface ProblemSummary {
  slug: string
  framework: string
  title: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  tags: string[]
  solved: boolean
  collections?: CollectionRef[]
  variants?: VariantInfo[]
}

export interface ProblemDetail extends ProblemSummary {
  description: string
  notes: string
  starter_code: string
  active_variant: string
}

export interface CollectionSummary {
  slug: string
  name: string
  description: string
  problem_count: number
  solved_count: number
}

export interface CollectionDetailInfo {
  slug: string
  name: string
  description: string
  problems: ProblemSummary[]
}

// --- Project collections (separate registry / ledger / progress) ------------
export interface ProjectSummary {
  slug: string
  name: string
  summary: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  status: 'draft' | 'live'
  step_count: number
  display_order: number
  completed_steps: number
}

export interface ProjectStep {
  step: number
  slug: string
  title: string
  title_en: string
  why: string
  depends_on: number[]
  verification: string[]
  scaffold_imports: string[]
  final_step: boolean
  status: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  tags: string[]
  completed: boolean
}

export interface ProjectProgress {
  completed_steps: number[]
  unlocked_overview: boolean
}

export interface ProjectDetailInfo {
  slug: string
  name: string
  summary: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  status: 'draft' | 'live'
  step_count: number
  framework: string
  scaffold_version: string
  architecture: string
  overview: string | null
  overview_unlocked: boolean
  overview_unlock_step: number | null
  review_step: number | null
  progress: ProjectProgress
  steps: ProjectStep[]
}

export interface ProjectStepDetail {
  project: { slug: string; name: string; name_en: string }
  step: {
    step: number
    slug: string
    title: string
    difficulty: 'Easy' | 'Medium' | 'Hard'
    why: string
    depends_on: number[]
    verification: string[]
    scaffold_imports: string[]
    final_step: boolean
    completed: boolean
  }
  description: string
  notes: string
  starter_code: string
  active_variant: string
  total_steps: number
  prev_step: number | null
  next_step: number | null
}

export interface CaseResult {
  name: string
  passed: boolean
  message: string | null
  elapsed_ms: number
}

export interface RunResult {
  status: string
  accepted: boolean
  passed: number
  total: number
  mode?: 'run' | 'submit'
  cases: CaseResult[]
  stdout: string
  error: string | null
  elapsed_ms: number | null
  /** 参考解各用例耗时 {case_name: ms}；首次 AC 后台测量，后续提交附带 */
  benchmark?: Record<string, number>
  /** 本次评测实际使用的 variant（不可用的请求会回退） */
  variant?: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init)
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const body = await resp.json()
      if (body.detail) detail = body.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return resp.json() as Promise<T>
}

export const api = {
  listProblems: () => request<ProblemSummary[]>(withLang('/api/problems')),
  getProblem: (slug: string, variant?: string) =>
    request<ProblemDetail>(
      withLang(
        `/api/problems/${slug}${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`
      )
    ),
  run: (slug: string, code: string, variant?: string) =>
    request<RunResult>(`/api/problems/${slug}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant, lang: locale.value }),
    }),
  submit: (slug: string, code: string, variant?: string) =>
    request<RunResult>(`/api/problems/${slug}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant, lang: locale.value }),
    }),
  getSolution: (slug: string, variant?: string) =>
    request<{ slug: string; variant: string; solution_code: string }>(
      withLang(
        `/api/problems/${slug}/solution${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`
      )
    ),
  resetProgress: () =>
    request<{ total: number; solved: number }>('/api/progress/reset', { method: 'POST' }),
  unmarkProblem: (slug: string) =>
    request<{ slug: string; removed: boolean }>(`/api/problems/${slug}/unmark`, { method: 'POST' }),
  exportProgress: async () => {
    const resp = await fetch('/api/progress/export')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    return resp.blob()
  },
  importProgress: (data: unknown) =>
    request<{ imported: number; skipped: number }>(withLang('/api/progress/import'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),
  cancelJudge: () =>
    request<{ cancelled: boolean }>('/api/judge/cancel', { method: 'POST' }),
  collections: () => request<CollectionSummary[]>(withLang('/api/collections')),
  getCollection: (slug: string) =>
    request<CollectionDetailInfo>(withLang(`/api/collections/${slug}`)),

  // 项目侧接口与普通题库完全分开：独立注册表、独立判题路径、独立进度台账
  projects: () => request<ProjectSummary[]>(withLang('/api/projects')),
  getProject: (slug: string) =>
    request<ProjectDetailInfo>(withLang(`/api/projects/${slug}`)),
  getProjectStep: (slug: string, step: number) =>
    request<ProjectStepDetail>(withLang(`/api/projects/${slug}/steps/${step}`)),
  runProjectStep: (slug: string, step: number, code: string) =>
    request<RunResult>(`/api/projects/${slug}/steps/${step}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant: 'pytorch', lang: locale.value }),
    }),
  submitProjectStep: (slug: string, step: number, code: string) =>
    request<RunResult>(`/api/projects/${slug}/steps/${step}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant: 'pytorch', lang: locale.value }),
    }),
  getProjectSolution: (slug: string, step: number) =>
    request<{ project: string; step: number; variant: string; solution_code: string }>(
      withLang(`/api/projects/${slug}/steps/${step}/solution`)
    ),
  getProjectProgress: (slug: string) =>
    request<ProjectProgress>(`/api/projects/${slug}/progress`),
  markProjectStep: (slug: string, step: number) =>
    request<ProjectProgress>(`/api/projects/${slug}/progress`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ step }),
    }),
  unmarkProjectStep: (slug: string, step: number) =>
    request<ProjectProgress>(`/api/projects/${slug}/progress/unmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ step }),
    }),
}
