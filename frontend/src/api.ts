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
  listProblems: () => request<ProblemSummary[]>('/api/problems'),
  getProblem: (slug: string, variant?: string) =>
    request<ProblemDetail>(
      `/api/problems/${slug}${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`
    ),
  run: (slug: string, code: string, variant?: string) =>
    request<RunResult>(`/api/problems/${slug}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant }),
    }),
  submit: (slug: string, code: string, variant?: string) =>
    request<RunResult>(`/api/problems/${slug}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant }),
    }),
  getSolution: (slug: string, variant?: string) =>
    request<{ slug: string; variant: string; solution_code: string }>(
      `/api/problems/${slug}/solution${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`
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
    request<{ imported: number; skipped: number }>('/api/progress/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),
  cancelJudge: () =>
    request<{ cancelled: boolean }>('/api/judge/cancel', { method: 'POST' }),
  collections: () => request<CollectionSummary[]>('/api/collections'),
  getCollection: (slug: string) =>
    request<CollectionDetailInfo>(`/api/collections/${slug}`),
}
