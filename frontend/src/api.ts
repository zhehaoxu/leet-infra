export interface ProblemSummary {
  slug: string
  framework: string
  title: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  tags: string[]
  solved: boolean
}

export interface ProblemDetail extends ProblemSummary {
  description: string
  starter_code: string
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
  getProblem: (slug: string) => request<ProblemDetail>(`/api/problems/${slug}`),
  run: (slug: string, code: string) =>
    request<RunResult>(`/api/problems/${slug}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    }),
  submit: (slug: string, code: string) =>
    request<RunResult>(`/api/problems/${slug}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    }),
  getSolution: (slug: string) =>
    request<{ slug: string; solution_code: string }>(`/api/problems/${slug}/solution`),
  collections: () => request<CollectionSummary[]>('/api/collections'),
  getCollection: (slug: string) =>
    request<CollectionDetailInfo>(`/api/collections/${slug}`),
}
