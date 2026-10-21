import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, type ProblemSummary } from '../api'

export const useProblemStore = defineStore('problems', () => {
  const problems = ref<ProblemSummary[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const solvedCount = computed(() => problems.value.filter((p) => p.solved).length)

  async function fetchProblems() {
    loading.value = true
    error.value = null
    try {
      problems.value = await api.listProblems()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  function markSolved(slug: string) {
    const p = problems.value.find((x) => x.slug === slug)
    if (p) p.solved = true
  }

  function unmarkSolved(slug: string) {
    const p = problems.value.find((x) => x.slug === slug)
    if (p) p.solved = false
  }

  return { problems, loading, error, solvedCount, fetchProblems, markSolved, unmarkSolved }
})
