<script setup lang="ts">
import { computed } from 'vue'

// 圆环进度：底环 + 已解决比例的薄荷绿弧，从 12 点方向起算
const props = defineProps<{ value: number; total: number; size?: number }>()

const CIRC = 2 * Math.PI * 9
const dash = computed(() => {
  const ratio = props.total ? Math.min(props.value / props.total, 1) : 0
  return `${ratio * CIRC} ${CIRC}`
})
// 满环/零进度用直头，避免 round 端点在 0 长度时画出孤立小点
const linecap = computed(() => (props.total > 0 && props.value >= props.total ? 'butt' : 'round'))
</script>

<template>
  <svg
    class="progress-ring"
    :width="size ?? 24"
    :height="size ?? 24"
    viewBox="0 0 24 24"
    role="img"
    :aria-label="`${value}/${total}`"
  >
    <circle cx="12" cy="12" r="9" fill="none" stroke="var(--border)" :stroke-width="size && size < 18 ? 3.5 : 3" />
    <circle
      v-if="value > 0"
      cx="12"
      cy="12"
      r="9"
      fill="none"
      stroke="var(--accent)"
      :stroke-width="size && size < 18 ? 3.5 : 3"
      :stroke-dasharray="dash"
      :stroke-linecap="linecap"
      transform="rotate(-90 12 12)"
    />
  </svg>
</template>

<style scoped>
.progress-ring {
  display: block;
}
</style>
