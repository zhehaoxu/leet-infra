/** 编辑器字号全局设置：localStorage 持久化，所有 Monaco 实例共享。 */
import { ref, watch } from 'vue'

const KEY = 'leet-infra:editor-fontsize'
export const MIN_FONT = 10
export const MAX_FONT = 24
export const DEFAULT_FONT = 13

function initial(): number {
  const v = Number(localStorage.getItem(KEY))
  return Number.isInteger(v) && v >= MIN_FONT && v <= MAX_FONT ? v : DEFAULT_FONT
}

/** 全局响应式字号；改一处，所有编辑器实时跟随。 */
export const editorFontSize = ref(initial())

watch(editorFontSize, (v) => {
  const size = Math.min(MAX_FONT, Math.max(MIN_FONT, v))
  if (size !== v) {
    editorFontSize.value = size // 越界回推，避免持久化非法值
    return
  }
  localStorage.setItem(KEY, String(size))
})

export function bumpFontSize(delta: number) {
  editorFontSize.value = Math.min(MAX_FONT, Math.max(MIN_FONT, editorFontSize.value + delta))
}
