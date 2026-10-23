<script setup lang="ts">
import { computed, onMounted, nextTick, ref, watch } from 'vue'
import { marked } from 'marked'
import renderMathInElement from 'katex/contrib/auto-render'
import 'katex/dist/katex.min.css'

const props = defineProps<{ source: string }>()
const root = ref<HTMLElement | null>(null)

// 先保护代码块/行内码，再把 $$..$$、$..$ 抽出成占位符交给 marked，
// marked 处理完再原样回填。避免 LaTeX 里的 _ & \\ 等被 Markdown 当强调/实体破坏，
// 致使 KaTeX auto-render 匹配不到定界符（历史上 rope 题面块级矩阵公式裸奔的根因）。
function extractMath(src: string): { text: string; math: string[] } {
  const code: string[] = []
  const math: string[] = []
  const noCode = src.replace(
    /(```[\s\S]*?```|~~~[\s\S]*?~~~|`+[^`\n]+`+)/g,
    (c) => {
      code.push(c)
      return `\uE000C${code.length - 1}\uE000`
    }
  )
  const noMath = noCode
    .replace(/\$\$[\s\S]+?\$\$/g, (mm) => {
      math.push(mm)
      return `\uE000M${math.length - 1}\uE000`
    })
    .replace(/\$[^$\n]+?\$/g, (mm) => {
      math.push(mm)
      return `\uE000M${math.length - 1}\uE000`
    })
  const restored = noMath.replace(/\uE000C(\d+)\uE000/g, (_m, i) => code[+i])
  return { text: restored, math }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

const html = computed(() => {
  const { text, math } = extractMath(props.source)
  const parsed = marked.parse(text, { async: false }) as string
  // 回填数学：marked 从未见过其内容，故特殊字符需按 HTML 文本转义后原样插入，
  // auto-render 从 textContent 读回即得原始 LaTeX
  return parsed.replace(/\uE000M(\d+)\uE000/g, (_m, i) => escapeHtml(math[+i]))
})

async function renderMath() {
  await nextTick()
  if (root.value) {
    renderMathInElement(root.value, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
    })
  }
}

onMounted(renderMath)
watch(html, renderMath)
</script>

<template>
  <div class="markdown" ref="root" v-html="html" />
</template>

<style scoped>
.markdown {
  line-height: 1.75;
  word-break: break-word;
}
.markdown :deep(h1) {
  font-size: 21px;
  margin: 0 0 14px;
}
.markdown :deep(h2) {
  font-size: 16px;
  margin: 22px 0 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.markdown :deep(p),
.markdown :deep(ul),
.markdown :deep(ol) {
  margin: 8px 0;
}
.markdown :deep(ul),
.markdown :deep(ol) {
  padding-left: 22px;
}
.markdown :deep(code) {
  font-family: var(--font-mono);
  font-size: 12.5px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 6px;
}
.markdown :deep(pre) {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  overflow-x: auto;
}
.markdown :deep(pre code) {
  background: transparent;
  border: none;
  padding: 0;
}
.markdown :deep(blockquote) {
  margin: 10px 0;
  padding: 6px 14px;
  border-left: 3px solid var(--accent);
  background: var(--accent-dim);
  border-radius: 0 6px 6px 0;
}
.markdown :deep(table) {
  border-collapse: collapse;
  margin: 10px 0;
}
.markdown :deep(th),
.markdown :deep(td) {
  border: 1px solid var(--border);
  padding: 5px 12px;
}
/* 宽公式（如多个 pmatrix 并排）在窄栏里不裁切，改为横向滚动兜底 */
.markdown :deep(.katex-display) {
  overflow-x: auto;
  overflow-y: hidden;
  max-width: 100%;
  padding: 2px 0;
}
</style>
