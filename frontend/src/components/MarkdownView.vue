<script setup lang="ts">
import { computed, onMounted, ref, nextTick, watch } from 'vue'
import { marked } from 'marked'
import renderMathInElement from 'katex/contrib/auto-render'
import 'katex/dist/katex.min.css'

const props = defineProps<{ source: string }>()
const root = ref<HTMLElement | null>(null)

const html = computed(() => marked.parse(props.source, { async: false }) as string)

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
</style>
