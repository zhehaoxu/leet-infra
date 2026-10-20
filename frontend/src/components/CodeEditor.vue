<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import monaco from '../monaco-setup'

const props = defineProps<{ modelValue: string; readOnly?: boolean }>()
const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'run'): void
  (e: 'submit'): void
}>()

const container = ref<HTMLDivElement | null>(null)
let editor: monaco.editor.IStandaloneCodeEditor | null = null

onMounted(() => {
  if (!container.value) return
  editor = monaco.editor.create(container.value, {
    value: props.modelValue,
    language: 'python',
    theme: 'leet-infra',
    automaticLayout: true,
    fontSize: 13,
    fontFamily: 'var(--font-mono), monospace',
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    tabSize: 4,
    readOnly: props.readOnly ?? false,
    lineNumbers: props.readOnly ? 'off' : 'on',
    renderLineHighlight: props.readOnly ? 'none' : 'line',
  })
  editor.onDidChangeModelContent(() => {
    emit('update:modelValue', editor?.getValue() ?? '')
  })
  // Cmd+Enter / Ctrl+Enter 运行样例；Cmd+Shift+Enter / Ctrl+Shift+Enter 提交（只读实例不注册）
  if (!props.readOnly) {
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => emit('run'))
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter, () =>
      emit('submit')
    )
  }
})

watch(
  () => props.modelValue,
  (v) => {
    if (editor && editor.getValue() !== v) editor.setValue(v)
  }
)

onBeforeUnmount(() => {
  editor?.dispose()
  editor = null
})
</script>

<template>
  <div ref="container" class="code-editor" />
</template>

<style scoped>
.code-editor {
  width: 100%;
  height: 100%;
  min-height: 200px;
}
</style>
