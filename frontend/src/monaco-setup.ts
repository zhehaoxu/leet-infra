import * as monaco from 'monaco-editor'
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker'

// Python 高亮只需要基础 editor worker（无需 language service worker）
;(self as any).MonacoEnvironment = {
  getWorker: () => new editorWorker(),
}

monaco.editor.defineTheme('leet-infra', {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '5f6b85', fontStyle: 'italic' },
    { token: 'keyword', foreground: '2dd4a7' },
    { token: 'string', foreground: 'e6c260' },
    { token: 'number', foreground: '6aa9f1' },
    { token: 'identifier', foreground: 'd7dbe7' },
  ],
  colors: {
    'editor.background': '#12151f',
    'editorGutter.background': '#12151f',
    'editor.lineHighlightBackground': '#1a1e2c',
    'editorLineNumber.foreground': '#4a5268',
    'editorIndentGuide.background': '#20242f',
  },
})

export default monaco
