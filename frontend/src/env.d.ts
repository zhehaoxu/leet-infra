/// <reference types="vite/client" />

declare module 'katex/contrib/auto-render' {
  const renderMathInElement: (el: HTMLElement, opts?: any) => void
  export default renderMathInElement
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
