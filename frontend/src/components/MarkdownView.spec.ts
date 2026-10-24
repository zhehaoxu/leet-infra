import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import MarkdownView from './MarkdownView.vue'

// 端到端确认：含 _ & \ 的 LaTeX 经 marked 渲染后不被强调/实体破坏（rope 题面回归）。
describe('MarkdownView 数学渲染', () => {
  it('块级公式里的下划线不产生 <em>，LaTeX 原文进入 DOM', async () => {
    const src = '矩阵 $$R = \\begin{pmatrix} \\cos\\theta_1 & -\\sin\\theta_1 \\\\ \\sin\\theta_2 & \\cos\\theta_2 \\end{pmatrix}$$ 结束'
    const wrapper = mount(MarkdownView, { props: { source: src } })
    // 冲刷 onMounted/nextTick 的 KaTeX auto-render
    await new Promise((r) => setTimeout(r, 0))
    const html = wrapper.html()

    // 若无保护，marked 会把 `theta_1 & -\sin\theta_1 ... \theta_2` 的下划线渲染成 <em>
    expect(html).not.toContain('<em>')
    // 关键 token 仍在（KaTeX 渲染则进 annotation，未渲染则原样保留），定界符内容未被吃掉
    expect(html).toContain('pmatrix')
    expect(html).toContain('cos')
  })

  it('普通 Markdown（标题/列表）正常渲染', async () => {
    const wrapper = mount(MarkdownView, { props: { source: '## 标题\n\n- 项目一' } })
    await new Promise((r) => setTimeout(r, 0))
    expect(wrapper.find('h2').exists()).toBe(true)
    expect(wrapper.find('li').exists()).toBe(true)
  })
})
