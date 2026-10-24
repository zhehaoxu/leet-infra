import { describe, it, expect } from 'vitest'
import { extractMath, escapeHtml } from './markdownMath'

// 回归：LaTeX 里的 _ & \ 等曾被 marked 当强调/实体破坏，致使 KaTeX 匹配不到定界符。
// 注意：JS 字符串里每个 LaTeX 反斜杠都要写成 `\\`，否则 \t/\b 等会被当转义吞掉。
describe('extractMath 数学保护', () => {
  it('块级 $$..$$ 抽出为占位符，正文不含原始公式', () => {
    const { text, math } = extractMath('前 $$a_1 \\times b_2$$ 后')
    expect(math).toHaveLength(1)
    expect(math[0]).toBe('$$a_1 \\times b_2$$')
    expect(text).not.toContain('a_1')
    expect(text).toContain('\uE000M0\uE000')
  })

  it('下划线/反斜杠/& 在数学段内原样保留，不被 Markdown 侵蚀', () => {
    const { math } = extractMath('$$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$')
    expect(math[0]).toContain('a & b')
    expect(math[0]).toContain('\\begin{pmatrix}')
    expect(math[0]).not.toContain('<em>')
  })

  it('行内 $..$ 与块级 $$..$$ 均被抽出，占位符与数组一一对应', () => {
    const { text, math } = extractMath('$\\alpha$ 与 $$\\beta_1$$ 与 $\\gamma$')
    expect(math).toHaveLength(3)
    // 块级先于行内被扫描（两趟正则），故数组非文档序；但占位符编号与数组严格对应
    expect(new Set(math)).toEqual(new Set(['$\\alpha$', '$$\\beta_1$$', '$\\gamma$']))
    for (let i = 0; i < math.length; i++) {
      expect(text).toContain(`\uE000M${i}\uE000`)
    }
    expect(text).not.toContain('$')
  })

  it('代码块内的 $ 不当作数学（先保护代码再抽数学），代码原样回填', () => {
    const src = '```python\nx = f"$not math$"\n```\n真正的 $\\theta$ 公式'
    const { text, math } = extractMath(src)
    expect(math).toHaveLength(1)
    expect(math[0]).toBe('$\\theta$')
    expect(text).toContain('```python')
    expect(text).toContain('$not math$')
  })

  it('行内码 `..` 受保护，不被当数学或强调', () => {
    const { math } = extractMath('变量 `$x$` 不是公式')
    expect(math).toHaveLength(0)
  })
})

describe('escapeHtml', () => {
  it('转义 & < > 供 HTML 文本节点回填', () => {
    expect(escapeHtml('a < b & c > d')).toBe('a &lt; b &amp; c &gt; d')
  })
})
