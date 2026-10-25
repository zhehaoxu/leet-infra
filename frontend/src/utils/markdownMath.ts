// Markdown 数学保护：把代码块与 $..$/$$..$$ 数学段抽出为占位符，
// 交 marked 处理正文后再原样回填，避免 LaTeX 里的 _ & \\ 等被 Markdown 当强调/实体破坏，
// 致使 KaTeX auto-render 匹配不到定界符（rope 题面块级公式曾因此失效）。

/** 抽取代码块与数学段，返回把数学替换为占位符的正文 + 被保护的数学原文列表 */
export function extractMath(src: string): { text: string; math: string[] } {
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

/** 按 HTML 文本节点转义（marked 未见过数学内容，回填时特殊字符需转义后原样插入） */
export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
