/**
 * 零依赖彩花（confetti）特效：提交 Accepted 时喷放。
 * 全屏 fixed canvas，动画结束自动移除，不阻塞交互。
 */

const COLORS = ['#2dd4a7', '#e6c260', '#6aa9f1', '#f16a6a', '#a78bfa', '#e8eef5']

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  rot: number
  vr: number
  size: number
  color: string
  round: boolean
  life: number
}

interface Wave {
  at: number // 触发延迟（ms）
  x: () => number
  y: () => number
  angle: number // 中心发射角（弧度，-PI/2 为正上）
  spread: number // 扇形半角
  speed: [number, number]
  count: number
}

export function burstConfetti(): void {
  const canvas = document.createElement('canvas')
  canvas.style.cssText =
    'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:9999'
  const dpr = window.devicePixelRatio || 1
  canvas.width = window.innerWidth * dpr
  canvas.height = window.innerHeight * dpr
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.scale(dpr, dpr)
  document.body.appendChild(canvas)

  const waves: Wave[] = [
    { at: 0, x: () => 0, y: () => window.innerHeight, angle: -Math.PI / 3.2, spread: 0.45, speed: [14, 24], count: 80 },
    { at: 140, x: () => window.innerWidth, y: () => window.innerHeight, angle: -Math.PI + Math.PI / 3.2, spread: 0.45, speed: [14, 24], count: 80 },
    { at: 300, x: () => window.innerWidth / 2, y: () => window.innerHeight * 0.42, angle: 0, spread: Math.PI, speed: [5, 13], count: 110 },
  ]

  const rnd = (a: number, b: number) => a + Math.random() * (b - a)
  const parts: Particle[] = []

  function spawn(w: Wave) {
    for (let i = 0; i < w.count; i++) {
      const angle = w.angle + (Math.random() - 0.5) * 2 * w.spread
      const v = rnd(w.speed[0], w.speed[1])
      parts.push({
        x: w.x(),
        y: w.y(),
        vx: Math.cos(angle) * v,
        vy: Math.sin(angle) * v,
        rot: rnd(0, Math.PI * 2),
        vr: rnd(-0.35, 0.35),
        size: rnd(4, 9),
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        round: Math.random() < 0.3,
        life: rnd(1.1, 2.0),
      })
    }
  }

  const t0 = performance.now()
  const fired = new Set<number>()

  function frame(now: number) {
    const elapsed = now - t0
    waves.forEach((w, i) => {
      if (!fired.has(i) && elapsed >= w.at) {
        fired.add(i)
        spawn(w)
      }
    })

    ctx!.clearRect(0, 0, window.innerWidth, window.innerHeight)
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i]
      p.vy += 0.38 // 重力
      p.vx *= 0.99
      p.x += p.vx
      p.y += p.vy
      p.rot += p.vr
      p.life -= 0.016
      if (p.life <= 0 || p.y > window.innerHeight + 30) {
        parts.splice(i, 1)
        continue
      }
      ctx!.save()
      ctx!.globalAlpha = Math.min(1, p.life * 1.6)
      ctx!.translate(p.x, p.y)
      ctx!.rotate(p.rot)
      ctx!.fillStyle = p.color
      if (p.round) {
        ctx!.beginPath()
        ctx!.arc(0, 0, p.size / 2, 0, Math.PI * 2)
        ctx!.fill()
      } else {
        ctx!.fillRect(-p.size / 2, -p.size / 3, p.size, p.size * 0.66)
      }
      ctx!.restore()
    }

    if (elapsed < 3200 || parts.length > 0) {
      requestAnimationFrame(frame)
    } else {
      canvas.remove()
    }
  }

  requestAnimationFrame(frame)
}
