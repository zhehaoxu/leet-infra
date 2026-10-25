<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'

const emit = defineEmits<{ (e: 'close'): void }>()

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="overlay" @click.self="emit('close')">
    <div class="modal" role="dialog" aria-label="帮助">
      <button class="close" title="关闭" @click="emit('close')">✕</button>
      <h2>使用帮助</h2>

      <h3>快捷键</h3>
      <ul>
        <li><kbd>⌘/Ctrl + Enter</kbd> 运行：只跑样例用例</li>
        <li><kbd>⌘/Ctrl + Shift + Enter</kbd> 提交：全量评测</li>
        <li><kbd>Esc</kbd> 关闭本弹窗</li>
      </ul>

      <h3>运行 vs 提交</h3>
      <ul>
        <li>「运行」只执行题目标注的样例用例，快速验证思路，<b>不记录进度</b></li>
        <li>「提交」执行全部测试用例，全部通过才记为已解决并解锁参考答案</li>
      </ul>

      <h3>评测环境</h3>
      <ul>
        <li>代码在你本机独立的 Python 虚拟环境（.venv-eval）中以子进程真实运行</li>
        <li>预装 <code>torch</code> / <code>numpy</code>；<code>llm-models</code> 合集的参考解还可用 <code>transformers</code>（离线）作为大模型组件的对拍 oracle</li>
        <li>评测前固定随机种子 <code>torch.manual_seed(42)</code>：请保证解法可复现，不要依赖外部随机状态</li>
        <li>请在函数内部创建新张量完成计算；单次评测默认 30 秒超时（JUDGE_TIMEOUT_S 可调）</li>
      </ul>

      <h3>扩充题库 / 合集</h3>
      <ul>
        <li>题库：<code>problems/pytorch/&lt;题目slug&gt;/</code>（meta.json、description.md、starter.py、test_cases.py、solution.py），热加载，免重启</li>
        <li>合集：<code>problems/collections.json</code> 配置名称、描述与有序题目列表</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(5, 8, 12, 0.62);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}
.modal {
  position: relative;
  width: min(560px, 92vw);
  max-height: 82vh;
  overflow-y: auto;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 26px 30px;
}
.close {
  position: absolute;
  top: 14px;
  right: 14px;
  border: none;
  background: none;
  color: var(--text-dim);
  font-size: 14px;
  cursor: pointer;
}
.close:hover {
  color: var(--text);
}
h2 {
  margin: 0 0 14px;
  font-size: 19px;
}
h3 {
  margin: 18px 0 8px;
  font-size: 14px;
  color: var(--accent);
}
ul {
  margin: 0;
  padding-left: 18px;
  color: var(--text-dim);
  font-size: 13.5px;
  line-height: 1.9;
}
kbd,
code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1px 6px;
}

@media (max-width: 640px) {
  .overlay {
    padding: 12px 0;
    align-items: flex-start;
  }
  .modal {
    width: 94vw;
    max-height: 88vh;
    border-radius: 12px;
    padding: 22px 18px;
  }
  ul {
    font-size: 13px;
    line-height: 1.75;
  }
}
</style>
