import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'
import { cleanupOrphanStorage } from './utils/storageCleanup'
import './styles.css'

// 启动一次性清理历史孤儿键（如早期可编辑笔记遗留的 leet-infra:notes:*）
cleanupOrphanStorage()

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
