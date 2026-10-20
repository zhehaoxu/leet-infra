import { createRouter, createWebHashHistory } from 'vue-router'
import ProblemList from './views/ProblemList.vue'
import ProblemWorkbench from './views/ProblemWorkbench.vue'

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'list', component: ProblemList },
    { path: '/problems/:slug', name: 'workbench', component: ProblemWorkbench, props: true },
  ],
})
