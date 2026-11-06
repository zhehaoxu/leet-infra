import { createRouter, createWebHashHistory } from 'vue-router'
import ProblemList from './views/ProblemList.vue'
import ProblemWorkbench from './views/ProblemWorkbench.vue'
import CollectionDetail from './views/CollectionDetail.vue'
import ProjectDetail from './views/ProjectDetail.vue'
import ProjectStepView from './views/ProjectStepView.vue'

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'list', component: ProblemList },
    { path: '/collections/:slug', name: 'collection', component: CollectionDetail, props: true },
    { path: '/problems/:slug', name: 'workbench', component: ProblemWorkbench, props: true },
    { path: '/projects/:slug', name: 'project', component: ProjectDetail, props: true },
    {
      path: '/projects/:slug/steps/:step',
      name: 'project-step',
      component: ProjectStepView,
      props: (route) => ({ slug: String(route.params.slug), step: Number(route.params.step) }),
    },
  ],
})
