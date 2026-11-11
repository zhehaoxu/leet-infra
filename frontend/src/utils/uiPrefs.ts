// 界面偏好持久化（localStorage）纯读写helper（ProblemList.vue 消费，单测覆盖）。

export const HOME_TAB_KEY = 'leet-infra:home-tab'
export type HomeTab = 'all' | 'collections' | 'projects'

const VALID_TABS: HomeTab[] = ['all', 'collections', 'projects']

/** 读取首页标签；只认白名单内的值，其余（含缺失/脏值）回落 'all' */
export function readHomeTab(store: Pick<Storage, 'getItem'> = localStorage): HomeTab {
  const value = store.getItem(HOME_TAB_KEY) as HomeTab | null
  return value && VALID_TABS.includes(value) ? value : 'all'
}

export function writeHomeTab(tab: HomeTab, store: Pick<Storage, 'setItem'> = localStorage): void {
  store.setItem(HOME_TAB_KEY, tab)
}
