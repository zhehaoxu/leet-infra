import { computed, ref } from 'vue'

export type Locale = 'zh' | 'en'

const STORAGE_KEY = 'leet-infra:locale'

const zh = {
  'app.title': 'leet-infra · AI Infra 刷题',
  'app.progress': '{solved} / {total} 已解决',
  'app.env': 'PyTorch · 本地 CPU',
  'app.helpTitle': '帮助（快捷键与评测规则）',
  'app.langTitle': '切换到英文',

  'common.loading': '加载中…',
  'common.loadFailed': '加载失败：{error}',
  'common.emptyProblems': '暂无题目',
  'common.noMatches': '没有匹配的题目',
  'common.solvedOf': '{solved}/{total} 已解决',

  'home.heroTitle': 'AI Infra 从动手写开始',
  'home.heroSubtitle':
    'PyTorch 题库 · 代码在你本机 Python 虚拟环境中真实运行 · {solved}/{total} 已解决',
  'home.reset': '重置进度',
  'home.resetTitle': '清空全部已解决记录，用于二刷题目',
  'home.export': '导出进度',
  'home.exportTitle': '下载进度文件 progress.json',
  'home.import': '导入进度',
  'home.importTitle': '从导出的 JSON 恢复进度（覆盖当前）',
  'home.tabAll': '全部',
  'home.tabCollections': '合集',
  'home.searchCollections': '搜索合集…',
  'home.noCollections': '没有匹配的合集',
  'home.enterCollection': '进入合集 →',
  'home.resetConfirm': '清空全部解题进度？题目代码草稿不受影响。此操作不可撤销。',
  'home.resetFailed': '重置失败：{error}',
  'home.exportFailed': '导出失败：{error}',
  'home.importConfirm': '导入将覆盖当前全部解题进度，继续？',
  'home.importDone': '导入完成：恢复 {imported} 题',
  'home.importSkipped': '，跳过不认识的条目 {skipped} 个',
  'home.importFailed': '导入失败：{error}，请确认选择的是 leet-infra 导出的进度文件',

  'collection.back': '← 返回题库',
  'collection.progress': '{solved}/{total} 已解决',
  'collection.implementation': '实现方式',
  'collection.variantHint': '题目入口将沿用该选择；尚未支持的实现方式在做题时自动回退',
  'collection.empty': '该合集暂无题目',

  'table.search': '搜索题目…',
  'table.allDifficulties': '全部难度',
  'table.allStatuses': '全部状态',
  'table.solved': '已解决',
  'table.unsolved': '未解决',
  'table.group': '分组',
  'table.groupTitle': '按 family 分组折叠',
  'table.status': '状态',
  'table.title': '标题',
  'table.difficulty': '难度',
  'table.tags': '标签',
  'table.solve': '解答 →',
  'table.collectionTitle': '合集：{name}',

  'difficulty.easy': '简单',
  'difficulty.medium': '中等',
  'difficulty.hard': '困难',

  'workbench.back': '← 题库',
  'workbench.solved': '✓ 已解决',
  'workbench.draftRestored': '草稿已恢复',
  'workbench.draftRestoredTitle': '已从 localStorage 恢复上次编辑的草稿',
  'workbench.implementation': '实现方式',
  'workbench.comingSoon': '（即将支持）',
  'workbench.variantFallback': '{variant} 暂未支持，当前以 {active} 评测',
  'workbench.prev': '← 上一题',
  'workbench.next': '下一题 →',
  'workbench.nextUnsolved': '下一道未解 →',
  'workbench.goTo': '去做：{title}',
  'workbench.redo': '重新做',
  'workbench.redoTitle': '清除本题已解记录与草稿，重新做一遍',
  'workbench.redoConfirm': '重新做本题？将清除本题已解记录与代码草稿（笔记保留）。',
  'workbench.reset': '重置',
  'workbench.run': '运行',
  'workbench.cancel': '取消',
  'workbench.submit': '提交',
  'workbench.submitting': '提交中…',
  'workbench.running': '运行中…',
  'workbench.tabDescription': '题目说明',
  'workbench.tabNotes': '笔记',
  'workbench.tabSolution': '参考答案',
  'workbench.notesTitle': '题目算法详解',
  'workbench.notesDisabled': '本题无附加算法笔记',
  'workbench.solutionUnlocked': '查看参考答案',
  'workbench.solutionLocked': '提交通过后解锁参考答案',
  'workbench.copyCode': '复制代码',
  'workbench.copied': '已复制',
  'workbench.loadingProblem': '加载题目中…',

  'result.running': '正在本地评测环境运行…（子进程执行 PyTorch 代码）',
  'result.requestFailed': '请求失败：{error}',
  'result.idle':
    '尚未运行。「运行」只跑样例用例（⌘/Ctrl+Enter），「提交」全量评测并记录进度（⌘/Ctrl+Shift+Enter）。',
  'result.acceptedRun': '样例通过 · 可提交全量评测',
  'result.acceptedSubmit': 'Accepted · 全部通过',
  'result.wrongRun': '样例未通过',
  'result.wrongSubmit': 'Wrong Answer',
  'result.timeout': 'Time Limit Exceeded',
  'result.syntax': 'Syntax Error',
  'result.noSolve': '未实现 solve()',
  'result.cancelled': '已取消',
  'result.envMissing': '评测环境缺失',
  'result.sampleMetric': '{passed}/{total} 样例用例通过',
  'result.caseMetric': '{passed}/{total} 用例通过',
  'result.totalTime': '总耗时 {ms} ms',
  'result.passedToggle': '{count} 个用例通过',
  'result.bench': '参考 {ms}ms · {ratio}×',
  'result.benchTitle': '参考解在本机的实测耗时，仅作对比',

  'help.title': '使用帮助',
  'help.close': '关闭',
  'help.shortcuts': '快捷键',
  'help.shortcutRun': '运行：只跑样例用例',
  'help.shortcutSubmit': '提交：全量评测',
  'help.shortcutClose': '关闭本弹窗',
  'help.runVsSubmit': '运行 vs 提交',
  'help.runRule': '「运行」只执行题目标注的样例用例，快速验证思路，不记录进度',
  'help.submitRule': '「提交」执行全部测试用例，全部通过才记为已解决并解锁参考答案',
  'help.env': '评测环境',
  'help.envLocal': '代码在你本机独立的 Python 虚拟环境（.venv-eval）中以子进程真实运行',
  'help.envPackages':
    '预装 torch / numpy；llm-models 合集的参考解还可用 transformers（离线）作为大模型组件的对拍 oracle',
  'help.envSeed': '评测前固定随机种子 torch.manual_seed(42)：请保证解法可复现，不要依赖外部随机状态',
  'help.envTensors': '请在函数内部创建新张量完成计算；单次评测默认 30 秒超时（JUDGE_TIMEOUT_S 可调）',
  'help.extending': '扩充题库 / 合集',
  'help.extendProblems':
    '题库：problems/pytorch/<题目slug>/（meta.json、description.md、starter.py、test_cases.py、solution.py），热加载，免重启',
  'help.extendCollections': '合集：problems/collections.json 配置名称、描述与有序题目列表',

  'family.activation': '激活函数',
  'family.normalization': '归一化',
  'family.conv': '卷积',
  'family.vision': '池化 · 视觉',
  'family.gemm': '矩阵乘 · 线代',
  'family.attention': '注意力',
  'family.tensor': '张量操作',
  'family.loss': '损失',
  'family.optimizer': '优化器',
  'family.quantization': '量化',
  'family.sampling': '采样',
  'family.rnn': '循环网络',
  'family.metric': '检测 · 度量',
  'family.systems': '显存 · 性能',
  'family.tokenization': '分词 · 数据',
  'family.distributed': '分布式',
  'family.llm': '大模型结构',
  'family.moe': 'MoE 路由',
  'family.inference': '推理 Serving',
  'family.alignment': '对齐目标',
  'family.kvcache': 'KV Cache 压缩',
  'family.speculative': '推测解码',
  'family.peft': '参数高效微调',
  'family.diffusion': '扩散生成',
  'family.sparsity': '稀疏剪枝',
  'family.multimodal': '多模态',
  'family.audio': '语音前端',
  'family.retrieval': '向量检索',
  'family.evaluation': '评测指标',
  'family.rl': 'RL 基建',
  'family.agentic': '智能体系统',
  'family.other': '框架 / 其他',
} as const

export type MessageKey = keyof typeof zh

const en: Record<MessageKey, string> = {
  'app.title': 'leet-infra · AI Infra Practice',
  'app.progress': '{solved} / {total} solved',
  'app.env': 'PyTorch · Local CPU',
  'app.helpTitle': 'Help (shortcuts & judging rules)',
  'app.langTitle': 'Switch to Chinese',

  'common.loading': 'Loading…',
  'common.loadFailed': 'Failed to load: {error}',
  'common.emptyProblems': 'No problems yet',
  'common.noMatches': 'No matching problems',
  'common.solvedOf': '{solved}/{total} solved',

  'home.heroTitle': 'Learn AI Infra by Building It',
  'home.heroSubtitle':
    'PyTorch problem bank · Code runs locally in your Python environment · {solved}/{total} solved',
  'home.reset': 'Reset progress',
  'home.resetTitle': 'Clear all solved records to start a second pass',
  'home.export': 'Export progress',
  'home.exportTitle': 'Download the progress file',
  'home.import': 'Import progress',
  'home.importTitle': 'Restore progress from an exported JSON file (overwrites current)',
  'home.tabAll': 'All',
  'home.tabCollections': 'Collections',
  'home.searchCollections': 'Search collections…',
  'home.noCollections': 'No matching collections',
  'home.enterCollection': 'Open collection →',
  'home.resetConfirm':
    'Clear all solved progress? Code drafts are kept. This action cannot be undone.',
  'home.resetFailed': 'Reset failed: {error}',
  'home.exportFailed': 'Export failed: {error}',
  'home.importConfirm': 'Importing will overwrite all current progress. Continue?',
  'home.importDone': 'Import complete: restored {imported} problems',
  'home.importSkipped': ', skipped {skipped} unknown entries',
  'home.importFailed':
    'Import failed: {error}. Make sure you selected a progress file exported by leet-infra.',

  'collection.back': '← Back to problems',
  'collection.progress': '{solved}/{total} solved',
  'collection.implementation': 'Implementation',
  'collection.variantHint':
    'Problem links keep this choice; unsupported implementations fall back automatically.',
  'collection.empty': 'No problems in this collection yet',

  'table.search': 'Search problems…',
  'table.allDifficulties': 'All difficulties',
  'table.allStatuses': 'All statuses',
  'table.solved': 'Solved',
  'table.unsolved': 'Unsolved',
  'table.group': 'Group',
  'table.groupTitle': 'Group and collapse by family',
  'table.status': 'Status',
  'table.title': 'Title',
  'table.difficulty': 'Difficulty',
  'table.tags': 'Tags',
  'table.solve': 'Solve →',
  'table.collectionTitle': 'Collection: {name}',

  'difficulty.easy': 'Easy',
  'difficulty.medium': 'Medium',
  'difficulty.hard': 'Hard',

  'workbench.back': '← Problems',
  'workbench.solved': '✓ Solved',
  'workbench.draftRestored': 'Draft restored',
  'workbench.draftRestoredTitle': 'Restored your last draft from localStorage',
  'workbench.implementation': 'Implementation',
  'workbench.comingSoon': ' (coming soon)',
  'workbench.variantFallback': '{variant} is not supported yet; judging with {active}',
  'workbench.prev': '← Previous',
  'workbench.next': 'Next →',
  'workbench.nextUnsolved': 'Next unsolved →',
  'workbench.goTo': 'Go to: {title}',
  'workbench.redo': 'Redo',
  'workbench.redoTitle': "Clear this problem's solved record and draft to start over",
  'workbench.redoConfirm':
    "Redo this problem? Its solved record and code draft will be cleared (notes are kept).",
  'workbench.reset': 'Reset',
  'workbench.run': 'Run',
  'workbench.cancel': 'Cancel',
  'workbench.submit': 'Submit',
  'workbench.submitting': 'Submitting…',
  'workbench.running': 'Running…',
  'workbench.tabDescription': 'Description',
  'workbench.tabNotes': 'Notes',
  'workbench.tabSolution': 'Solution',
  'workbench.notesTitle': 'Algorithm notes',
  'workbench.notesDisabled': 'No notes for this problem',
  'workbench.solutionUnlocked': 'View the reference solution',
  'workbench.solutionLocked': 'Pass all tests to unlock the reference solution',
  'workbench.copyCode': 'Copy code',
  'workbench.copied': 'Copied',
  'workbench.loadingProblem': 'Loading problem…',

  'result.running': 'Running in the local judge… (PyTorch code in a subprocess)',
  'result.requestFailed': 'Request failed: {error}',
  'result.idle':
    'Not run yet. Run executes sample cases (⌘/Ctrl+Enter); Submit runs all cases and records progress (⌘/Ctrl+Shift+Enter).',
  'result.acceptedRun': 'Samples passed · ready to submit',
  'result.acceptedSubmit': 'Accepted · all tests passed',
  'result.wrongRun': 'Samples failed',
  'result.wrongSubmit': 'Wrong Answer',
  'result.timeout': 'Time Limit Exceeded',
  'result.syntax': 'Syntax Error',
  'result.noSolve': 'solve() not implemented',
  'result.cancelled': 'Cancelled',
  'result.envMissing': 'Judge environment missing',
  'result.sampleMetric': '{passed}/{total} sample cases passed',
  'result.caseMetric': '{passed}/{total} cases passed',
  'result.totalTime': 'Total {ms} ms',
  'result.passedToggle': '{count} cases passed',
  'result.bench': 'ref {ms}ms · {ratio}×',
  'result.benchTitle': 'Reference solution timing on this machine, for comparison only',

  'help.title': 'Help',
  'help.close': 'Close',
  'help.shortcuts': 'Keyboard shortcuts',
  'help.shortcutRun': 'Run: sample cases only',
  'help.shortcutSubmit': 'Submit: full evaluation',
  'help.shortcutClose': 'Close this dialog',
  'help.runVsSubmit': 'Run vs Submit',
  'help.runRule':
    'Run executes only the sample cases marked in the problem for a quick check; progress is not recorded.',
  'help.submitRule':
    'Submit executes all test cases. Only a full pass marks the problem solved and unlocks the reference solution.',
  'help.env': 'Judge environment',
  'help.envLocal':
    'Your code runs for real in a subprocess inside a local Python virtualenv (.venv-eval).',
  'help.envPackages':
    'torch and numpy are preinstalled; solutions in the llm-models collection can also use transformers (offline) as an oracle for LLM components.',
  'help.envSeed':
    'The seed is fixed before judging (torch.manual_seed(42)): keep solutions reproducible and do not depend on external random state.',
  'help.envTensors':
    'Create new tensors inside the function to compute the result. Each run has a 30-second timeout by default (configurable via JUDGE_TIMEOUT_S).',
  'help.extending': 'Adding problems / collections',
  'help.extendProblems':
    'Problems: problems/pytorch/<slug>/ (meta.json, description.md, starter.py, test_cases.py, solution.py), hot reloaded without restarting.',
  'help.extendCollections':
    'Collections: problems/collections.json defines the name, description, and ordered problem list.',

  'family.activation': 'Activations',
  'family.normalization': 'Normalization',
  'family.conv': 'Convolution',
  'family.vision': 'Pooling · Vision',
  'family.gemm': 'GEMM · Linear Algebra',
  'family.attention': 'Attention',
  'family.tensor': 'Tensor Ops',
  'family.loss': 'Losses',
  'family.optimizer': 'Optimizers',
  'family.quantization': 'Quantization',
  'family.sampling': 'Sampling',
  'family.rnn': 'Recurrent Networks',
  'family.metric': 'Detection · Metrics',
  'family.systems': 'Memory · Performance',
  'family.tokenization': 'Tokenization · Data',
  'family.distributed': 'Distributed',
  'family.llm': 'LLM Architecture',
  'family.moe': 'MoE Routing',
  'family.inference': 'Inference Serving',
  'family.alignment': 'Alignment',
  'family.kvcache': 'KV Cache Compression',
  'family.speculative': 'Speculative Decoding',
  'family.peft': 'Parameter-Efficient Fine-Tuning',
  'family.diffusion': 'Diffusion',
  'family.sparsity': 'Sparsity · Pruning',
  'family.multimodal': 'Multimodal',
  'family.audio': 'Speech Front-End',
  'family.retrieval': 'Vector Retrieval',
  'family.evaluation': 'Evaluation Metrics',
  'family.rl': 'RL Systems',
  'family.agentic': 'Agentic Systems',
  'family.other': 'Framework / Other',
}

const messages: Record<Locale, Record<string, string>> = { zh, en }

function readStoredLocale(): Locale {
  const stored = localStorage.getItem(STORAGE_KEY)
  return stored === 'en' ? 'en' : 'zh'
}

export const locale = ref<Locale>(readStoredLocale())

function applyDocumentLocale() {
  document.documentElement.lang = locale.value === 'zh' ? 'zh-CN' : 'en'
  document.title = messages[locale.value]['app.title']
}

export function setLocale(next: Locale) {
  locale.value = next
  localStorage.setItem(STORAGE_KEY, next)
  applyDocumentLocale()
}

export function toggleLocale() {
  setLocale(locale.value === 'zh' ? 'en' : 'zh')
}

export const localeButtonLabel = computed(() => (locale.value === 'zh' ? 'EN' : '中文'))

applyDocumentLocale()

export function t(key: MessageKey, params?: Record<string, string | number>): string {
  const text = messages[locale.value][key] ?? zh[key]
  if (!params) return text
  return text.replace(/\{(\w+)\}/g, (match, name: string) =>
    name in params ? String(params[name]) : match
  )
}
