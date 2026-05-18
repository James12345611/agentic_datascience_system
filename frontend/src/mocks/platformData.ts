import type {
  DashboardSnapshot,
  MonitoringBundle,
  TaskCenterSnapshot,
  WorkbenchSnapshot,
} from '../types/platform'

const tasks = [
  {
    id: 'TASK-20260518-014',
    name: '长三角碳排放面板预处理',
    type: 'preprocess',
    dataset: '长三角工业能耗季度数据',
    datasetId: 'mock-dataset-001',
    status: 'running',
    owner: '系统代理',
    startedAt: '2026-05-18 08:12',
    duration: '04:18',
    stage: '字段约束校验',
    records: '128,430',
    version: 'v0.9.3',
    alerts: 3,
  },
  {
    id: 'TASK-20260518-013',
    name: '区域经济韧性回归建模',
    type: 'modeling',
    dataset: '城市群经济韧性年度数据',
    datasetId: 'mock-dataset-002',
    status: 'completed',
    owner: '系统代理',
    startedAt: '2026-05-18 07:41',
    duration: '11:06',
    stage: '结果归档完成',
    records: '8,420',
    version: 'v0.9.3',
    alerts: 1,
  },
  {
    id: 'TASK-20260518-011',
    name: '产业结构分组统计诊断',
    type: 'diagnostic',
    dataset: '省际产业升级面板',
    datasetId: 'mock-dataset-003',
    status: 'failed',
    owner: '系统代理',
    startedAt: '2026-05-18 06:53',
    duration: '02:57',
    stage: '高基数字段清洗',
    records: '62,104',
    version: 'v0.9.2',
    alerts: 5,
  },
  {
    id: 'TASK-20260517-118',
    name: '绿色金融指数随机森林建模',
    type: 'modeling',
    dataset: '绿色金融季度指标',
    datasetId: 'mock-dataset-004',
    status: 'queued',
    owner: '系统代理',
    startedAt: '2026-05-17 23:11',
    duration: '待执行',
    stage: '队列等待',
    records: '10,256',
    version: 'v0.9.2',
    alerts: 0,
  },
] as const

export const dashboardSnapshot: DashboardSnapshot = {
  headline: '把预处理、诊断、建模和人工修正放进同一条中文数据工作流',
  subtitle:
    '这一版 React 前端先聚焦正式产品壳与监控交互。当前数据来自前端演示层，后续会通过 FastAPI 接入真实任务、日志、字段字典和模型结果。',
  heroStats: [
    { label: '原始数据集', value: '12' },
    { label: '预处理版本', value: '17' },
    { label: '任务快照数', value: '29' },
  ],
  summaryCards: [
    { title: '今日任务总量', value: '28', delta: '较昨日 +6', note: '覆盖预处理、统计诊断和建模任务', tone: 'teal' },
    { title: '运行成功率', value: '89.7%', delta: '近 7 日稳定上升', note: '失败任务主要集中在高基数分类变量处理', tone: 'amber' },
    { title: '字段人工修正数', value: '43', delta: '本周已累计 126 次', note: '类型纠偏、单位标注、范围约束最常见', tone: 'copper' },
    { title: '可复用配置版本', value: '17', delta: '新增 4 版', note: '后续可直接绑定数据集模板和代理策略', tone: 'slate' },
  ],
  trend: [
    { label: '05-12', total: 11, preprocess: 6, modeling: 3 },
    { label: '05-13', total: 14, preprocess: 7, modeling: 4 },
    { label: '05-14', total: 17, preprocess: 8, modeling: 6 },
    { label: '05-15', total: 15, preprocess: 8, modeling: 3 },
    { label: '05-16', total: 20, preprocess: 10, modeling: 6 },
    { label: '05-17', total: 22, preprocess: 12, modeling: 7 },
    { label: '05-18', total: 28, preprocess: 15, modeling: 8 },
  ],
  ranking: [
    { label: 'OLS 回归', score: 0.82 },
    { label: '随机森林回归', score: 0.87 },
    { label: '梯度提升树', score: 0.84 },
    { label: 'Logistic 回归', score: 0.79 },
  ],
  alerts: [
    { title: '城市群名称字段被系统识别为普通分类', detail: '建议切换到地理区域语义，并补充行政层级标注。', severity: '高' },
    { title: '排放量占比字段缺少比例量纲说明', detail: '后续缩放与异常值裁剪阶段容易误判。', severity: '高' },
    { title: '时间字段粒度混合了年和季度', detail: '建议在导入阶段强制选择时间粒度模板。', severity: '中' },
  ],
  tasks: tasks.slice(0, 3),
}

export const taskCenterSnapshot: TaskCenterSnapshot = {
  tasks: [...tasks],
  stageStats: [
    { label: '字段识别', value: 8 },
    { label: '质量校验', value: 6 },
    { label: '异常值处理', value: 5 },
    { label: '建模训练', value: 7 },
    { label: '结果归档', value: 2 },
  ],
  statusStats: [
    { label: '运行中', value: 1 },
    { label: '成功', value: 1 },
    { label: '失败', value: 1 },
    { label: '排队中', value: 1 },
  ],
}

export const monitoringBundles: Record<string, MonitoringBundle> = {
  'TASK-20260518-014': {
    task: tasks[0],
    timeline: [
      { name: '载入数据集', status: 'done', duration: '00:12', detail: '成功读取 128,430 行，字段数 42。' },
      { name: '自动识别字段类型', status: 'done', duration: '00:39', detail: '识别出 9 个需人工确认字段。' },
      { name: '字段约束校验', status: 'working', duration: '01:08', detail: '正在校验比例字段、单位字段与时间字段粒度。' },
      { name: '异常值处理', status: 'waiting', duration: '--', detail: '等待字段元数据确认后继续执行。' },
      { name: '结果落库', status: 'waiting', duration: '--', detail: '待处理完成后同步写入原型数据库。' },
    ],
    logs: [
      { id: 'L-1', time: '08:12:11', level: '信息', module: 'reader', message: 'CSV 文件读取完成，检测到 UTF-8 with BOM 编码。' },
      { id: 'L-2', time: '08:12:44', level: '警告', module: 'type_infer', message: '字段“城市群名称”当前被识别为普通分类，建议人工确认其语义层级。', field: '城市群名称' },
      { id: 'L-3', time: '08:13:02', level: '警告', module: 'validator', message: '字段“排放量占比”缺少比例量纲，后续缩放策略可能失真。', field: '排放量占比' },
      { id: 'L-4', time: '08:13:46', level: '信息', module: 'validator', message: '字段“统计年份”已识别为时间类型，准备检查粒度和取值范围。', field: '统计年份' },
      { id: 'L-5', time: '08:14:09', level: '错误', module: 'validator', message: '字段“单位GDP能耗”存在 3 条超出软范围的记录，建议人工复核后决定是否裁剪。', field: '单位GDP能耗' },
    ],
    tuningFields: [
      { id: 'F-1', fieldName: '城市群名称', dataType: '分类', semanticType: '地理区域', unit: '无', softRange: '不适用', note: '建议补充区域层级，例如国家级城市群/省内城市圈。' },
      { id: 'F-2', fieldName: '排放量占比', dataType: '数值', semanticType: '比例', unit: '%', softRange: '0 ~ 100', note: '后续可按 0-100 或 0-1 两种标尺统一换算。' },
      { id: 'F-3', fieldName: '统计年份', dataType: '时间', semanticType: '年度索引', unit: '年', softRange: '2010 ~ 2024', note: '建议锁定年度粒度，避免季度混入。' },
      { id: 'F-4', fieldName: '单位GDP能耗', dataType: '数值', semanticType: '强度指标', unit: '吨标准煤/万元', softRange: '0.1 ~ 3.0', note: '建议结合行业口径后再做异常值截断。' },
    ],
    revisions: [
      { id: 'R-1', time: '08:10', actor: '人工标注', fieldName: '排放量占比', summary: '将原始数值字段补充为比例字段，并指定单位为 %。' },
      { id: 'R-2', time: '08:08', actor: '系统建议', fieldName: '城市群名称', summary: '建议改为地理区域语义，并补充层级标签。' },
    ],
  },
  'TASK-20260518-013': {
    task: tasks[1],
    timeline: [
      { name: '准备特征矩阵', status: 'done', duration: '01:42', detail: '完成 12 个字段清理与哑变量展开。' },
      { name: '训练 OLS 回归', status: 'done', duration: '02:18', detail: '模型拟合完成，R² 达到 0.82。' },
      { name: '生成系数解释', status: 'done', duration: '03:01', detail: '输出显著性、置信区间与敏感字段提示。' },
      { name: '结果归档', status: 'done', duration: '04:05', detail: '任务产物已落库，可进入结果对比页。' },
    ],
    logs: [
      { id: 'M-1', time: '07:42:16', level: '信息', module: 'modeling', message: 'OLS 模型启动，目标变量为“区域经济韧性指数”。' },
      { id: 'M-2', time: '07:44:08', level: '警告', module: 'modeling', message: '字段“数字基础设施指数”与“创新强度”存在中度共线性。', field: '数字基础设施指数' },
      { id: 'M-3', time: '07:47:55', level: '信息', module: 'reporter', message: '模型摘要已归档，任务可进入结果解释页面。' },
    ],
    tuningFields: [
      { id: 'MF-1', fieldName: '数字基础设施指数', dataType: '数值', semanticType: '指数', unit: '分', softRange: '0 ~ 100', note: '可考虑在下一轮模型中做标准化对比。' },
    ],
    revisions: [
      { id: 'MR-1', time: '07:40', actor: '人工标注', fieldName: '区域经济韧性指数', summary: '确认该字段为连续型目标变量，采用 OLS 回归。' },
    ],
  },
}

export const workbenchSnapshot: WorkbenchSnapshot = {
  modules: [
    { label: '数据接入层', status: '已可演示', detail: '上传、历史回读、字段基础字典已形成原型。' },
    { label: '预处理层', status: '已可演示', detail: '字段纠偏、缺失值、异常值、编码、缩放已打通。' },
    { label: '统计分析层', status: '已可演示', detail: 'EDA、统计检验、诊断与自动洞察已可用。' },
    { label: '建模层', status: '基础可用', detail: 'OLS、Logistic、随机森林、梯度提升已接入。' },
    { label: 'API 接口层', status: '已接入核心任务链路', detail: '上传数据集、预处理启动、建模启动和监控查询已打通。' },
  ],
  datasets: [
    { label: '原始数据记录', value: 12 },
    { label: '预处理结果版本', value: 17 },
    { label: '任务配置快照', value: 29 },
  ],
  nextActions: [
    '把字段级预处理细配也迁移到 React 工作台，而不只留在监控页。',
    '将 SQLite 原型逐步迁移到 PostgreSQL，并保留任务快照表。',
    '继续补异步任务、结果页跳转和多用户审计链路。',
  ],
  apiContracts: [
    { name: '/api/datasets', method: 'GET', status: '已接入' },
    { name: '/api/datasets/upload', method: 'POST', status: '已接入' },
    { name: '/api/datasets/{dataset_id}/preview', method: 'GET', status: '已接入' },
    { name: '/api/preprocess/run', method: 'POST', status: '已接入' },
    { name: '/api/modeling/run', method: 'POST', status: '已接入' },
  ],
}
