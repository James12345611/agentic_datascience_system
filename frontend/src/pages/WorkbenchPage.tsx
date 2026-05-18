import type { ChangeEvent } from 'react'
import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Panel } from '../components/Panel'
import { TaskStatusPill } from '../components/TaskStatusPill'
import {
  fetchDashboardSnapshot,
  fetchDatasetDictionary,
  fetchDatasetPreview,
  fetchHealthSnapshot,
  fetchMonitoringBundle,
  fetchTaskCenterSnapshot,
  fetchTaskDetail,
  fetchTaskList,
  fetchTaskLogs,
  fetchWorkbenchDatasets,
  fetchWorkbenchSnapshot,
  runModelingTask,
  runPreprocessTask,
  saveDatasetDictionaryField,
  uploadDatasetFile,
} from '../services/platformService'
import type {
  DashboardSnapshot,
  DatasetDictionaryBundle,
  DatasetPreview,
  DatasetUploadResult,
  HealthSnapshot,
  ModelingRunResult,
  MonitoringBundle,
  PreprocessTaskResult,
  TaskCenterSnapshot,
  TaskItem,
  TaskListResponse,
  TaskLogsResponse,
  TuningField,
  WorkbenchDatasetItem,
  WorkbenchSnapshot,
} from '../types/platform'

const missingStrategyOptions = [
  { label: '中位数填充', value: 'median' },
  { label: '均值填充', value: 'mean' },
  { label: '众数填充', value: 'mode' },
  { label: '固定值填充', value: 'constant' },
  { label: '不处理', value: 'none' },
]

const encodingOptions = [
  { label: '独热编码', value: 'onehot' },
  { label: '标签编码', value: 'label' },
  { label: '不编码', value: 'none' },
]

const scalingOptions = [
  { label: '不缩放', value: 'none' },
  { label: '标准化', value: 'standard' },
  { label: '归一化', value: 'minmax' },
]

const outlierOptions = [
  { label: '不处理', value: 'none' },
  { label: 'IQR 截断', value: 'clip_iqr' },
  { label: '分位数截断', value: 'clip_quantile' },
  { label: 'IQR 删除异常样本', value: 'drop_iqr' },
]

const modelFamilyOptions = [
  { label: '经典统计建模', value: 'statistical' },
  { label: '机器学习建模', value: 'machine_learning' },
]

const problemTypeOptions = [
  { label: '回归', value: 'regression' },
  { label: '二分类', value: 'binary_classification' },
  { label: '多分类', value: 'multiclass_classification' },
]

const statisticalAlgorithmOptions = [
  { label: '线性回归 OLS', value: 'ols' },
  { label: '二分类 Logistic 回归', value: 'logit' },
]

const machineLearningAlgorithmOptions = [
  { label: '随机森林', value: 'random_forest' },
  { label: '梯度提升树', value: 'gradient_boosting' },
]

const dataTypeOptions = ['数值', '分类', '时间', '布尔', '文本', '标识']
const semanticTypeOptions = [
  '普通字段',
  '比例',
  '金额',
  '计数',
  '评分',
  '指数',
  '时长',
  '普通分类',
  '有序分类',
  '地理区域',
  '行业分类',
  '行政区域',
  '时间索引',
  '布尔标签',
  '文本标签',
  '自定义',
]
const nonScalarTypes = new Set(['分类', '布尔', '文本', '标识'])

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return '—'
  }
  if (typeof value === 'object') {
    return JSON.stringify(value)
  }
  return String(value)
}

function formatTimestamp(value?: string | null): string {
  if (!value) {
    return '—'
  }
  return value.replace('T', ' ')
}

function buildDefaultTarget(columns: string[]): string {
  const preferred = columns.find((column) => /target|label|y|目标|因变量/i.test(column))
  return preferred ?? columns[columns.length - 1] ?? ''
}

function buildDefaultFeatures(columns: string[], targetColumn: string): string[] {
  return columns.filter((column) => column !== targetColumn).slice(0, 20)
}

function readMultiSelectValues(event: ChangeEvent<HTMLSelectElement>): string[] {
  return Array.from(event.target.selectedOptions).map((option) => option.value)
}

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

export function WorkbenchPage() {
  const [snapshot, setSnapshot] = useState<WorkbenchSnapshot | null>(null)
  const [overview, setOverview] = useState<DashboardSnapshot | null>(null)
  const [health, setHealth] = useState<HealthSnapshot | null>(null)
  const [taskSummary, setTaskSummary] = useState<TaskCenterSnapshot | null>(null)

  const [datasets, setDatasets] = useState<WorkbenchDatasetItem[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [rawPreview, setRawPreview] = useState<DatasetPreview | null>(null)
  const [modelingPreview, setModelingPreview] = useState<DatasetPreview | null>(null)
  const [previewView, setPreviewView] = useState<'raw' | 'modeling'>('raw')
  const [dictionaryBundle, setDictionaryBundle] = useState<DatasetDictionaryBundle | null>(null)
  const [selectedDictionaryFieldId, setSelectedDictionaryFieldId] = useState('')
  const [dictionaryDraft, setDictionaryDraft] = useState<TuningField | null>(null)
  const [dictionarySummaryText, setDictionarySummaryText] = useState('')

  const [taskListResponse, setTaskListResponse] = useState<TaskListResponse | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [taskDetail, setTaskDetail] = useState<TaskItem | null>(null)
  const [taskLogsResponse, setTaskLogsResponse] = useState<TaskLogsResponse | null>(null)
  const [taskMonitoring, setTaskMonitoring] = useState<MonitoringBundle | null>(null)

  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [loadingTaskPanel, setLoadingTaskPanel] = useState(false)
  const [loadingSystemPanel, setLoadingSystemPanel] = useState(false)
  const [savingDictionary, setSavingDictionary] = useState(false)

  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadDatasetName, setUploadDatasetName] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<DatasetUploadResult | null>(null)

  const [dropColumns, setDropColumns] = useState<string[]>([])
  const [missingNumeric, setMissingNumeric] = useState('median')
  const [missingCategorical, setMissingCategorical] = useState('mode')
  const [encoding, setEncoding] = useState('onehot')
  const [scaling, setScaling] = useState('none')
  const [outlierStrategy, setOutlierStrategy] = useState('none')
  const [removeDuplicates, setRemoveDuplicates] = useState(true)
  const [dropAllNullColumns, setDropAllNullColumns] = useState(true)
  const [dropSingleValueColumns, setDropSingleValueColumns] = useState(true)
  const [savePreprocessArtifacts, setSavePreprocessArtifacts] = useState(false)
  const [enableSplit, setEnableSplit] = useState(false)
  const [splitTestSize, setSplitTestSize] = useState(0.2)
  const [preprocessing, setPreprocessing] = useState(false)
  const [preprocessResult, setPreprocessResult] = useState<PreprocessTaskResult | null>(null)

  const [modelFamily, setModelFamily] = useState('machine_learning')
  const [problemType, setProblemType] = useState('regression')
  const [algorithm, setAlgorithm] = useState('random_forest')
  const [targetColumn, setTargetColumn] = useState('')
  const [featureColumns, setFeatureColumns] = useState<string[]>([])
  const [testSize, setTestSize] = useState(0.2)
  const [nEstimators, setNEstimators] = useState(200)
  const [maxDepth, setMaxDepth] = useState(5)
  const [saveModelArtifacts, setSaveModelArtifacts] = useState(false)
  const [modeling, setModeling] = useState(false)
  const [modelingResult, setModelingResult] = useState<ModelingRunResult | null>(null)

  const [workbenchNotice, setWorkbenchNotice] = useState('')
  const [workbenchError, setWorkbenchError] = useState('')

  const selectedDataset = datasets.find((item) => item.datasetId === selectedDatasetId) ?? null
  const preview = previewView === 'raw' ? rawPreview : modelingPreview
  const currentTaskLogs = taskLogsResponse?.logs ?? []

  const endpointCoverage = useMemo(
    () => [
      { name: '/api/health', active: Boolean(health) },
      { name: '/api/overview', active: Boolean(overview) },
      { name: '/api/workbench', active: Boolean(snapshot) },
      { name: '/api/tasks/summary', active: Boolean(taskSummary) },
      { name: '/api/tasks', active: Boolean(taskListResponse) },
      { name: '/api/tasks/{task_id}', active: Boolean(taskDetail) },
      { name: '/api/tasks/{task_id}/logs', active: Boolean(taskLogsResponse) },
      { name: '/api/tasks/{task_id}/monitoring', active: Boolean(taskMonitoring) },
      { name: '/api/datasets', active: datasets.length > 0 },
      { name: '/api/datasets/upload', active: Boolean(uploadResult) },
      { name: '/api/datasets/{dataset_id}/preview', active: Boolean(rawPreview || modelingPreview) },
      { name: '/api/datasets/{dataset_id}/dictionary', active: Boolean(dictionaryBundle) },
      { name: '/api/datasets/{dataset_id}/dictionary PATCH', active: Boolean(dictionaryBundle?.revisions.length) },
      { name: '/api/preprocess/run', active: Boolean(preprocessResult) },
      { name: '/api/modeling/run', active: Boolean(modelingResult) },
    ],
    [
      health,
      overview,
      snapshot,
      taskSummary,
      taskListResponse,
      taskDetail,
      taskLogsResponse,
      taskMonitoring,
      datasets.length,
      uploadResult,
      rawPreview,
      modelingPreview,
      dictionaryBundle,
      preprocessResult,
      modelingResult,
    ],
  )

  useEffect(() => {
    void refreshSystemPanels()
    void refreshDatasets()
    void refreshTaskInterfaces()
  }, [])

  useEffect(() => {
    if (!selectedDatasetId) {
      return
    }
    void hydrateDataset(selectedDatasetId)
  }, [selectedDatasetId])

  useEffect(() => {
    if (!selectedTaskId) {
      return
    }
    void hydrateTask(selectedTaskId)
  }, [selectedTaskId])

  async function refreshSystemPanels() {
    setLoadingSystemPanel(true)
    try {
      const [nextHealth, nextOverview, nextWorkbench, nextTaskSummary] = await Promise.all([
        fetchHealthSnapshot(),
        fetchDashboardSnapshot(),
        fetchWorkbenchSnapshot(),
        fetchTaskCenterSnapshot(),
      ])
      setHealth(nextHealth)
      setOverview(nextOverview)
      setSnapshot(nextWorkbench)
      setTaskSummary(nextTaskSummary)
    } catch (error) {
      console.error(error)
      setWorkbenchError('系统接口刷新失败，请确认后端已正常启动。')
    } finally {
      setLoadingSystemPanel(false)
    }
  }

  async function refreshDatasets(nextSelectedDatasetId?: string) {
    setLoadingDatasets(true)
    try {
      const items = await fetchWorkbenchDatasets()
      setDatasets(items)
      setSelectedDatasetId((current) => {
        if (nextSelectedDatasetId) {
          return nextSelectedDatasetId
        }
        if (current && items.some((item) => item.datasetId === current)) {
          return current
        }
        return items[0]?.datasetId ?? ''
      })
    } catch (error) {
      console.error(error)
      setWorkbenchError('数据集目录加载失败，请确认 FastAPI 后端已启动。')
    } finally {
      setLoadingDatasets(false)
    }
  }

  async function refreshTaskInterfaces(nextSelectedTaskId?: string) {
    setLoadingTaskPanel(true)
    try {
      const nextTaskList = await fetchTaskList({ limit: 20 })
      setTaskListResponse(nextTaskList)
      setSelectedTaskId((current) => {
        if (nextSelectedTaskId) {
          return nextSelectedTaskId
        }
        if (current && nextTaskList.items.some((item) => item.id === current)) {
          return current
        }
        return nextTaskList.items[0]?.id ?? ''
      })
    } catch (error) {
      console.error(error)
      setWorkbenchError('任务接口刷新失败，请检查后端服务状态。')
    } finally {
      setLoadingTaskPanel(false)
    }
  }

  async function hydrateDataset(datasetId: string) {
    try {
      const [nextRawPreview, nextModelingPreview, nextDictionaryBundle] = await Promise.all([
        fetchDatasetPreview(datasetId, 'raw'),
        fetchDatasetPreview(datasetId, 'modeling'),
        fetchDatasetDictionary(datasetId),
      ])

      setRawPreview(nextRawPreview)
      setModelingPreview(nextModelingPreview)
      setDictionaryBundle(nextDictionaryBundle)
      setDropColumns((current) => current.filter((column) => nextRawPreview.columns.includes(column)))

      const dictionaryFieldId = nextDictionaryBundle.fields[0]?.id ?? ''
      setSelectedDictionaryFieldId((current) =>
        current && nextDictionaryBundle.fields.some((field) => field.id === current) ? current : dictionaryFieldId,
      )
      setDictionaryDraft((current) => {
        if (current && nextDictionaryBundle.fields.some((field) => field.id === current.id)) {
          return nextDictionaryBundle.fields.find((field) => field.id === current.id) ?? nextDictionaryBundle.fields[0] ?? null
        }
        return nextDictionaryBundle.fields[0] ?? null
      })
      setDictionarySummaryText('')

      const nextTarget = buildDefaultTarget(nextModelingPreview.columns)
      const nextFeatures = buildDefaultFeatures(nextModelingPreview.columns, nextTarget)
      setTargetColumn((current) => (current && nextModelingPreview.columns.includes(current) ? current : nextTarget))
      setFeatureColumns((current) => {
        const filtered = current.filter(
          (column) => column !== nextTarget && nextModelingPreview.columns.includes(column),
        )
        return filtered.length > 0 ? filtered : nextFeatures
      })
    } catch (error) {
      console.error(error)
      setWorkbenchError('数据集相关接口加载失败，请确认该数据集仍然有效。')
    }
  }

  async function hydrateTask(taskId: string) {
    try {
      const [nextTaskDetail, nextTaskLogs, nextTaskMonitoring] = await Promise.all([
        fetchTaskDetail(taskId),
        fetchTaskLogs(taskId),
        fetchMonitoringBundle(taskId),
      ])
      setTaskDetail(nextTaskDetail)
      setTaskLogsResponse(nextTaskLogs)
      setTaskMonitoring(nextTaskMonitoring)
    } catch (error) {
      console.error(error)
      setWorkbenchError('任务详情接口加载失败，请确认任务仍然存在。')
    }
  }

  async function handleUpload() {
    if (!uploadFile) {
      setWorkbenchError('请先选择要上传的数据文件。')
      return
    }

    setUploading(true)
    setWorkbenchError('')
    setWorkbenchNotice('')

    try {
      const result = await uploadDatasetFile(uploadFile, uploadDatasetName)
      setUploadResult(result)
      setUploadDatasetName(result.datasetName)
      setUploadFile(null)
      setWorkbenchNotice(`数据集“${result.datasetName}”上传成功，已写入原始数据表。`)
      await refreshDatasets(result.datasetId)
      await refreshSystemPanels()
      setPreviewView('raw')
    } catch (error) {
      console.error(error)
      setWorkbenchError('数据上传失败，请检查文件格式、编码或后端服务状态。')
    } finally {
      setUploading(false)
    }
  }

  async function handleRunPreprocess() {
    if (!selectedDatasetId) {
      setWorkbenchError('请先选择一个数据集。')
      return
    }

    setPreprocessing(true)
    setWorkbenchError('')
    setWorkbenchNotice('')

    try {
      const result = await runPreprocessTask({
        datasetId: selectedDatasetId,
        dropColumns,
        missingNumeric,
        missingCategorical,
        outlierStrategy,
        outlierLowerQuantile: 0.01,
        outlierUpperQuantile: 0.99,
        encoding,
        scaling,
        removeDuplicates,
        dropAllNullColumns,
        dropSingleValueColumns,
        saveArtifacts: savePreprocessArtifacts,
        enableSplit,
        splitTestSize,
      })
      setPreprocessResult(result)
      setWorkbenchNotice(`预处理任务已完成，任务号：${result.taskId}`)
      setPreviewView('modeling')
      await refreshDatasets(selectedDatasetId)
      await refreshTaskInterfaces(result.taskId)
      await refreshSystemPanels()
      await hydrateDataset(selectedDatasetId)
    } catch (error) {
      console.error(error)
      setWorkbenchError('预处理任务启动失败，请检查数据集状态或后端报错信息。')
    } finally {
      setPreprocessing(false)
    }
  }

  async function handleRunModeling() {
    if (!selectedDatasetId || !targetColumn || featureColumns.length === 0) {
      setWorkbenchError('请先选择数据集、目标字段和至少一个特征字段。')
      return
    }

    setModeling(true)
    setWorkbenchError('')
    setWorkbenchNotice('')

    try {
      const result = await runModelingTask({
        datasetId: selectedDatasetId,
        preprocessRunId: modelingPreview?.preprocessRunId,
        config: {
          model_family: modelFamily,
          algorithm,
          problem_type: problemType,
          target_column: targetColumn,
          feature_columns: featureColumns,
          test_size: testSize,
          save_artifacts: saveModelArtifacts,
          n_estimators: nEstimators,
          max_depth: maxDepth,
        },
      })
      setModelingResult(result)
      setWorkbenchNotice(`建模任务已完成，任务号：${result.task.task_id}`)
      await refreshTaskInterfaces(result.task.task_id)
      await refreshSystemPanels()
    } catch (error) {
      console.error(error)
      setWorkbenchError('建模任务启动失败，请确认预处理结果、目标字段和问题类型是否匹配。')
    } finally {
      setModeling(false)
    }
  }

  function handleTargetChange(nextTarget: string) {
    const nextFeatures = (modelingPreview?.columns ?? []).filter((column) => column !== nextTarget)
    setTargetColumn(nextTarget)
    setFeatureColumns((current) => {
      const filtered = current.filter((column) => column !== nextTarget && nextFeatures.includes(column))
      return filtered.length > 0 ? filtered : nextFeatures.slice(0, 20)
    })
  }

  function handleModelFamilyChange(nextFamily: string) {
    const nextAlgorithmOptions =
      nextFamily === 'statistical' ? statisticalAlgorithmOptions : machineLearningAlgorithmOptions
    setModelFamily(nextFamily)
    setAlgorithm((current) =>
      nextAlgorithmOptions.some((item) => item.value === current)
        ? current
        : (nextAlgorithmOptions[0]?.value ?? ''),
    )
  }

  function selectDictionaryField(fieldId: string) {
    setSelectedDictionaryFieldId(fieldId)
    setDictionaryDraft(dictionaryBundle?.fields.find((field) => field.id === fieldId) ?? null)
  }

  function updateDictionaryDraft(patch: Partial<TuningField>) {
    setDictionaryDraft((current) => (current ? { ...current, ...patch } : null))
  }

  function updateDictionaryDataType(nextType: string) {
    if (!dictionaryDraft) {
      return
    }

    if (nonScalarTypes.has(nextType)) {
      updateDictionaryDraft({
        dataType: nextType,
        unit: '不适用',
        softRange: '不适用',
      })
      return
    }

    if (nextType === '时间') {
      updateDictionaryDraft({
        dataType: nextType,
        unit: dictionaryDraft.unit === '不适用' ? '天' : dictionaryDraft.unit,
        softRange: dictionaryDraft.softRange === '不适用' ? '未设置' : dictionaryDraft.softRange,
      })
      return
    }

    updateDictionaryDraft({
      dataType: nextType,
      unit: dictionaryDraft.unit === '不适用' ? '' : dictionaryDraft.unit,
      softRange: dictionaryDraft.softRange === '不适用' ? '' : dictionaryDraft.softRange,
    })
  }

  async function handleSaveDictionary() {
    if (!selectedDatasetId || !dictionaryDraft) {
      setWorkbenchError('请先选择数据集和字段，再保存字段字典。')
      return
    }
    if (!dictionarySummaryText.trim()) {
      setWorkbenchError('请填写本次字段字典更新说明。')
      return
    }

    setSavingDictionary(true)
    setWorkbenchError('')
    setWorkbenchNotice('')

    try {
      const response = await saveDatasetDictionaryField(selectedDatasetId, {
        fieldName: dictionaryDraft.fieldName,
        dataType: dictionaryDraft.dataType,
        semanticType: dictionaryDraft.semanticType,
        unit: dictionaryDraft.unit,
        softRange: dictionaryDraft.softRange,
        note: dictionaryDraft.note,
        summary: dictionarySummaryText.trim(),
        actor: 'React 工作台',
        taskId: selectedTaskId || undefined,
      })
      setDictionaryBundle(response)
      const nextField = response.fields.find((field) => field.id === dictionaryDraft.id) ?? response.fields[0] ?? null
      setDictionaryDraft(nextField)
      setSelectedDictionaryFieldId(nextField?.id ?? '')
      setDictionarySummaryText('')
      setWorkbenchNotice(`字段“${dictionaryDraft.fieldName}”的字典配置已保存。`)
      await refreshSystemPanels()
    } catch (error) {
      console.error(error)
      setWorkbenchError('字段字典保存失败，请检查后端服务状态。')
    } finally {
      setSavingDictionary(false)
    }
  }

  if (!snapshot) {
    return <div className="empty-state">正在加载工作台信息…</div>
  }

  return (
    <div className="page-layout">
      <section className="hero-band">
        <h3 className="hero-title">React 工作台接口控制台</h3>
        <p className="hero-copy">
          这页现在不仅负责上传、预处理和建模，也把系统状态、任务链路、字段字典等现有 FastAPI
          接口统一收进同一个工作台。
        </p>
        <div className="hero-inline-stats">
          {snapshot.datasets.map((item) => (
            <div className="hero-stat" key={item.label}>
              <div className="hero-stat-value">{item.value}</div>
              <div className="hero-stat-label">{item.label}</div>
            </div>
          ))}
        </div>
      </section>

      {workbenchError ? <div className="status-banner error">{workbenchError}</div> : null}
      {workbenchNotice ? <div className="status-banner success">{workbenchNotice}</div> : null}

      <div className="content-grid">
        <Panel
          title="系统接口"
          subtitle="集中显示健康检查、总览、任务摘要和工作台接口状态。"
          toolbar={
            <button className="subtle-button" type="button" onClick={() => void refreshSystemPanels()} disabled={loadingSystemPanel}>
              {loadingSystemPanel ? '正在刷新…' : '刷新系统接口'}
            </button>
          }
        >
          <div className="workbench-stack">
            <div className="info-grid">
              <div className="info-chip">
                <strong>/api/health</strong>
                <span>{health ? `${health.status} · ${health.service}` : '未加载'}</span>
              </div>
              <div className="info-chip">
                <strong>/api/overview</strong>
                <span>{overview ? `${overview.summaryCards.length} 张摘要卡片` : '未加载'}</span>
              </div>
              <div className="info-chip">
                <strong>/api/tasks/summary</strong>
                <span>{taskSummary ? `${taskSummary.tasks.length} 条任务摘要` : '未加载'}</span>
              </div>
            </div>

            {overview ? (
              <div className="simple-row">
                <strong>{overview.headline}</strong>
                <span className="muted">{overview.subtitle}</span>
              </div>
            ) : null}

            {taskSummary ? (
              <div className="tag-row">
                {taskSummary.statusStats.map((item) => (
                  <span className="tag-chip" key={item.label}>
                    {item.label}：{item.value}
                  </span>
                ))}
              </div>
            ) : null}

            <pre className="json-block">{formatJson(health)}</pre>
          </div>
        </Panel>

        <Panel title="接口覆盖清单" subtitle="绿色表示该接口已经在当前工作台页面被真实调用。">
          <div className="endpoint-grid">
            {endpointCoverage.map((item) => (
              <div className={`endpoint-card ${item.active ? 'active' : ''}`} key={item.name}>
                <strong>{item.name}</strong>
                <span>{item.active ? '已接入本页' : '待本页触发'}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="content-grid">
        <Panel title="数据集接入" subtitle="支持直接上传 CSV、XLSX、XLS 文件，写入 raw_dataset 表。">
          <div className="workbench-stack">
            <div className="field-block">
              <label className="field-label">数据集名称</label>
              <input
                className="field-input"
                placeholder="不填写时默认使用文件名"
                value={uploadDatasetName}
                onChange={(event) => setUploadDatasetName(event.target.value)}
              />
            </div>
            <div className="field-block">
              <label className="field-label">上传文件</label>
              <input
                className="field-input"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
              />
            </div>
            <div className="field-actions">
              <span className="muted">当前支持 UTF-8、GB18030 编码的 CSV，以及 Excel 文件。</span>
              <button className="primary-button" type="button" onClick={handleUpload} disabled={uploading}>
                {uploading ? '正在上传…' : '上传并入库'}
              </button>
            </div>
            {uploadResult ? (
              <div className="result-card">
                <strong>{uploadResult.datasetName}</strong>
                <div className="muted">
                  数据集编号：{uploadResult.datasetId} | 行数：{uploadResult.rowCount} | 列数：{uploadResult.columnCount}
                </div>
                <div className="muted">
                  类型待确认字段数：{Object.keys(uploadResult.typeReviewSuggestions).length}
                </div>
              </div>
            ) : null}
          </div>
        </Panel>

        <Panel
          title="数据集目录"
          subtitle="读取 /api/datasets，并同步带出预览与字段字典。"
          toolbar={
            <select
              className="field-select"
              value={selectedDatasetId}
              onChange={(event) => setSelectedDatasetId(event.target.value)}
            >
              <option value="">请选择数据集</option>
              {datasets.map((item) => (
                <option key={item.datasetId} value={item.datasetId}>
                  {item.datasetName}
                </option>
              ))}
            </select>
          }
        >
          {loadingDatasets ? (
            <div className="empty-state">正在加载数据集目录…</div>
          ) : datasets.length === 0 ? (
            <div className="empty-state">当前还没有数据集，先在左侧上传一个文件。</div>
          ) : (
            <div className="dataset-catalog">
              {datasets.map((item) => (
                <button
                  className={`dataset-card ${item.datasetId === selectedDatasetId ? 'active' : ''}`}
                  key={item.datasetId}
                  type="button"
                  onClick={() => setSelectedDatasetId(item.datasetId)}
                >
                  <strong>{item.datasetName}</strong>
                  <span className="muted">
                    {item.rowCount.toLocaleString()} 行 · {item.columnCount} 列
                  </span>
                  <span className="muted">
                    最新预处理：{item.latestPreprocessTime ? formatTimestamp(item.latestPreprocessTime) : '暂无'}
                  </span>
                </button>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <div className="content-grid">
        <Panel title="预处理任务创建" subtitle="通过 /api/preprocess/run 创建真实预处理任务。">
          {rawPreview ? (
            <div className="workbench-stack">
              <div className="form-grid">
                <div className="field-block">
                  <label className="field-label">数值缺失策略</label>
                  <select className="field-select" value={missingNumeric} onChange={(event) => setMissingNumeric(event.target.value)}>
                    {missingStrategyOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">分类型缺失策略</label>
                  <select className="field-select" value={missingCategorical} onChange={(event) => setMissingCategorical(event.target.value)}>
                    {missingStrategyOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">编码方式</label>
                  <select className="field-select" value={encoding} onChange={(event) => setEncoding(event.target.value)}>
                    {encodingOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">缩放方式</label>
                  <select className="field-select" value={scaling} onChange={(event) => setScaling(event.target.value)}>
                    {scalingOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">异常值策略</label>
                  <select className="field-select" value={outlierStrategy} onChange={(event) => setOutlierStrategy(event.target.value)}>
                    {outlierOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">训练/测试划分比例</label>
                  <input
                    className="field-input"
                    type="number"
                    min={0.1}
                    max={0.4}
                    step={0.05}
                    value={splitTestSize}
                    onChange={(event) => setSplitTestSize(Number(event.target.value))}
                  />
                </div>
              </div>

              <div className="field-block">
                <label className="field-label">直接删除字段</label>
                <select
                  className="field-select multi-select"
                  multiple
                  size={Math.min(Math.max(rawPreview.columns.length, 4), 10)}
                  value={dropColumns}
                  onChange={(event) => setDropColumns(readMultiSelectValues(event))}
                >
                  {rawPreview.columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
                <div className="footer-note">按住 `Ctrl` 或 `Shift` 可以多选。</div>
              </div>

              <div className="checkbox-grid">
                <label className="check-row">
                  <input type="checkbox" checked={removeDuplicates} onChange={(event) => setRemoveDuplicates(event.target.checked)} />
                  <span>删除重复行</span>
                </label>
                <label className="check-row">
                  <input type="checkbox" checked={dropAllNullColumns} onChange={(event) => setDropAllNullColumns(event.target.checked)} />
                  <span>删除全空列</span>
                </label>
                <label className="check-row">
                  <input type="checkbox" checked={dropSingleValueColumns} onChange={(event) => setDropSingleValueColumns(event.target.checked)} />
                  <span>删除单一取值列</span>
                </label>
                <label className="check-row">
                  <input type="checkbox" checked={enableSplit} onChange={(event) => setEnableSplit(event.target.checked)} />
                  <span>生成训练/测试切分</span>
                </label>
                <label className="check-row">
                  <input type="checkbox" checked={savePreprocessArtifacts} onChange={(event) => setSavePreprocessArtifacts(event.target.checked)} />
                  <span>保存预处理产物</span>
                </label>
              </div>

              <div className="field-actions">
                <span className="muted">运行后会写入 `preprocessed_data`，并同步进入任务中心。</span>
                <button className="primary-button" type="button" onClick={handleRunPreprocess} disabled={preprocessing}>
                  {preprocessing ? '正在运行…' : '启动预处理任务'}
                </button>
              </div>
            </div>
          ) : (
            <div className="empty-state">先选择一个数据集，再配置预处理任务。</div>
          )}
        </Panel>

        <Panel title="建模任务创建" subtitle="通过 /api/modeling/run 创建真实建模任务。">
          {modelingPreview ? (
            <div className="workbench-stack">
              <div className="form-grid">
                <div className="field-block">
                  <label className="field-label">建模家族</label>
                  <select className="field-select" value={modelFamily} onChange={(event) => handleModelFamilyChange(event.target.value)}>
                    {modelFamilyOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">问题类型</label>
                  <select className="field-select" value={problemType} onChange={(event) => setProblemType(event.target.value)}>
                    {problemTypeOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">建模算法</label>
                  <select className="field-select" value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
                    {(modelFamily === 'statistical' ? statisticalAlgorithmOptions : machineLearningAlgorithmOptions).map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">目标字段</label>
                  <select className="field-select" value={targetColumn} onChange={(event) => handleTargetChange(event.target.value)}>
                    {modelingPreview.columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">测试集比例</label>
                  <input
                    className="field-input"
                    type="number"
                    min={0.1}
                    max={0.4}
                    step={0.05}
                    value={testSize}
                    onChange={(event) => setTestSize(Number(event.target.value))}
                  />
                </div>
                <div className="field-block">
                  <label className="field-label">树模型数量</label>
                  <input
                    className="field-input"
                    type="number"
                    min={50}
                    max={500}
                    step={50}
                    value={nEstimators}
                    onChange={(event) => setNEstimators(Number(event.target.value))}
                    disabled={modelFamily !== 'machine_learning'}
                  />
                </div>
                <div className="field-block">
                  <label className="field-label">树深度</label>
                  <input
                    className="field-input"
                    type="number"
                    min={2}
                    max={12}
                    step={1}
                    value={maxDepth}
                    onChange={(event) => setMaxDepth(Number(event.target.value))}
                    disabled={modelFamily !== 'machine_learning'}
                  />
                </div>
                <div className="field-block">
                  <label className="field-label">建模数据源</label>
                  <input
                    className="field-input"
                    readOnly
                    value={modelingPreview.sourceKind === 'preprocessed_data' ? '最新预处理结果' : '原始数据表'}
                  />
                </div>
              </div>

              <div className="field-block">
                <label className="field-label">特征字段</label>
                <div className="inline-actions">
                  <button
                    className="subtle-button"
                    type="button"
                    onClick={() => setFeatureColumns(buildDefaultFeatures(modelingPreview.columns, targetColumn))}
                  >
                    选择默认特征
                  </button>
                  <button className="subtle-button" type="button" onClick={() => setFeatureColumns([])}>
                    清空特征
                  </button>
                </div>
                <select
                  className="field-select multi-select"
                  multiple
                  size={Math.min(Math.max(modelingPreview.columns.length, 4), 10)}
                  value={featureColumns}
                  onChange={(event) => setFeatureColumns(readMultiSelectValues(event))}
                >
                  {modelingPreview.columns
                    .filter((column) => column !== targetColumn)
                    .map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                </select>
              </div>

              <div className="checkbox-grid">
                <label className="check-row">
                  <input type="checkbox" checked={saveModelArtifacts} onChange={(event) => setSaveModelArtifacts(event.target.checked)} />
                  <span>保存建模产物</span>
                </label>
              </div>

              <div className="field-actions">
                <span className="muted">建模结果会写入任务结果，并在任务接口面板中即时可查。</span>
                <button className="primary-button" type="button" onClick={handleRunModeling} disabled={modeling}>
                  {modeling ? '正在建模…' : '启动建模任务'}
                </button>
              </div>
            </div>
          ) : (
            <div className="empty-state">先选择数据集或先完成一轮预处理，再创建建模任务。</div>
          )}
        </Panel>
      </div>

      <div className="content-grid">
        <Panel
          title="任务接口"
          subtitle="统一调用 /api/tasks、/api/tasks/{task_id}、/logs、/monitoring。"
          toolbar={
            <div className="panel-toolbar">
              <select
                className="field-select"
                value={selectedTaskId}
                onChange={(event) => setSelectedTaskId(event.target.value)}
              >
                <option value="">请选择任务</option>
                {(taskListResponse?.items ?? []).map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
              <button className="subtle-button" type="button" onClick={() => void refreshTaskInterfaces()} disabled={loadingTaskPanel}>
                {loadingTaskPanel ? '正在刷新…' : '刷新任务接口'}
              </button>
            </div>
          }
        >
          {taskListResponse && taskListResponse.items.length > 0 ? (
            <div className="workbench-stack">
              <div className="info-grid">
                <div className="info-chip">
                  <strong>/api/tasks</strong>
                  <span>{taskListResponse.total} 条任务</span>
                </div>
                <div className="info-chip">
                  <strong>/api/tasks/{'{task_id}'}</strong>
                  <span>{taskDetail ? taskDetail.name : '未选中任务'}</span>
                </div>
                <div className="info-chip">
                  <strong>/api/tasks/{'{task_id}'}/logs</strong>
                  <span>{currentTaskLogs.length} 条日志</span>
                </div>
              </div>

              {taskDetail ? (
                <article className="task-row">
                  <div>
                    <h4 className="task-title">{taskDetail.name}</h4>
                    <div className="footer-note">{taskDetail.dataset}</div>
                  </div>
                  <div className="task-meta">
                    <span>阶段：{taskDetail.stage}</span>
                    <span>启动：{taskDetail.startedAt}</span>
                    <span>记录数：{taskDetail.records}</span>
                  </div>
                  <div className="task-side">
                    <TaskStatusPill status={taskDetail.status} />
                    <span>耗时：{taskDetail.duration}</span>
                    <span>告警数：{taskDetail.alerts}</span>
                  </div>
                </article>
              ) : null}

              {taskMonitoring ? (
                <div className="timeline">
                  {taskMonitoring.timeline.slice(0, 6).map((step) => (
                    <div
                      className={`timeline-item ${step.status === 'failed' ? 'failed' : step.status === 'waiting' ? 'waiting' : ''}`}
                      key={step.name}
                    >
                      <div className="timeline-title">
                        {step.name} · {step.duration}
                      </div>
                      <div className="timeline-detail">{step.detail}</div>
                    </div>
                  ))}
                </div>
              ) : null}

              <pre className="json-block">{formatJson(taskLogsResponse)}</pre>
            </div>
          ) : (
            <div className="empty-state">当前没有任务记录，先运行预处理或建模任务。</div>
          )}
        </Panel>

        <Panel title="字段字典接口" subtitle="统一调用 /api/datasets/{dataset_id}/dictionary 和 PATCH 保存。">
          {dictionaryBundle ? (
            <div className="workbench-stack">
              <div className="tag-row">
                {dictionaryBundle.fields.map((field) => (
                  <button
                    className="subtle-button"
                    key={field.id}
                    type="button"
                    onClick={() => selectDictionaryField(field.id)}
                    style={{
                      background: field.id === selectedDictionaryFieldId ? 'rgba(32, 84, 91, 0.12)' : undefined,
                      color: field.id === selectedDictionaryFieldId ? '#1d4f57' : undefined,
                    }}
                  >
                    {field.fieldName}
                  </button>
                ))}
              </div>

              {dictionaryDraft ? (
                <>
                  <div className="form-grid">
                    <div className="field-block">
                      <label className="field-label">字段名称</label>
                      <input className="field-input" readOnly value={dictionaryDraft.fieldName} />
                    </div>
                    <div className="field-block">
                      <label className="field-label">基础类型</label>
                      <select
                        className="field-select"
                        value={dictionaryDraft.dataType}
                        onChange={(event) => updateDictionaryDataType(event.target.value)}
                      >
                        {dataTypeOptions.map((item) => (
                          <option key={item} value={item}>
                            {item}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-block">
                      <label className="field-label">语义类型</label>
                      <select
                        className="field-select"
                        value={dictionaryDraft.semanticType}
                        onChange={(event) => updateDictionaryDraft({ semanticType: event.target.value })}
                      >
                        {semanticTypeOptions.map((item) => (
                          <option key={item} value={item}>
                            {item}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-block">
                      <label className="field-label">{dictionaryDraft.dataType === '时间' ? '时间单位/粒度' : '单位'}</label>
                      <input
                        className="field-input"
                        value={dictionaryDraft.unit}
                        disabled={nonScalarTypes.has(dictionaryDraft.dataType)}
                        onChange={(event) => updateDictionaryDraft({ unit: event.target.value })}
                      />
                    </div>
                    <div className="field-block">
                      <label className="field-label">{dictionaryDraft.dataType === '时间' ? '时间范围' : '软范围'}</label>
                      <input
                        className="field-input"
                        value={dictionaryDraft.softRange}
                        disabled={nonScalarTypes.has(dictionaryDraft.dataType)}
                        onChange={(event) => updateDictionaryDraft({ softRange: event.target.value })}
                      />
                    </div>
                    <div className="field-block">
                      <label className="field-label">字段备注</label>
                      <input
                        className="field-input"
                        value={dictionaryDraft.note}
                        onChange={(event) => updateDictionaryDraft({ note: event.target.value })}
                      />
                    </div>
                  </div>

                  <div className="field-block">
                    <label className="field-label">本次更新说明</label>
                    <textarea
                      className="field-textarea"
                      placeholder="例如：将城市群名称调整为地理区域语义，并禁用单位与范围。"
                      value={dictionarySummaryText}
                      onChange={(event) => setDictionarySummaryText(event.target.value)}
                    />
                  </div>

                  <div className="field-actions">
                    <span className="muted">
                      保存后会写入字段字典覆盖表与修订历史表。
                    </span>
                    <button className="primary-button" type="button" onClick={handleSaveDictionary} disabled={savingDictionary}>
                      {savingDictionary ? '正在保存…' : '保存字典修改'}
                    </button>
                  </div>
                </>
              ) : null}

              <div className="history-list">
                {dictionaryBundle.revisions.slice(0, 6).map((revision) => (
                  <article className="history-row" key={revision.id}>
                    <strong>
                      {revision.time} · {revision.fieldName}
                    </strong>
                    <div>{revision.summary}</div>
                    <div className="muted">执行主体：{revision.actor}</div>
                  </article>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-state">先选择数据集，再加载字段字典接口。</div>
          )}
        </Panel>
      </div>

      <div className="content-grid">
        <Panel
          title="数据预览接口"
          subtitle="在原始数据视图和建模数据视图之间切换，分别对应不同的 preview 调用。"
          toolbar={
            <div className="tag-row">
              <button
                className="subtle-button"
                type="button"
                onClick={() => setPreviewView('raw')}
                style={{ background: previewView === 'raw' ? 'rgba(32, 84, 91, 0.12)' : undefined }}
              >
                原始数据
              </button>
              <button
                className="subtle-button"
                type="button"
                onClick={() => setPreviewView('modeling')}
                style={{ background: previewView === 'modeling' ? 'rgba(32, 84, 91, 0.12)' : undefined }}
              >
                建模数据
              </button>
            </div>
          }
        >
          {preview ? (
            <div className="workbench-stack">
              <div className="info-grid">
                <div className="info-chip">
                  <strong>数据源</strong>
                  <span>{preview.sourceKind === 'preprocessed_data' ? '预处理结果表' : '原始数据表'}</span>
                </div>
                <div className="info-chip">
                  <strong>维度</strong>
                  <span>{preview.rowCount.toLocaleString()} 行 · {preview.columnCount} 列</span>
                </div>
                <div className="info-chip">
                  <strong>预处理版本</strong>
                  <span>{preview.preprocessRunId ?? '暂无'}</span>
                </div>
              </div>

              <div className="preview-table-wrap">
                {preview.rows.length > 0 ? (
                  <table className="preview-table">
                    <thead>
                      <tr>
                        {preview.columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.rows.map((row, rowIndex) => (
                        <tr key={`${preview.datasetId}-${rowIndex}`}>
                          {preview.columns.map((column) => (
                            <td key={`${rowIndex}-${column}`}>{formatCellValue(row[column])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="empty-state">当前预览暂无样例行，但列结构已经成功返回。</div>
                )}
              </div>
            </div>
          ) : (
            <div className="empty-state">选择一个数据集后，这里会显示真实预览。</div>
          )}
        </Panel>

        <Panel title="接口返回摘要" subtitle="方便你快速确认这一页已经把主要接口都真正打通。">
          <div className="simple-list">
            <div className="simple-row">
              <strong>上传接口</strong>
              {uploadResult ? (
                <span className="muted">
                  {uploadResult.datasetName} · {uploadResult.rowCount} 行 · {uploadResult.columnCount} 列
                </span>
              ) : (
                <span className="muted">暂未触发上传接口。</span>
              )}
            </div>
            <div className="simple-row">
              <strong>预处理接口</strong>
              {preprocessResult ? (
                <span className="muted">
                  任务号 {preprocessResult.taskId} · {preprocessResult.rawShape.join('x')} → {preprocessResult.transformedShape.join('x')}
                </span>
              ) : (
                <span className="muted">暂未触发预处理接口。</span>
              )}
            </div>
            <div className="simple-row">
              <strong>建模接口</strong>
              {modelingResult ? (
                <span className="muted">
                  任务号 {modelingResult.task.task_id} · 指标 {Object.entries(modelingResult.result.metrics).map(([key, value]) => `${key}=${value.toFixed(4)}`).join(' | ')}
                </span>
              ) : (
                <span className="muted">暂未触发建模接口。</span>
              )}
            </div>
            <div className="simple-row">
              <strong>任务日志接口</strong>
              <span className="muted">{currentTaskLogs.length > 0 ? `${currentTaskLogs.length} 条日志已返回` : '暂未加载任务日志。'}</span>
            </div>
            <div className="simple-row">
              <strong>字段字典接口</strong>
              <span className="muted">{dictionaryBundle ? `${dictionaryBundle.fields.length} 个字段，${dictionaryBundle.revisions.length} 条修订` : '暂未加载字段字典。'}</span>
            </div>
            {selectedDataset ? (
              <div className="simple-row">
                <strong>当前选中数据集</strong>
                <span className="muted">
                  {selectedDataset.datasetName} · 上传时间 {formatTimestamp(selectedDataset.uploadTime)}
                </span>
              </div>
            ) : null}
          </div>
        </Panel>
      </div>

      <div className="content-grid equal">
        <Panel title="工作台接口契约" subtitle="这里直接展示当前由后端返回的接口清单。">
          <div className="task-list">
            {snapshot.apiContracts.map((item) => (
              <article className="task-row" key={`${item.method}-${item.name}`}>
                <div>
                  <h4 className="task-title">{item.name}</h4>
                </div>
                <div className="task-meta">
                  <span>请求方法：{item.method}</span>
                </div>
                <div className="task-side">
                  <span className="tag-chip">{item.status}</span>
                </div>
              </article>
            ))}
          </div>
        </Panel>

        <Panel title="系统状态图" subtitle="继续保留工作台的后端实时统计视图。">
          <ReactECharts
            option={{
              tooltip: { trigger: 'axis' },
              grid: { left: 10, right: 10, bottom: 12, top: 12, containLabel: true },
              xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(70, 90, 97, 0.08)' } } },
              yAxis: { type: 'category', data: snapshot.datasets.map((item) => item.label) },
              series: [
                {
                  type: 'bar',
                  data: snapshot.datasets.map((item) => item.value),
                  itemStyle: { color: '#244d58', borderRadius: [0, 10, 10, 0] },
                  label: { show: true, position: 'right' },
                },
              ],
            }}
            style={{ height: 280 }}
          />
        </Panel>
      </div>
    </div>
  )
}
