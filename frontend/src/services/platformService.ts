import {
  dashboardSnapshot,
  monitoringBundles,
  taskCenterSnapshot,
  workbenchSnapshot,
} from '../mocks/platformData'
import type {
  DashboardSnapshot,
  DatasetDictionaryBundle,
  DatasetPreview,
  DatasetUploadResult,
  HealthSnapshot,
  ModelingRunResult,
  MonitoringBundle,
  RevisionEntry,
  TaskItem,
  TaskListResponse,
  TaskLogsResponse,
  TaskCenterSnapshot,
  TuningField,
  WorkbenchDatasetItem,
  WorkbenchSnapshot,
  PreprocessTaskResult,
} from '../types/platform'

export interface SaveDatasetDictionaryFieldPayload {
  fieldName: string
  dataType: string
  semanticType: string
  unit: string
  softRange: string
  note: string
  summary: string
  actor?: string
  taskId?: string
}

export interface RunPreprocessTaskPayload {
  datasetId: string
  sourceName?: string
  dropColumns: string[]
  missingNumeric: string
  missingCategorical: string
  missingConstantValue?: string
  outlierStrategy: string
  outlierLowerQuantile: number
  outlierUpperQuantile: number
  encoding: string
  scaling: string
  removeDuplicates: boolean
  dropAllNullColumns: boolean
  dropSingleValueColumns: boolean
  saveArtifacts: boolean
  enableSplit: boolean
  splitTestSize: number
}

export interface RunModelingTaskPayload {
  datasetId: string
  preprocessRunId?: string | null
  sourceName?: string
  config: {
    model_family: string
    algorithm: string
    problem_type: string
    target_column: string
    feature_columns: string[]
    test_size: number
    random_state?: number
    save_artifacts: boolean
    n_estimators?: number
    max_depth?: number
    learning_rate?: number
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')

  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData
  if (init?.body && !isFormData && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(path, {
    ...init,
    headers,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`API ${response.status}: ${path} -> ${detail}`)
  }

  return (await response.json()) as T
}

async function withFallback<T>(request: () => Promise<T>, fallback: () => T | Promise<T>): Promise<T> {
  try {
    return await request()
  } catch (error) {
    console.warn('API 请求失败，已回退到本地 mock 数据。', error)
    return await fallback()
  }
}

export async function fetchDashboardSnapshot(): Promise<DashboardSnapshot> {
  return withFallback(
    () => requestJson<DashboardSnapshot>('/api/overview'),
    async () => dashboardSnapshot,
  )
}

export async function fetchHealthSnapshot(): Promise<HealthSnapshot> {
  return requestJson<HealthSnapshot>('/api/health')
}

export async function fetchTaskCenterSnapshot(): Promise<TaskCenterSnapshot> {
  return withFallback(
    () => requestJson<TaskCenterSnapshot>('/api/tasks/summary'),
    async () => taskCenterSnapshot,
  )
}

export async function fetchTaskList(params?: {
  taskType?: string
  status?: string
  datasetId?: string
  limit?: number
}): Promise<TaskListResponse> {
  return withFallback(
    async () => {
      const search = new URLSearchParams()
      if (params?.taskType) search.set('task_type', params.taskType)
      if (params?.status) search.set('status', params.status)
      if (params?.datasetId) search.set('dataset_id', params.datasetId)
      if (params?.limit) search.set('limit', String(params.limit))
      const suffix = search.toString()
      return requestJson<TaskListResponse>(`/api/tasks${suffix ? `?${suffix}` : ''}`)
    },
    async () => {
      const items = taskCenterSnapshot.tasks.filter((task) => {
        if (params?.taskType && task.type !== params.taskType) return false
        if (params?.status && task.status !== params.status) return false
        if (params?.datasetId && task.datasetId !== params.datasetId) return false
        return true
      })
      return {
        items: items.slice(0, params?.limit ?? items.length),
        total: items.length,
      }
    },
  )
}

export async function fetchTaskDetail(taskId: string): Promise<TaskItem> {
  return withFallback(
    () => requestJson<TaskItem>(`/api/tasks/${taskId}`),
    async () => {
      const bundle = monitoringBundles[taskId] ?? monitoringBundles['TASK-20260518-014']
      return bundle.task
    },
  )
}

export async function fetchTaskLogs(taskId: string): Promise<TaskLogsResponse> {
  return withFallback(
    () => requestJson<TaskLogsResponse>(`/api/tasks/${taskId}/logs`),
    async () => {
      const bundle = monitoringBundles[taskId] ?? monitoringBundles['TASK-20260518-014']
      return {
        taskId,
        logs: bundle.logs,
      }
    },
  )
}

export async function fetchMonitoringBundle(taskId: string): Promise<MonitoringBundle> {
  return withFallback(
    () => requestJson<MonitoringBundle>(`/api/tasks/${taskId}/monitoring`),
    async () => monitoringBundles[taskId] ?? monitoringBundles['TASK-20260518-014'],
  )
}

export async function fetchWorkbenchSnapshot(): Promise<WorkbenchSnapshot> {
  return withFallback(
    () => requestJson<WorkbenchSnapshot>('/api/workbench'),
    async () => workbenchSnapshot,
  )
}

export async function fetchWorkbenchDatasets(): Promise<WorkbenchDatasetItem[]> {
  return withFallback(
    async () => {
      const response = await requestJson<{ items: WorkbenchDatasetItem[] }>('/api/datasets')
      return response.items
    },
    async () => buildMockDatasetCatalog(),
  )
}

export async function fetchDatasetPreview(
  datasetId: string,
  view: 'raw' | 'modeling' = 'raw',
): Promise<DatasetPreview> {
  return withFallback(
    () => requestJson<DatasetPreview>(`/api/datasets/${datasetId}/preview?view=${view}&limit=10`),
    async () => buildMockDatasetPreview(datasetId, view),
  )
}

export async function fetchDatasetDictionary(datasetId: string): Promise<DatasetDictionaryBundle> {
  return withFallback(
    () => requestJson<DatasetDictionaryBundle>(`/api/datasets/${datasetId}/dictionary`),
    async () => {
      const monitoringEntry =
        Object.values(monitoringBundles).find((item) => item.task.datasetId === datasetId)
        ?? monitoringBundles['TASK-20260518-014']
      return {
        datasetId,
        datasetName: monitoringEntry.task.dataset,
        fields: monitoringEntry.tuningFields,
        revisions: monitoringEntry.revisions,
      }
    },
  )
}

export async function uploadDatasetFile(file: File, datasetName?: string): Promise<DatasetUploadResult> {
  const formData = new FormData()
  formData.append('file', file)
  if (datasetName?.trim()) {
    formData.append('dataset_name', datasetName.trim())
  }

  return requestJson<DatasetUploadResult>('/api/datasets/upload', {
    method: 'POST',
    body: formData,
  })
}

export async function runPreprocessTask(payload: RunPreprocessTaskPayload): Promise<PreprocessTaskResult> {
  return requestJson<PreprocessTaskResult>('/api/preprocess/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function runModelingTask(payload: RunModelingTaskPayload): Promise<ModelingRunResult> {
  return requestJson<ModelingRunResult>('/api/modeling/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function saveDatasetDictionaryField(
  datasetId: string,
  payload: SaveDatasetDictionaryFieldPayload,
): Promise<DatasetDictionaryBundle> {
  return withFallback(
    () =>
      requestJson<DatasetDictionaryBundle>(`/api/datasets/${datasetId}/dictionary`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      }),
    async () => saveMockDictionaryField(datasetId, payload),
  )
}

function buildMockDatasetCatalog(): WorkbenchDatasetItem[] {
  const seen = new Set<string>()
  return Object.values(monitoringBundles)
    .filter((bundle) => {
      if (!bundle.task.datasetId || seen.has(bundle.task.datasetId)) {
        return false
      }
      seen.add(bundle.task.datasetId)
      return true
    })
    .map((bundle) => ({
      datasetId: bundle.task.datasetId!,
      datasetName: bundle.task.dataset,
      sourceFileName: `${bundle.task.dataset}.csv`,
      uploadTime: bundle.task.startedAt,
      rowCount: Number(bundle.task.records.replaceAll(',', '')) || 0,
      columnCount: bundle.tuningFields.length,
      latestPreprocessRunId: bundle.task.type === 'preprocess' ? bundle.task.version : undefined,
      latestPreprocessTime: bundle.task.startedAt,
    }))
}

function buildMockDatasetPreview(
  datasetId: string,
  view: 'raw' | 'modeling',
): DatasetPreview {
  const bundle =
    Object.values(monitoringBundles).find((item) => item.task.datasetId === datasetId)
    ?? monitoringBundles['TASK-20260518-014']

  const columns = bundle.tuningFields.map((field) => field.fieldName)
  return {
    datasetId,
    datasetName: bundle.task.dataset,
    sourceKind: view === 'modeling' ? 'preprocessed_data' : 'raw_dataset',
    preprocessRunId: view === 'modeling' ? bundle.task.version : null,
    rowCount: Number(bundle.task.records.replaceAll(',', '')) || 0,
    columnCount: columns.length,
    columns,
    rows: [],
  }
}

function saveMockDictionaryField(
  datasetId: string,
  payload: SaveDatasetDictionaryFieldPayload,
): DatasetDictionaryBundle {
  const monitoringEntry =
    Object.values(monitoringBundles).find((item) => item.task.datasetId === datasetId)
    ?? monitoringBundles[payload.taskId ?? 'TASK-20260518-014']
    ?? monitoringBundles['TASK-20260518-014']

  const fields = monitoringEntry.tuningFields.map((field) =>
    field.fieldName === payload.fieldName
      ? {
          ...field,
          dataType: payload.dataType,
          semanticType: payload.semanticType,
          unit: payload.unit,
          softRange: payload.softRange,
          note: payload.note,
        }
      : field,
  )

  const nextRevision: RevisionEntry = {
    id: `MOCK-REV-${Date.now()}`,
    time: '刚刚',
    actor: payload.actor ?? '前端演示',
    fieldName: payload.fieldName,
    summary: payload.summary || '更新了字段元数据。',
  }

  monitoringEntry.tuningFields = fields as TuningField[]
  monitoringEntry.revisions = [nextRevision, ...monitoringEntry.revisions]

  return {
    datasetId,
    datasetName: monitoringEntry.task.dataset,
    fields,
    revisions: monitoringEntry.revisions,
  }
}
