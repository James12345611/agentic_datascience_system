export type TaskStatus = 'completed' | 'running' | 'failed' | 'queued'
export type Tone = 'teal' | 'amber' | 'copper' | 'slate'
export type LogLevel = '信息' | '警告' | '错误'

export interface SummaryCard {
  title: string
  value: string
  delta: string
  note: string
  tone: Tone
}

export interface HeroStat {
  label: string
  value: string
}

export interface TrendPoint {
  label: string
  total: number
  preprocess: number
  modeling: number
}

export interface RankPoint {
  label: string
  score: number
}

export interface AlertItem {
  title: string
  detail: string
  severity: '高' | '中' | '低'
}

export interface TaskItem {
  id: string
  name: string
  type: string
  dataset: string
  datasetId?: string
  status: TaskStatus
  owner: string
  startedAt: string
  duration: string
  stage: string
  records: string
  version: string
  alerts: number
}

export interface DashboardSnapshot {
  headline: string
  subtitle: string
  heroStats: HeroStat[]
  summaryCards: SummaryCard[]
  trend: TrendPoint[]
  ranking: RankPoint[]
  alerts: AlertItem[]
  tasks: TaskItem[]
}

export interface StageStat {
  label: string
  value: number
}

export interface TaskCenterSnapshot {
  tasks: TaskItem[]
  stageStats: StageStat[]
  statusStats: StageStat[]
}

export interface TimelineStep {
  name: string
  status: 'done' | 'working' | 'waiting' | 'failed'
  duration: string
  detail: string
}

export interface LogEntry {
  id: string
  time: string
  level: LogLevel
  module: string
  message: string
  field?: string
}

export interface TuningField {
  id: string
  fieldName: string
  dataType: string
  semanticType: string
  unit: string
  softRange: string
  note: string
}

export interface RevisionEntry {
  id: string
  time: string
  actor: string
  fieldName: string
  summary: string
}

export interface DatasetDictionaryBundle {
  datasetId: string
  datasetName: string
  fields: TuningField[]
  revisions: RevisionEntry[]
}

export interface MonitoringBundle {
  task: TaskItem
  timeline: TimelineStep[]
  logs: LogEntry[]
  tuningFields: TuningField[]
  revisions: RevisionEntry[]
}

export interface ModuleStatus {
  label: string
  status: string
  detail: string
}

export interface WorkbenchSnapshot {
  modules: ModuleStatus[]
  datasets: StageStat[]
  nextActions: string[]
  apiContracts: Array<{ name: string; method: string; status: string }>
}

export interface WorkbenchDatasetItem {
  datasetId: string
  datasetName: string
  sourceFileName?: string | null
  uploadTime: string
  rowCount: number
  columnCount: number
  latestPreprocessRunId?: string | null
  latestPreprocessTime?: string | null
}

export interface HealthSnapshot {
  status: string
  service: string
  dbPath: string
}

export interface DatasetPreview {
  datasetId: string
  datasetName: string
  sourceKind: 'raw_dataset' | 'preprocessed_data'
  preprocessRunId?: string | null
  rowCount: number
  columnCount: number
  columns: string[]
  rows: Array<Record<string, unknown>>
}

export interface DatasetUploadResult {
  datasetId: string
  datasetName: string
  sourceFileName?: string | null
  rowCount: number
  columnCount: number
  preview: DatasetPreview
  inferredTypes: Record<string, string>
  typeReviewSuggestions: Record<string, string[]>
  datasetSummary: Record<string, unknown>
}

export interface PreprocessTaskResult {
  datasetId: string
  datasetName: string
  taskId: string
  taskStatus: TaskStatus | string
  preprocessRunId?: string | null
  rawShape: number[]
  cleanedShape: number[]
  transformedShape: number[]
  validationIssueCount: number
  suggestionCount: number
  preview: DatasetPreview
}

export interface BackendTaskRun {
  task_id: string
  task_type: string
  dataset_id?: string | null
  status: string
  created_time: string
  updated_time: string
  source_name?: string | null
  request_payload_json?: Record<string, unknown>
  result_payload_json?: Record<string, unknown> | null
  error_message?: string | null
}

export interface ModelingMetrics {
  [key: string]: number
}

export interface ModelingCoefficientRow {
  term: string
  coefficient: number
  p_value?: number | null
  ci_lower?: number | null
  ci_upper?: number | null
}

export interface ModelingFeatureImportanceRow {
  feature: string
  importance: number
}

export interface ModelingResultPayload {
  model_type: string
  model_family: string
  problem_type: string
  feature_columns: string[]
  sample_size: number
  train_size?: number
  test_size?: number
  metrics: ModelingMetrics
  coefficient_table?: ModelingCoefficientRow[]
  feature_importance_table?: ModelingFeatureImportanceRow[]
  summary_text?: string
}

export interface ModelingRunResult {
  dataSource: string
  task: BackendTaskRun
  result: ModelingResultPayload
  artifacts: Record<string, unknown>
}

export interface TaskListResponse {
  items: TaskItem[]
  total: number
}

export interface TaskLogsResponse {
  taskId: string
  logs: LogEntry[]
}
