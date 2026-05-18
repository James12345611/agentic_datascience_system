import { useEffect, useMemo, useState } from 'react'
import { Panel } from '../components/Panel'
import { TaskStatusPill } from '../components/TaskStatusPill'
import {
  fetchMonitoringBundle,
  fetchTaskCenterSnapshot,
  saveDatasetDictionaryField,
} from '../services/platformService'
import type { MonitoringBundle, TuningField } from '../types/platform'

const baseTypeOptions = ['数值', '分类', '时间', '布尔', '文本', '标识']
const baseSemanticOptions = [
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

function uniqueOptions(values: Array<string | undefined>) {
  return values.filter((value, index, array): value is string => Boolean(value) && array.indexOf(value) === index)
}

export function MonitoringPage() {
  const [taskIds, setTaskIds] = useState<string[]>([])
  const [selectedTaskId, setSelectedTaskId] = useState<string>('')
  const [bundle, setBundle] = useState<MonitoringBundle | null>(null)
  const [draftFields, setDraftFields] = useState<TuningField[]>([])
  const [selectedFieldId, setSelectedFieldId] = useState<string>('')
  const [revisionNotes, setRevisionNotes] = useState<string>('')
  const [tasksLoaded, setTasksLoaded] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState<string>('')

  useEffect(() => {
    void fetchTaskCenterSnapshot()
      .then((snapshot) => {
        const ids = snapshot.tasks.map((task) => task.id)
        setTaskIds(ids)
        setSelectedTaskId(ids[0] ?? '')
      })
      .finally(() => setTasksLoaded(true))
  }, [])

  useEffect(() => {
    if (!selectedTaskId) {
      return
    }

    void fetchMonitoringBundle(selectedTaskId).then((nextBundle) => {
      setBundle(nextBundle)
      setDraftFields(nextBundle.tuningFields)
      setSelectedFieldId((current) => current || nextBundle.tuningFields[0]?.id || '')
      setRevisionNotes('')
    })
  }, [selectedTaskId])

  const selectedField = draftFields.find((field) => field.id === selectedFieldId) ?? null

  const typeOptions = useMemo(
    () => uniqueOptions([...baseTypeOptions, selectedField?.dataType]),
    [selectedField],
  )
  const semanticOptions = useMemo(
    () => uniqueOptions([...baseSemanticOptions, selectedField?.semanticType]),
    [selectedField],
  )
  const disableUnitAndRange = selectedField ? nonScalarTypes.has(selectedField.dataType) : false
  const unitLabel = selectedField?.dataType === '时间' ? '时间单位/粒度' : '单位'
  const rangeLabel = selectedField?.dataType === '时间' ? '时间范围' : '软范围'

  function updateFieldDraft(patch: Partial<TuningField>) {
    if (!selectedField) {
      return
    }

    setDraftFields((current) =>
      current.map((field) => (field.id === selectedField.id ? { ...field, ...patch } : field)),
    )
  }

  function updateDataType(nextType: string) {
    if (nonScalarTypes.has(nextType)) {
      updateFieldDraft({
        dataType: nextType,
        unit: '不适用',
        softRange: '不适用',
      })
      return
    }

    if (nextType === '时间') {
      updateFieldDraft({
        dataType: nextType,
        unit: selectedField?.unit === '不适用' ? '天' : selectedField?.unit,
        softRange: selectedField?.softRange === '不适用' ? '未设置' : selectedField?.softRange,
      })
      return
    }

    updateFieldDraft({
      dataType: nextType,
      unit: selectedField?.unit === '不适用' ? '' : selectedField?.unit,
      softRange: selectedField?.softRange === '不适用' ? '' : selectedField?.softRange,
    })
  }

  async function saveRevision() {
    if (!bundle || !selectedField || !revisionNotes.trim()) {
      setSaveMessage('请先选择字段，并填写本次修正说明。')
      return
    }

    if (!bundle.task.datasetId) {
      setSaveMessage('当前任务未关联数据集，暂时无法保存字段字典。')
      return
    }

    setIsSaving(true)
    setSaveMessage('')

    try {
      const response = await saveDatasetDictionaryField(bundle.task.datasetId, {
        fieldName: selectedField.fieldName,
        dataType: selectedField.dataType,
        semanticType: selectedField.semanticType,
        unit: selectedField.unit,
        softRange: selectedField.softRange,
        note: selectedField.note,
        summary: revisionNotes.trim(),
        actor: '前端人工微调',
        taskId: bundle.task.id,
      })

      setDraftFields(response.fields)
      setBundle({
        ...bundle,
        tuningFields: response.fields,
        revisions: response.revisions,
      })
      setSelectedFieldId((current) => {
        if (response.fields.some((field) => field.id === current)) {
          return current
        }
        return response.fields[0]?.id ?? ''
      })
      setRevisionNotes('')
      setSaveMessage('字段微调已保存，后续重跑任务可以直接复用这份字典配置。')
    } catch (error) {
      console.error(error)
      setSaveMessage('保存失败，请检查 FastAPI 后端是否正常启动。')
    } finally {
      setIsSaving(false)
    }
  }

  if (!tasksLoaded) {
    return <div className="empty-state">正在载入运行监控数据…</div>
  }

  if (taskIds.length === 0) {
    return <div className="empty-state">当前还没有可监控任务。先跑一次预处理或建模任务，再回到这里查看。</div>
  }

  if (!bundle) {
    return <div className="empty-state">正在载入运行监控数据…</div>
  }

  return (
    <div className="page-layout">
      <div className="content-grid">
        <Panel
          title="当前监控任务"
          subtitle="后面会接任务中心筛选器和路由参数，这里先保留单任务深入查看模式。"
          toolbar={
            <select
              className="field-select"
              value={selectedTaskId}
              onChange={(event) => setSelectedTaskId(event.target.value)}
            >
              {taskIds.map((taskId) => (
                <option key={taskId} value={taskId}>
                  {taskId}
                </option>
              ))}
            </select>
          }
        >
          <div className="task-list">
            <article className="task-row">
              <div>
                <h4 className="task-title">{bundle.task.name}</h4>
                <div className="footer-note">{bundle.task.dataset}</div>
              </div>
              <div className="task-meta">
                <span>当前阶段：{bundle.task.stage}</span>
                <span>版本：{bundle.task.version}</span>
                <span>记录数：{bundle.task.records}</span>
              </div>
              <div className="task-side">
                <TaskStatusPill status={bundle.task.status} />
                <span>启动：{bundle.task.startedAt}</span>
                <span>耗时：{bundle.task.duration}</span>
              </div>
            </article>
          </div>

          <div className="timeline" style={{ marginTop: 18 }}>
            {bundle.timeline.map((step) => (
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
        </Panel>

        <Panel title="结构化日志" subtitle="先做成结构化日志面板，避免用户面对一大段原始文本无从下手。">
          <div className="log-list">
            {bundle.logs.map((log) => (
              <article
                className={`log-row ${log.level === '警告' ? 'warning' : log.level === '错误' ? 'error' : ''}`}
                key={log.id}
              >
                <div className="log-time">{log.time}</div>
                <div className="log-module">
                  {log.level}
                  <br />
                  {log.module}
                </div>
                <div className="log-message">
                  {log.message}
                  {log.field ? <div className="log-field">关联字段：{log.field}</div> : null}
                </div>
              </article>
            ))}
          </div>
        </Panel>
      </div>

      <div className="content-grid">
        <Panel title="字段微调面板" subtitle="这里是监控模块真正有价值的部分：看到问题后，用户可以直接修正字段定义。">
          <div className="tag-row" style={{ marginBottom: 16 }}>
            {draftFields.map((field) => (
              <button
                key={field.id}
                className="subtle-button"
                type="button"
                onClick={() => setSelectedFieldId(field.id)}
                style={{
                  background: field.id === selectedFieldId ? 'rgba(32, 84, 91, 0.12)' : undefined,
                  color: field.id === selectedFieldId ? '#1d4f57' : undefined,
                }}
              >
                {field.fieldName}
              </button>
            ))}
          </div>

          {selectedField ? (
            <>
              <div className="form-grid">
                <div className="field-block">
                  <label className="field-label">字段名称</label>
                  <input className="field-input" value={selectedField.fieldName} readOnly />
                </div>
                <div className="field-block">
                  <label className="field-label">基础类型</label>
                  <select
                    className="field-select"
                    value={selectedField.dataType}
                    onChange={(event) => updateDataType(event.target.value)}
                  >
                    {typeOptions.map((item) => (
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
                    value={selectedField.semanticType}
                    onChange={(event) => updateFieldDraft({ semanticType: event.target.value })}
                  >
                    {semanticOptions.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-block">
                  <label className="field-label">{unitLabel}</label>
                  <input
                    className="field-input"
                    value={selectedField.unit}
                    disabled={disableUnitAndRange}
                    onChange={(event) => updateFieldDraft({ unit: event.target.value })}
                  />
                </div>
                <div className="field-block">
                  <label className="field-label">{rangeLabel}</label>
                  <input
                    className="field-input"
                    value={selectedField.softRange}
                    disabled={disableUnitAndRange}
                    onChange={(event) => updateFieldDraft({ softRange: event.target.value })}
                  />
                </div>
                <div className="field-block">
                  <label className="field-label">字段备注</label>
                  <input
                    className="field-input"
                    value={selectedField.note}
                    onChange={(event) => updateFieldDraft({ note: event.target.value })}
                  />
                </div>
              </div>

              <div className="field-block" style={{ marginTop: 16 }}>
                <label className="field-label">本次修正说明</label>
                <textarea
                  className="field-textarea"
                  placeholder="例如：将“城市群名称”从普通分类调整为地理区域，并要求保留层级信息。"
                  value={revisionNotes}
                  onChange={(event) => setRevisionNotes(event.target.value)}
                />
              </div>

              <div className="field-actions">
                <span className="muted">
                  {disableUnitAndRange
                    ? '当前字段为非数值型，单位和范围已禁用。'
                    : '保存后会直接写入字段字典，后续重跑任务可以复用。'}
                </span>
                <div className="tag-row">
                  <button className="subtle-button" type="button">
                    基于当前配置重跑
                  </button>
                  <button className="primary-button" type="button" onClick={saveRevision} disabled={isSaving}>
                    {isSaving ? '正在保存…' : '保存本次微调'}
                  </button>
                </div>
              </div>

              {saveMessage ? <div className="footer-note">{saveMessage}</div> : null}
            </>
          ) : (
            <div className="empty-state">当前任务没有可微调字段。</div>
          )}
        </Panel>

        <Panel title="修正历史" subtitle="正式版本建议把这里接成字段级版本链与可回滚记录。">
          <div className="history-list">
            {bundle.revisions.map((revision) => (
              <article className="history-row" key={revision.id}>
                <strong>
                  {revision.time} · {revision.fieldName}
                </strong>
                <div>{revision.summary}</div>
                <div className="muted">执行主体：{revision.actor}</div>
              </article>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  )
}
