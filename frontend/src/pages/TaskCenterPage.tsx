import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Panel } from '../components/Panel'
import { TaskStatusPill } from '../components/TaskStatusPill'
import { fetchTaskCenterSnapshot } from '../services/platformService'
import type { TaskCenterSnapshot } from '../types/platform'

export function TaskCenterPage() {
  const [snapshot, setSnapshot] = useState<TaskCenterSnapshot | null>(null)

  useEffect(() => {
    void fetchTaskCenterSnapshot().then(setSnapshot)
  }, [])

  if (!snapshot) {
    return <div className="empty-state">正在载入任务中心…</div>
  }

  return (
    <div className="page-layout">
      <div className="content-grid">
        <Panel title="任务池全景" subtitle="正式产品里这一页会支持筛选、检索、排序、重跑和结果跳转。">
          <div className="task-list">
            {snapshot.tasks.map((task) => (
              <article className="task-row" key={task.id}>
                <div>
                  <h4 className="task-title">{task.name}</h4>
                  <div className="footer-note">{task.id} · {task.dataset}</div>
                </div>
                <div className="task-meta">
                  <span>任务类型：{task.type}</span>
                  <span>当前阶段：{task.stage}</span>
                  <span>记录规模：{task.records}</span>
                </div>
                <div className="task-side">
                  <TaskStatusPill status={task.status} />
                  <span>启动时间：{task.startedAt}</span>
                  <span>耗时：{task.duration}</span>
                </div>
              </article>
            ))}
          </div>
        </Panel>

        <Panel title="任务状态结构" subtitle="把任务池压成结构图，便于后面加入运维和资源视角。">
          <ReactECharts
            option={{
              tooltip: { trigger: 'item' },
              legend: { bottom: 0, textStyle: { color: '#465a61' } },
              series: [
                {
                  type: 'pie',
                  radius: ['44%', '68%'],
                  center: ['50%', '46%'],
                  label: { formatter: '{b}\n{c}' },
                  data: snapshot.statusStats.map((item, index) => ({
                    value: item.value,
                    name: item.label,
                    itemStyle: { color: ['#2f7d71', '#244d58', '#b85f3c', '#d08e44'][index] },
                  })),
                },
              ],
            }}
            style={{ height: 320 }}
          />
        </Panel>
      </div>

      <div className="content-grid equal">
        <Panel title="阶段分布" subtitle="后面可以进一步拆成每个 agent 节点的运行耗时和失败率。">
          <ReactECharts
            option={{
              tooltip: { trigger: 'axis' },
              grid: { left: 10, right: 10, bottom: 12, top: 12, containLabel: true },
              xAxis: { type: 'category', data: snapshot.stageStats.map((item) => item.label), axisLabel: { interval: 0, rotate: 18 } },
              yAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(70, 90, 97, 0.08)' } } },
              series: [
                { type: 'bar', data: snapshot.stageStats.map((item) => item.value), itemStyle: { color: '#2f7d71', borderRadius: [10, 10, 0, 0] }, barWidth: 26 },
              ],
            }}
            style={{ height: 280 }}
          />
        </Panel>

        <Panel title="下一步建议" subtitle="从原型往正式平台迁移时，任务中心会是最先沉淀稳定的数据结构。">
          <div className="simple-list">
            <div className="simple-row">
              <strong>1. 把 task_run 从当前 SQLite 迁移到 PostgreSQL</strong>
              <span className="muted">这样 React 前端就可以稳定拉取分页任务列表和筛选结果。</span>
            </div>
            <div className="simple-row">
              <strong>2. 为任务补 task_step_log 和 config_snapshot</strong>
              <span className="muted">这两张表是运行监控和人工微调的真正基础。</span>
            </div>
            <div className="simple-row">
              <strong>3. 把重跑、回滚、结果跳转做成统一动作</strong>
              <span className="muted">用户不应在不同模块中学习不同操作逻辑。</span>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
