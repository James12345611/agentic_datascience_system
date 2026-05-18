import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Panel } from '../components/Panel'
import { StatCard } from '../components/StatCard'
import { TaskStatusPill } from '../components/TaskStatusPill'
import { fetchDashboardSnapshot } from '../services/platformService'
import type { DashboardSnapshot } from '../types/platform'

export function OverviewPage() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null)

  useEffect(() => {
    void fetchDashboardSnapshot().then(setSnapshot)
  }, [])

  if (!snapshot) {
    return <div className="empty-state">正在加载前端总览数据…</div>
  }

  return (
    <div className="page-layout">
      <section className="hero-band">
        <h3 className="hero-title">{snapshot.headline}</h3>
        <p className="hero-copy">{snapshot.subtitle}</p>
        <div className="hero-inline-stats">
          {snapshot.heroStats.map((item) => (
            <div className="hero-stat" key={item.label}>
              <div className="hero-stat-value">{item.value}</div>
              <div className="hero-stat-label">{item.label}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="stats-grid">
        {snapshot.summaryCards.map((card) => (
          <StatCard key={card.title} {...card} />
        ))}
      </div>

      <div className="content-grid">
        <Panel title="最近 7 日任务走势" subtitle="先把运行趋势、预处理量和建模量可视化，后面接真实任务统计接口。">
          <ReactECharts
            option={{
              tooltip: { trigger: 'axis' },
              legend: { top: 0, textStyle: { color: '#465a61' } },
              grid: { left: 8, right: 12, bottom: 12, top: 44, containLabel: true },
              xAxis: {
                type: 'category',
                data: snapshot.trend.map((item) => item.label),
                axisLine: { lineStyle: { color: '#bed0d2' } },
              },
              yAxis: {
                type: 'value',
                splitLine: { lineStyle: { color: 'rgba(70, 90, 97, 0.08)' } },
              },
              series: [
                { name: '总任务量', type: 'line', smooth: true, data: snapshot.trend.map((item) => item.total), lineStyle: { color: '#1f5d66', width: 3 }, areaStyle: { color: 'rgba(31, 93, 102, 0.12)' }, itemStyle: { color: '#1f5d66' } },
                { name: '预处理', type: 'bar', data: snapshot.trend.map((item) => item.preprocess), itemStyle: { color: '#2f7d71', borderRadius: [8, 8, 0, 0] } },
                { name: '建模', type: 'bar', data: snapshot.trend.map((item) => item.modeling), itemStyle: { color: '#d08e44', borderRadius: [8, 8, 0, 0] } },
              ],
            }}
            style={{ height: 320 }}
          />
        </Panel>

        <Panel title="当前关注风险" subtitle="这些项后面都会直接连到运行监控和字段微调面板。">
          <div className="simple-list">
            {snapshot.alerts.map((alert) => (
              <div className="simple-row" key={alert.title}>
                <strong>
                  [{alert.severity}优先级] {alert.title}
                </strong>
                <span className="muted">{alert.detail}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="content-grid equal">
        <Panel title="模型表现对比" subtitle="这一块后续可直接接真实评估指标与模型版本。">
          <ReactECharts
            option={{
              tooltip: { trigger: 'axis' },
              grid: { left: 8, right: 12, bottom: 12, top: 18, containLabel: true },
              xAxis: { type: 'value', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(70, 90, 97, 0.08)' } } },
              yAxis: { type: 'category', data: snapshot.ranking.map((item) => item.label), axisLine: { show: false } },
              series: [
                {
                  type: 'bar',
                  data: snapshot.ranking.map((item) => item.score),
                  color: ['#244d58', '#2f7d71', '#d08e44', '#8a9ca4'],
                  label: { show: true, position: 'right', formatter: '{c}' },
                  barWidth: 18,
                },
              ],
            }}
            style={{ height: 280 }}
          />
          <p className="footer-note">建议后续把不同模型族的指标拆分成回归和分类两套面板。</p>
        </Panel>

        <Panel title="最近关键任务" subtitle="任务中心已经是统一入口，预处理与建模不再分裂展示。">
          <div className="task-list">
            {snapshot.tasks.map((task) => (
              <article className="task-row" key={task.id}>
                <div>
                  <h4 className="task-title">{task.name}</h4>
                  <div className="footer-note">{task.dataset} · {task.records} 条记录 · {task.version}</div>
                </div>
                <div className="task-meta">
                  <span>当前阶段：{task.stage}</span>
                  <span>启动时间：{task.startedAt}</span>
                  <span>发起主体：{task.owner}</span>
                </div>
                <div className="task-side">
                  <TaskStatusPill status={task.status} />
                  <span>耗时：{task.duration}</span>
                  <span>告警数：{task.alerts}</span>
                </div>
              </article>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  )
}
