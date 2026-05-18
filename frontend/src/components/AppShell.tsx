import { Activity, Blocks, ChartSpline, LayoutDashboard, ShieldCheck } from 'lucide-react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

const navItems = [
  { to: '/overview', title: '平台总览', description: '看整体运行态势与风险项', icon: LayoutDashboard },
  { to: '/workbench', title: '工作台', description: '看模块进度与接口衔接状态', icon: Blocks },
  { to: '/tasks', title: '任务中心', description: '看任务池、状态和归档版本', icon: ChartSpline },
  { to: '/monitoring', title: '运行监控', description: '看步骤日志并做字段微调', icon: Activity },
] as const

const pageMeta: Record<string, { title: string; subtitle: string }> = {
  '/overview': {
    title: '数据科学平台前端基线',
    subtitle: '先把正式前端壳、路由和监控交互搭起来，后续再逐步切换到真实 API。',
  },
  '/workbench': {
    title: '模块工作台',
    subtitle: '这里更偏产品视角，用来对齐当前可用模块、接口空位和下一阶段研发拆分。',
  },
  '/tasks': {
    title: '任务中心',
    subtitle: '把预处理、统计诊断和建模统一到同一个任务池，后续接异步任务系统时可以直接复用。',
  },
  '/monitoring': {
    title: '运行监控与人工微调',
    subtitle: '日志不是只用来看报错，更重要的是让用户在看到异常时能直接修正字段信息并准备重跑。',
  },
}

export function AppShell() {
  const location = useLocation()
  const meta = pageMeta[location.pathname] ?? pageMeta['/overview']

  return (
    <div className="app-shell">
      <aside className="shell-sidebar">
        <div className="brand-block">
          <p className="brand-eyebrow">Agentic Data Science</p>
          <h1 className="brand-title">数策台</h1>
          <p className="brand-subtitle">
            面向中文场景的智能数据科学前端基线，当前聚焦任务监控、字段微调和结果工作流。
          </p>
        </div>

        <div>
          <p className="sidebar-section-title">主导航</p>
          <nav className="nav-group">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
                >
                  <Icon size={18} />
                  <span className="nav-copy">
                    <span className="nav-title">{item.title}</span>
                    <span className="nav-description">{item.description}</span>
                  </span>
                </NavLink>
              )
            })}
          </nav>
        </div>

        <div className="sidebar-health">
          <p className="sidebar-section-title">当前状态</p>
          <div className="health-list">
            <div className="health-item">
              <span className="health-label">前端模式</span>
              <span className="health-value">React 原型</span>
            </div>
            <div className="health-item">
              <span className="health-label">后端接口</span>
              <span className="health-value">待 FastAPI 接入</span>
            </div>
            <div className="health-item">
              <span className="health-label">任务轨迹</span>
              <span className="health-value">已纳入设计</span>
            </div>
            <div className="health-item">
              <span className="health-label">人工微调</span>
              <span className="health-value">前端交互就绪</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="shell-main">
        <header className="shell-topbar">
          <div>
            <h2 className="topbar-title">{meta.title}</h2>
            <p className="topbar-subtitle">{meta.subtitle}</p>
          </div>
          <div className="topbar-actions">
            <span className="topbar-badge">
              <ShieldCheck size={14} style={{ marginRight: 6, verticalAlign: 'text-top' }} />
              原型环境
            </span>
            <span className="topbar-badge">接口层未接入</span>
            <button className="topbar-button" type="button">
              规划下一轮 API
            </button>
          </div>
        </header>

        <Outlet />
      </main>
    </div>
  )
}
