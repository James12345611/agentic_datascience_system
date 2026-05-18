import { Suspense, lazy } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import './App.css'

const OverviewPage = lazy(async () => {
  const module = await import('./pages/OverviewPage')
  return { default: module.OverviewPage }
})

const WorkbenchPage = lazy(async () => {
  const module = await import('./pages/WorkbenchPage')
  return { default: module.WorkbenchPage }
})

const TaskCenterPage = lazy(async () => {
  const module = await import('./pages/TaskCenterPage')
  return { default: module.TaskCenterPage }
})

const MonitoringPage = lazy(async () => {
  const module = await import('./pages/MonitoringPage')
  return { default: module.MonitoringPage }
})

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div className="empty-state">正在加载页面资源…</div>}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/workbench" element={<WorkbenchPage />} />
            <Route path="/tasks" element={<TaskCenterPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}

export default App
