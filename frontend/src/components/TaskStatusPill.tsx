import clsx from 'clsx'
import type { TaskStatus } from '../types/platform'

const statusLabelMap: Record<TaskStatus, string> = {
  completed: '已完成',
  running: '运行中',
  failed: '失败',
  queued: '排队中',
}

interface TaskStatusPillProps {
  status: TaskStatus
}

export function TaskStatusPill({ status }: TaskStatusPillProps) {
  return <span className={clsx('pill', status)}>{statusLabelMap[status]}</span>
}
