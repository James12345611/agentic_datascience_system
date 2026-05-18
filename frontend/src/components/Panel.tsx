import type { PropsWithChildren, ReactNode } from 'react'

interface PanelProps extends PropsWithChildren {
  title: string
  subtitle?: string
  toolbar?: ReactNode
}

export function Panel({ title, subtitle, toolbar, children }: PanelProps) {
  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h3 className="panel-title">{title}</h3>
          {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
        </div>
        {toolbar ? <div className="panel-toolbar">{toolbar}</div> : null}
      </header>
      {children}
    </section>
  )
}
