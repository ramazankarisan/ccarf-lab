import { useState, type ReactNode } from 'react'
import type { ShopData } from '../api'

interface DataPanelProps {
  data: ShopData | null
  onAsk: (text: string) => void
  onReset: () => void
}

const day = (timestamp: number | null) =>
  timestamp === null ? null : new Date(timestamp * 1000).toISOString().slice(0, 10)

interface RowProps {
  title: ReactNode
  extra?: ReactNode
  record: unknown
  action: { label: string; onClick: () => void }
}

function Row({ title, extra, record, action }: RowProps) {
  const [open, setOpen] = useState(false)

  // The header toggles too, but not when the click ends a text selection,
  // so IDs and names can be selected and copied.
  function toggleUnlessSelecting() {
    if (!window.getSelection()?.toString()) setOpen((current) => !current)
  }

  return (
    <div className={`row${open ? ' open' : ''}`}>
      <div className="row-head" onClick={toggleUnlessSelecting}>
        <button
          className="chevron"
          aria-expanded={open}
          aria-label={open ? 'Collapse' : 'Expand'}
          onClick={(event) => {
            event.stopPropagation()
            setOpen(!open)
          }}
        >
          ›
        </button>
        <span className="row-title">{title}</span>
        {extra}
      </div>
      {open && (
        <div className="row-body">
          <pre>{JSON.stringify(record, null, 2)}</pre>
          <button className="ghost" onClick={action.onClick}>
            {action.label}
          </button>
        </div>
      )}
    </div>
  )
}

export function DataPanel({ data, onAsk, onReset }: DataPanelProps) {
  if (!data) return <aside className="data">Loading data…</aside>

  return (
    <aside className="data" aria-label="Shop data">
      <header>
        <h2>Shop data</h2>
        <button
          className="ghost"
          onClick={onReset}
          title="Restore the seed data, clear refunds and tickets, and end all chats"
        >
          Reset
        </button>
      </header>

      <h3>Customers</h3>
      {Object.entries(data.customers).map(([id, customer]) => (
        <Row
          key={id}
          title={
            <>
              <span className="id">{id}</span> {customer.name}
            </>
          }
          record={{ customer_id: id, ...customer }}
          action={{
            label: 'Use this email',
            onClick: () => onAsk(`My email is ${customer.email}`),
          }}
        />
      ))}

      <h3>Orders</h3>
      {Object.entries(data.orders).map(([id, order]) => (
        <Row
          key={id}
          title={
            <>
              <span className="id">#{id}</span> ${order.total.toFixed(2)}
            </>
          }
          extra={<span className={`status ${order.status}`}>{order.status}</span>}
          record={{
            order_id: id,
            customer_id: order.customer_id,
            status: order.status,
            total: order.total,
            delivered: day(order.delivered_at),
            items: order.items.map(({ name, qty, unit_price }) => ({ name, qty, unit_price })),
          }}
          action={{
            label: 'Ask about this order',
            onClick: () => onAsk(`What's the status of order #${id}?`),
          }}
        />
      ))}

      {data.refunds.length > 0 && (
        <>
          <h3>Refunds</h3>
          <pre className="list">{JSON.stringify(data.refunds, null, 2)}</pre>
        </>
      )}

      {data.escalations.length > 0 && (
        <>
          <h3>Tickets</h3>
          <pre className="list">{JSON.stringify(data.escalations, null, 2)}</pre>
        </>
      )}
    </aside>
  )
}
