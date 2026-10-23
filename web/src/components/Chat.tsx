import { useEffect, useRef, type FormEvent } from 'react'
import type { ToolCall } from '../api'

export interface ChatMessage {
  role: 'customer' | 'agent' | 'error'
  text: string
  toolCalls?: ToolCall[]
  meta?: string
}

// The model writes **bold** markdown; render just that, without a markdown library.
function withBold(text: string) {
  return text
    .split(/\*\*(.+?)\*\*/g)
    .map((part, i) => (i % 2 === 1 ? <strong key={i}>{part}</strong> : part))
}

interface ChatProps {
  messages: ChatMessage[]
  draft: string
  busy: boolean
  onDraftChange: (draft: string) => void
  onSend: () => void
}

export function Chat({ messages, draft, busy, onDraftChange, onSend }: ChatProps) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView?.({ block: 'end' })
  }, [messages, busy])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (draft.trim() && !busy) onSend()
  }

  return (
    <section className="chat" aria-label="Chat">
      <div className="messages">
        {messages.length === 0 && (
          <p className="empty">Write as the customer, or pick an order or customer on the right.</p>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.toolCalls && message.toolCalls.length > 0 && (
              <ul className="tools" aria-label="Tool calls">
                {message.toolCalls.map((call, i) => (
                  <li key={i} className={call.is_error ? 'failed' : undefined}>
                    <details>
                      <summary>
                        <code>{call.name}</code>
                        <span className="tag">{call.is_error ? 'error' : 'ok'}</span>
                      </summary>
                      <pre>{JSON.stringify(call.input, null, 2)}</pre>
                    </details>
                  </li>
                ))}
              </ul>
            )}
            <p>{withBold(message.text)}</p>
            {message.meta && <small>{message.meta}</small>}
          </div>
        ))}
        {busy && <p className="thinking">Agent is working…</p>}
        <div ref={end} />
      </div>
      <form className="composer" onSubmit={submit}>
        <input
          aria-label="Message"
          placeholder="e.g. check my order #12345"
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          maxLength={2000}
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
    </section>
  )
}
