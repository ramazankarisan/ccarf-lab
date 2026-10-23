import { useEffect, useState } from 'react'
import { fetchData, resetData, sendMessage, type Mode, type ShopData } from './api'
import { Chat, type ChatMessage } from './components/Chat'
import { DataPanel } from './components/DataPanel'

const MODES: { id: Mode; label: string; note: string }[] = [
  {
    id: 'client',
    label: 'Client SDK',
    note: 'loop.py: the loop written by hand. Same prompt, but no hooks, so the rules are only requests. Watch for: a $650 refund going through, internal fields and raw timestamps, switching accounts mid-chat.',
  },
  {
    id: 'agent',
    label: 'Agent SDK',
    note: 'session.py: hooks enforce the refund limit and hide other customers’ orders; case facts keep IDs and amounts exact.',
  },
]

export default function App() {
  const [mode, setMode] = useState<Mode>('agent')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [data, setData] = useState<ShopData | null>(null)

  const refresh = () => fetchData().then(setData, () => setData(null))
  useEffect(() => {
    refresh()
  }, [])

  function newChat(nextMode: Mode) {
    setMode(nextMode)
    setConversationId(null)
    setMessages([])
  }

  async function send() {
    const text = draft.trim()
    setDraft('')
    setMessages((current) => [...current, { role: 'customer', text }])
    setBusy(true)
    try {
      const reply = await sendMessage(mode, text, conversationId)
      setConversationId(reply.conversation_id)
      setMessages((current) => [
        ...current,
        {
          role: 'agent',
          text: reply.reply || '(no reply)',
          toolCalls: reply.tool_calls,
          meta: `${reply.status} · ${reply.usage}`,
        },
      ])
      await refresh() // refunds and tickets change the data
    } catch (error) {
      const text = error instanceof Error ? error.message : 'Something went wrong.'
      setMessages((current) => [...current, { role: 'error', text }])
    } finally {
      setBusy(false)
    }
  }

  async function reset() {
    setData(await resetData())
    newChat(mode)
  }

  const current = MODES.find((m) => m.id === mode)!

  return (
    <main className="layout">
      <div className="left">
        <header className="top">
          <h1>Support agent</h1>
          <div className="switch" role="group" aria-label="Agent mode">
            {MODES.map((m) => (
              <button
                key={m.id}
                aria-pressed={m.id === mode}
                onClick={() => m.id !== mode && newChat(m.id)}
                disabled={busy}
              >
                {m.label}
              </button>
            ))}
          </div>
        </header>
        <p className="note">{current.note}</p>
        <Chat
          messages={messages}
          draft={draft}
          busy={busy}
          onDraftChange={setDraft}
          onSend={send}
        />
      </div>
      <DataPanel data={data} onAsk={setDraft} onReset={reset} />
    </main>
  )
}
