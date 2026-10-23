export type Mode = 'client' | 'agent'

export interface ToolCall {
  name: string
  input: unknown
  is_error: boolean
}

export interface ChatReply {
  conversation_id: string
  reply: string
  tool_calls: ToolCall[]
  status: string
  usage: string
}

export interface Customer {
  name: string
  email: string
  address: string
}

export interface Order {
  customer_id: string
  total: number
  status: string
  delivered_at: number | null
  items: { sku?: string; name: string; qty: number; unit_price: number }[]
  [field: string]: unknown
}

export interface ShopData {
  customers: Record<string, Customer>
  orders: Record<string, Order>
  refunds: { refund_id: string; order_id: string; amount: number }[]
  escalations: { ticket_id: string; customer_id: string | null }[]
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    const error = (await response.json().catch(() => ({}))) as { detail?: unknown }
    const detail = typeof error.detail === 'string' ? error.detail : `HTTP ${response.status}`
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export const fetchData = () => request<ShopData>('/api/data')
export const resetData = () => request<ShopData>('/api/reset', {})
export const sendMessage = (mode: Mode, message: string, conversationId: string | null) =>
  request<ChatReply>('/api/chat', { mode, message, conversation_id: conversationId })
