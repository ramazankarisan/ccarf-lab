import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import type { ShopData } from './api'

// Offline: fetch is replaced by a fake API that records every chat request.

const DATA: ShopData = {
  customers: {
    C001: { name: 'Maria Schmidt', email: 'maria.schmidt@example.com', address: 'Berlin' },
  },
  orders: {
    '12345': {
      customer_id: 'C001',
      total: 129.99,
      status: 'delivered',
      delivered_at: 1792368000,
      items: [{ name: 'Wireless Headphones', qty: 1, unit_price: 99.99 }],
    },
  },
  refunds: [],
  escalations: [],
}

let chatRequests: { mode: string; message: string; conversation_id: string | null }[]

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status }))
}

beforeEach(() => {
  chatRequests = []
  vi.stubGlobal(
    'fetch',
    vi.fn((path: string, init?: RequestInit) => {
      if (path === '/api/chat') {
        const body = JSON.parse(String(init?.body))
        chatRequests.push(body)
        if (body.message === 'fail') return json({ detail: 'The agent failed (APIError).' }, 502)
        return json({
          conversation_id: 'conv-1',
          reply: 'Please verify your **email**.',
          tool_calls: [{ name: 'lookup_order', input: { order_id: '12345' }, is_error: false }],
          status: 'completed',
          usage: '$0.0010',
        })
      }
      return json(DATA)
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

// The reply's **email** is rendered as <strong>, so match on the paragraph's full text.
const replyText = (_: string, element: Element | null) =>
  element?.tagName === 'P' && element.textContent === 'Please verify your email.'

async function say(text: string) {
  const user = userEvent.setup()
  await user.clear(screen.getByLabelText('Message'))
  await user.type(screen.getByLabelText('Message'), text)
  await user.click(screen.getByRole('button', { name: 'Send' }))
  return user
}

test('sends a message and shows the reply with its tool calls', async () => {
  render(<App />)

  await say('check my order #12345')

  expect(await screen.findByText(replyText)).toBeInTheDocument()
  expect(screen.getByText('email').tagName).toBe('STRONG')
  const tools = screen.getByRole('list', { name: 'Tool calls' })
  expect(within(tools).getByText('lookup_order')).toBeInTheDocument()
  expect(screen.getByText('completed · $0.0010')).toBeInTheDocument()
  expect(chatRequests[0]).toEqual({
    mode: 'agent',
    message: 'check my order #12345',
    conversation_id: null,
  })
})

test('later messages continue the same conversation', async () => {
  render(<App />)

  await say('hi')
  await screen.findByText(replyText)
  await say('maria.schmidt@example.com')

  expect(chatRequests[1].conversation_id).toBe('conv-1')
})

test('switching the mode starts a new conversation', async () => {
  render(<App />)
  const user = await say('hi')
  await screen.findByText(replyText)

  await user.click(screen.getByRole('button', { name: 'Client SDK' }))
  expect(screen.queryByText(replyText)).not.toBeInTheDocument()
  await say('hello')

  expect(chatRequests[1]).toEqual({ mode: 'client', message: 'hello', conversation_id: null })
})

test('a row opens to show the raw record as JSON and fills the message', async () => {
  render(<App />)
  const user = userEvent.setup()

  await user.click(await screen.findByText('#12345'))
  await user.click(screen.getByRole('button', { name: 'Ask about this order' }))

  expect(screen.getByText(/"delivered_at": 1792368000/)).toBeInTheDocument()
  expect(screen.getByLabelText('Message')).toHaveValue("What's the status of order #12345?")
})

test('selecting text in a row header does not toggle the row', async () => {
  render(<App />)
  const user = userEvent.setup()
  const title = await screen.findByText('#12345')
  vi.spyOn(window, 'getSelection').mockReturnValue({
    toString: () => '12345',
  } as Selection)

  await user.click(title)

  expect(screen.queryByText(/"delivered_at"/)).not.toBeInTheDocument()
})

test('the chevron toggles a row even while text is selected', async () => {
  render(<App />)
  const user = userEvent.setup()
  await screen.findByText('#12345')
  vi.spyOn(window, 'getSelection').mockReturnValue({ toString: () => 'x' } as Selection)
  const [, orderChevron] = screen.getAllByRole('button', { name: 'Expand' }) // customer, order

  await user.click(orderChevron)

  expect(screen.getByText(/"delivered_at"/)).toBeInTheDocument()
  expect(orderChevron).toHaveAttribute('aria-expanded', 'true')
})

test('an API error is shown in the chat', async () => {
  render(<App />)

  await say('fail')

  expect(await screen.findByText('The agent failed (APIError).')).toBeInTheDocument()
})
