import { DEFAULT_BASE_URL } from './constants'
import type { AnyEvent } from './events'
import { getSecret } from './ipc'

export interface ChatControlTuning {
  k: number
  timeouts: { connectMs: number; readMs: number }
  parallelism: 'high' | 'limited' | 'conservative'
  citation_policy: 'optional' | 'preferred' | 'required'
  validation_level: 'off' | 'warn' | 'enforce'
}

export type OnEvent = (evt: AnyEvent) => void

export interface StartChatParams {
  baseUrl?: string
  query: string
  quality: 'quick' | 'balanced' | 'rigorous'
  tuning: ChatControlTuning
  signal?: AbortSignal
  onEvent: OnEvent
}

/**
 * Start a chat: try SSE first; if no events arrive, silently fallback to non-streaming.
 */
export async function startChat(params: StartChatParams): Promise<void> {
  const base = (params.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, '')
  // Map frontend tuning → backend ChatRequest reliability flags
  const payload: any = {
    message: params.query,
    // Prefer string tool results for simpler receipts; switch to 'object' when UI supports it
    options: { tool_results_format: 'string' },
    k: params.tuning.k,
    cite: params.tuning.citation_policy !== 'optional',
    check: params.tuning.validation_level,
    consensus: false,
  }

  let received = false
  const sseAbort = new AbortController()
  let userAbortHandler: (() => void) | null = null
  const connectMs = params.tuning?.timeouts?.connectMs ?? 3000
  let connectTimer: number | undefined

  const receipts: string[] = []
  let citations: { title: string; url: string }[] = []
  let gotToken = false

  // Helper to coerce tool_results → receipts strings
  const addToolResults = (tr: unknown) => {
    if (!Array.isArray(tr)) return
    for (const item of tr as any[]) {
      if (typeof item === 'string') receipts.push(item)
      else if (item && typeof item === 'object') receipts.push(JSON.stringify(item))
    }
  }

  // SSE streaming (POST body, Accept: text/event-stream)
  try {
    // Resolve API key if present in Dev Secrets (key = 'api_key')
    const apiKey = await getSecret('api_key').catch(() => null)
    const baseHeaders: Record<string, string> = { 'Content-Type': 'application/json', Accept: 'text/event-stream' }
    if (apiKey) baseHeaders['X-API-Key'] = apiKey
    if (params.signal) {
      if (params.signal.aborted) sseAbort.abort()
      else {
        userAbortHandler = () => sseAbort.abort()
        params.signal.addEventListener('abort', userAbortHandler, { once: true } as AddEventListenerOptions)
      }
    }
    connectTimer = window.setTimeout(() => { if (!received) sseAbort.abort() }, connectMs)

    const res = await fetch(`${base}/v1/chat/stream`, {
      method: 'POST',
      headers: baseHeaders,
      body: JSON.stringify(payload),
      signal: sseAbort.signal,
    })
    if (res.ok && res.body) {
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let eventName: string | null = null
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let idx: number
        while ((idx = buffer.indexOf('\n')) >= 0) {
          const raw = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 1)
          const line = raw.trim()
          if (!line) continue
          if (line.startsWith(':')) continue
          if (line.startsWith('event:')) { eventName = line.slice(6).trim() || null; continue }
          let payloadStr: string | null = null
          if (line.startsWith('data:')) payloadStr = line.slice(5).trim()
          else if (line[0] === '{') payloadStr = line
          if (!payloadStr) continue
          try {
            const obj = JSON.parse(payloadStr)
            // Backend emits {type:'token'|'final',...} and a separate summary event (event: summary)
            if (obj && obj.type === 'token' && typeof obj.content === 'string') {
              received = true
              gotToken = true
              if (connectTimer) { clearTimeout(connectTimer); connectTimer = undefined }
              params.onEvent({ type: 'final.delta', chunk: obj.content } as AnyEvent)
            } else if (obj && obj.type === 'final') {
              received = true
              if (connectTimer) { clearTimeout(connectTimer); connectTimer = undefined }
              if (!gotToken && typeof obj.content === 'string' && obj.content) {
                // Emit the final content as deltas when no tokens were streamed
                const chunks = String(obj.content).split(/(\r?\n)/).filter(Boolean)
                for (const chunk of chunks) params.onEvent({ type: 'final.delta', chunk } as AnyEvent)
              }
              if (obj.tool_results) addToolResults(obj.tool_results)
            } else if (eventName === 'summary' || (obj && typeof obj === 'object' && 'citations' in obj && !('type' in obj))) {
              // Summary with citations; normalize to {title,url}
              const cits = Array.isArray(obj.citations) ? obj.citations : []
              citations = cits.map((u: any) => {
                const s = String(u)
                return { title: s, url: s }
              })
            }
          } catch {
            // skip malformed lines
          }
        }
      }
    }
  } catch {
    // swallow per policy
  } finally {
    if (connectTimer) clearTimeout(connectTimer)
    if (params.signal && userAbortHandler) params.signal.removeEventListener('abort', userAbortHandler)
  }

  if (received) {
    // Synthesize final.end after stream completes
    params.onEvent({ type: 'final.end', citations, receipts } as AnyEvent)
    return
  }

  // Silent fallback to non-stream if nothing arrived
  try {
    const apiKey = await getSecret('api_key').catch(() => null)
    const baseHeaders: Record<string, string> = { 'Content-Type': 'application/json' }
    if (apiKey) baseHeaders['X-API-Key'] = apiKey
    const res = await fetch(`${base}/v1/chat`, {
      method: 'POST',
      headers: baseHeaders,
      body: JSON.stringify(payload),
      signal: params.signal,
    })
    if (!res.ok) return
    const data = await res.json().catch(() => null)
    if (!data) return
    const text: string | undefined = data.content
    if (Array.isArray(data.tool_results)) addToolResults(data.tool_results)
    if (text) {
      const chunks = String(text).split(/(\r?\n)/).filter(Boolean)
      for (const chunk of chunks) params.onEvent({ type: 'final.delta', chunk } as AnyEvent)
      params.onEvent({ type: 'final.end', citations, receipts } as AnyEvent)
    }
  } catch (_) {
    // swallow per policy
  }
}

