import { parseJsonlLine, type AnyEvent } from './events'

export interface StreamOptions {
  signal?: AbortSignal
  headers?: Record<string, string>
}

export type EventHandler = (evt: AnyEvent) => void

/**
 * Stream newline-delimited JSON (JSONL) via fetch. Each line is parsed into an event and dispatched.
 * Silent failure: on network/parse errors, this resolves and lets callers handle fallback.
 */
export async function streamJsonl(url: string, onEvent: EventHandler, opts: StreamOptions = {}): Promise<void> {
  try {
    const res = await fetch(url, { method: 'GET', headers: opts.headers, signal: opts.signal })
    if (!res.ok || !res.body) return
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

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
        // SSE compatibility: strip 'data:' prefix; ignore comments/keepalives starting with ':'
        if (line.startsWith(':')) continue
        const payload = line.startsWith('data:') ? line.slice(5).trim() : line
        if (!payload || payload[0] !== '{') continue
        const evt = parseJsonlLine(payload)
        if (evt) onEvent(evt)
      }
    }
    if (buffer.length > 0) {
      const tail = buffer.trim()
      if (tail && tail[0] === '{') {
        const evt = parseJsonlLine(tail)
        if (evt) onEvent(evt)
      }
    }
  } catch (_) {
    // Silent per product policy; caller may do non-stream fallback
    return
  }
}
