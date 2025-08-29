import React from 'react'
import CommandBar from '../components/CommandBar'
import AnswerView from '../components/AnswerView'
import ReceiptsDrawer from '../components/ReceiptsDrawer'
import QualitySlider from '../components/QualitySlider'
import DevSecretsPanel from '../components/DevSecretsPanel'
import { startChat } from '../lib/api'
import type { AnyEvent } from '../lib/events'
import { useQuality } from '../state/useQuality'

export default function App() {
  const [text, setText] = React.useState('')
  const [receipts, setReceipts] = React.useState<string[]>([])
  const [citations, setCitations] = React.useState<{ title: string; url: string }[]>([])
  const abortRef = React.useRef<AbortController | null>(null)
  const { quality, policy } = useQuality()

  const handleSubmit = React.useCallback((q: string) => {
    // cancel previous
    abortRef.current?.abort()
    setText('')
    setReceipts([])
    setCitations([])
    const ac = new AbortController()
    abortRef.current = ac
    startChat({
      query: q,
      quality,
      tuning: policy,
      signal: ac.signal,
      onEvent: (evt: AnyEvent) => {
        if (evt.type === 'final.delta') setText((prev) => prev + evt.chunk)
        else if (evt.type === 'final.end') {
          setReceipts(evt.receipts ?? [])
          setCitations(evt.citations ?? [])
        }
      },
    })
  }, [quality, policy])

  React.useEffect(() => () => abortRef.current?.abort(), [])

  return (
    <div className="min-h-screen grid grid-rows-[auto_1fr_auto] grid-cols-[1fr_auto]">
      {/* Command Bar (top) */}
      <header className="row-start-1 col-span-2 border-b border-black/10 dark:border-white/10">
        <CommandBar onSubmit={handleSubmit} />
        <DevSecretsPanel />
      </header>

      {/* Answer View (center) */}
      <main className="row-start-2 col-start-1 overflow-auto">
        <AnswerView text={text} />
      </main>

      {/* Receipts Drawer (right) */}
      <aside className="row-start-2 col-start-2 w-[360px] border-l border-black/10 dark:border-white/10 overflow-auto">
        <ReceiptsDrawer receipts={receipts} citations={citations} />
      </aside>

      {/* Quality Slider (bottom) */}
      <footer className="row-start-3 col-span-2 border-t border-black/10 dark:border-white/10">
        <QualitySlider />
      </footer>
    </div>
  )
}
