import React, { useCallback, useRef } from 'react'

interface Props {
  onSubmit?: (query: string) => void
}

export default function CommandBar({ onSubmit }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      // TODO: open command palette
      console.log('[CommandBar] Open command palette')
    }
  }, [])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const q = inputRef.current?.value?.trim()
    if (!q) return
    if (onSubmit) onSubmit(q)
    else console.log('[CommandBar] submit:', q)
    if (inputRef.current) inputRef.current.value = ''
  }, [onSubmit])

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-[1400px] px-4 py-2 flex items-center gap-3">
      <div className="text-sm text-neutral-500">Command Bar</div>
      <input
        ref={inputRef}
        aria-label="Command Bar"
        onKeyDown={onKeyDown}
        className="flex-1 rounded-[14px] border border-black/10 dark:border-white/10 px-3 py-2 outline-none focus:ring-2 focus:ring-[color:var(--accent)]"
        placeholder="Ask or type a command… (Ctrl+K for palette)"
      />
    </form>
  )
}
