import React from 'react'
import { useQuality } from '../state/useQuality'

export default function QualitySlider() {
  const { quality, setQuality } = useQuality()

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      // Avoid interfering with typing in inputs/editors
      const tag = (t?.tagName || '').toLowerCase()
      const isEditable = t?.isContentEditable || tag === 'input' || tag === 'textarea' || tag === 'select'
      if (isEditable) return
      if (!e.altKey) return
      if (e.key === '1') { setQuality('quick'); e.preventDefault() }
      else if (e.key === '2') { setQuality('balanced'); e.preventDefault() }
      else if (e.key === '3') { setQuality('rigorous'); e.preventDefault() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setQuality])

  const btn = (id: 'quick' | 'balanced' | 'rigorous', label: string) => {
    const active = quality === id
    const base = 'px-3 py-1 rounded-full text-sm border border-black/10 dark:border-white/10'
    const cls = active
      ? base + ' bg-[color:var(--accent)] text-white'
      : base
    return (
      <label key={id} className="inline-flex">
        <input
          type="radio"
          name="quality"
          value={id}
          checked={active}
          onChange={() => setQuality(id)}
          className="sr-only"
        />
        <span className={cls}>{label}</span>
      </label>
    )
  }

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-2 flex items-center gap-4">
      <div className="text-sm">Quality</div>
      <div className="flex items-center gap-2">
        {btn('quick', 'Quick')}
        {btn('balanced', 'Balanced')}
        {btn('rigorous', 'Rigorous')}
      </div>
      <div className="ml-auto text-xs text-neutral-500">Budget: time/$ preview</div>
    </div>
  )
}
