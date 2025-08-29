import React from 'react'

interface Props { text?: string }

export default function AnswerView({ text }: Props) {
  const hasText = !!(text && text.length > 0)
  return (
    <div className="mx-auto max-w-[900px] px-6 py-6">
      <h1 className="text-lg font-medium mb-2">Answer</h1>
      <div className="min-h-[240px] rounded-[14px] border border-black/10 dark:border-white/10 p-4">
        {hasText ? (
          <pre className="whitespace-pre-wrap break-words text-[15px] leading-7">{text}</pre>
        ) : (
          <p className="text-neutral-500">Streaming output will appear here…</p>
        )}
      </div>
    </div>
  )
}
