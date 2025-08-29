import React from 'react'

interface Props { receipts?: string[]; citations?: { title: string; url: string }[] }

export default function ReceiptsDrawer({ receipts, citations }: Props) {
  const items = receipts ?? []
  const cites = citations ?? []
  return (
    <div className="px-4 py-4">
      <h2 className="text-sm font-semibold mb-2">Receipts</h2>
      <div className="space-y-2">
        {items.length === 0 ? (
          <div className="rounded-[14px] border border-black/10 dark:border-white/10 p-3 text-sm text-neutral-500">
            No receipts yet.
          </div>
        ) : (
          items.map((r, i) => (
            <div key={i} className="rounded-[14px] border border-black/10 dark:border-white/10 p-3 text-xs break-all">
              {r}
            </div>
          ))
        )}
      </div>
      <h2 className="text-sm font-semibold mb-2 mt-4">Citations</h2>
      <div className="space-y-2">
        {cites.length === 0 ? (
          <div className="rounded-[14px] border border-black/10 dark:border-white/10 p-3 text-sm text-neutral-500">
            No citations.
          </div>
        ) : (
          cites.map((c, i) => (
            <a
              key={i}
              className="block rounded-[14px] border border-black/10 dark:border-white/10 p-3 text-xs break-words hover:bg-black/5 dark:hover:bg-white/5"
              href={c.url}
              target="_blank"
              rel="noreferrer noopener"
            >
              <div className="font-medium">{c.title}</div>
              <div className="text-neutral-500">{c.url}</div>
            </a>
          ))
        )}
      </div>
    </div>
  )
}
