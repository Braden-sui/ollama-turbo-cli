import React from 'react'
import { appVersion, deleteSecret, getSecret, setSecret } from '../lib/ipc'

export default function DevSecretsPanel() {
  if (!import.meta.env.DEV) return null

  const [key, setKey] = React.useState('agent_desk_pro_test')
  const [val, setVal] = React.useState('secret-value')
  const [msg, setMsg] = React.useState('')
  const [version, setVersion] = React.useState('')

  React.useEffect(() => {
    appVersion().then(setVersion).catch(() => setVersion('unknown'))
  }, [])

  const onSet = async () => {
    try {
      await setSecret(key, val)
      setMsg(`set ok (key=${key}, value_len=${val.length})`)
    } catch (e) {
      setMsg(`set error: ${(e as Error).message}`)
    }
  }

  const onGet = async () => {
    try {
      const v = await getSecret(key)
      if (v == null) setMsg(`get: null for key=${key}`)
      else setMsg(`get ok (key=${key}, value_len=${v.length})`)
    } catch (e) {
      setMsg(`get error: ${(e as Error).message}`)
    }
  }

  const onDel = async () => {
    try {
      await deleteSecret(key)
      setMsg(`delete ok (key=${key})`)
    } catch (e) {
      setMsg(`delete error: ${(e as Error).message}`)
    }
  }

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-3 text-xs text-neutral-600 dark:text-neutral-300">
      <details>
        <summary className="cursor-pointer select-none">Dev Secrets Panel · version: {version}</summary>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <input
            aria-label="Key"
            className="min-w-[220px] rounded border border-black/10 dark:border-white/10 px-2 py-1"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="key"
          />
          <input
            aria-label="Value"
            className="min-w-[220px] rounded border border-black/10 dark:border-white/10 px-2 py-1"
            value={val}
            onChange={(e) => setVal(e.target.value)}
            placeholder="value"
          />
          <button type="button" onClick={onSet} className="rounded bg-neutral-200 px-2 py-1 hover:bg-neutral-300 dark:bg-neutral-700 dark:hover:bg-neutral-600">Set</button>
          <button type="button" onClick={onGet} className="rounded bg-neutral-200 px-2 py-1 hover:bg-neutral-300 dark:bg-neutral-700 dark:hover:bg-neutral-600">Get</button>
          <button type="button" onClick={onDel} className="rounded bg-neutral-200 px-2 py-1 hover:bg-neutral-300 dark:bg-neutral-700 dark:hover:bg-neutral-600">Delete</button>
          <div aria-live="polite" className="ml-4 text-[11px] text-neutral-500">{msg}</div>
        </div>
      </details>
    </div>
  )
}
