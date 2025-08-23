import { invoke } from '@tauri-apps/api/core'

// Lightweight wrapper around Tauri commands with a dev fallback when not running inside Tauri
// Non-production fallback stores values in-memory only (cleared on reload)
const mem = new Map<string, string>()

function isTauri(): boolean {
  return typeof window !== 'undefined' && typeof (window as any).__TAURI__ !== 'undefined'
}

export async function getSecret(key: string): Promise<string | null> {
  if (isTauri()) {
    const val = await invoke<string | null>('get_secret', { key })
    return val ?? null
  }
  return mem.get(key) ?? null
}

export async function setSecret(key: string, value: string): Promise<void> {
  if (isTauri()) {
    await invoke('set_secret', { key, value })
    return
  }
  mem.set(key, value)
}

export async function deleteSecret(key: string): Promise<void> {
  if (isTauri()) {
    await invoke('delete_secret', { key })
    return
  }
  mem.delete(key)
}

export async function appVersion(): Promise<string> {
  if (isTauri()) {
    const res = await invoke<{ version: string }>('version')
    return res.version
  }
  return 'dev'
}
