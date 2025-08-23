export const DEFAULT_BASE_URL = (import.meta as any)?.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8765'

export type Quality = 'quick' | 'balanced' | 'rigorous'

export const QUALITY_POLICY: Record<Quality, {
  k: number
  timeouts: { connectMs: number; readMs: number }
  parallelism: 'high' | 'limited' | 'conservative'
  citation_policy: 'optional' | 'preferred' | 'required'
  validation_level: 'off' | 'warn' | 'enforce'
}> = {
  quick: {
    k: 1,
    timeouts: { connectMs: 2000, readMs: 12000 },
    parallelism: 'high',
    citation_policy: 'optional',
    validation_level: 'off',
  },
  balanced: {
    k: 3,
    timeouts: { connectMs: 4000, readMs: 20000 },
    parallelism: 'limited',
    citation_policy: 'preferred',
    validation_level: 'warn',
  },
  rigorous: {
    k: 5,
    timeouts: { connectMs: 8000, readMs: 40000 },
    parallelism: 'conservative',
    citation_policy: 'required',
    validation_level: 'enforce',
  },
}

export const STREAM_COALESCE_MS = 30
