import { z } from 'zod'

// JSONL EVENT SCHEMA (source of truth from backend)
export const TurnStart = z.object({
  type: z.literal('turn.start'),
  trace_id: z.string(),
  session_id: z.string(),
  mode: z.string(),
  persona: z.string().optional(),
})

export const PlanPreview = z.object({
  type: z.literal('plan.preview'),
  steps: z.array(z.object({ id: z.string(), tool: z.string(), args_preview: z.string().optional() })),
  cost: z.object({ time_ms_est: z.number().int().nonnegative(), dollars_est: z.number().nonnegative() }),
})

export const StepStart = z.object({
  type: z.literal('step.start'),
  step_id: z.string(),
  tool: z.string(),
  args_preview: z.string().optional(),
})

export const StepEnd = z.object({
  type: z.literal('step.end'),
  step_id: z.string(),
  tool: z.string(),
  ms: z.number().int().nonnegative(),
  bytes: z.number().int().nonnegative(),
  cache: z.boolean().optional(),
  citations: z.array(z.object({ title: z.string(), url: z.string().url() })).optional(),
})

export const RiskEval = z.object({
  type: z.literal('risk.eval'),
  score: z.number().min(0).max(1),
  reasons: z.array(z.string()).optional(),
})

export const GateShow = z.object({
  type: z.literal('gate.show'),
  level: z.enum(['medium', 'high']),
  items: z.array(z.object({ step_id: z.string(), reason: z.string() })),
})

export const MemPropose = z.object({
  type: z.literal('mem.propose'),
  id: z.string(),
  text: z.string(),
  tags: z.array(z.string()).optional(),
})

export const MemSave = z.object({
  type: z.literal('mem.save'),
  id: z.string(),
})

export const FinalDelta = z.object({
  type: z.literal('final.delta'),
  chunk: z.string(),
})

export const FinalEnd = z.object({
  type: z.literal('final.end'),
  citations: z.array(z.object({ title: z.string(), url: z.string().url() })).optional(),
  receipts: z.array(z.string()).optional(),
})

export const ErrorEvt = z.object({
  type: z.literal('error'),
  scope: z.string(),
  message: z.string(),
  step_id: z.string().optional(),
})

export const AnyEvent = z.discriminatedUnion('type', [
  TurnStart, PlanPreview, StepStart, StepEnd, RiskEval, GateShow, MemPropose, MemSave, FinalDelta, FinalEnd, ErrorEvt,
])

export type AnyEvent = z.infer<typeof AnyEvent>

export function parseJsonlLine(line: string): AnyEvent | null {
  if (!line) return null
  try {
    const obj = JSON.parse(line)
    const evt = AnyEvent.safeParse(obj)
    if (evt.success) return evt.data
    // Malformed lines are skipped with a warning
    console.warn('[events] skipped malformed line', evt.error?.issues)
    return null
  } catch (e) {
    console.warn('[events] skipped malformed line (JSON)', (e as Error).message)
    return null
  }
}
