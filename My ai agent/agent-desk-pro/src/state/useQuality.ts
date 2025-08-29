import { create } from 'zustand'
import { QUALITY_POLICY, type Quality } from '../lib/constants'

interface QualityState {
  quality: Quality
  policy: (typeof QUALITY_POLICY)[Quality]
  setQuality: (q: Quality) => void
}

export const useQuality = create<QualityState>((set) => ({
  quality: 'balanced',
  policy: QUALITY_POLICY['balanced'],
  setQuality: (q) => set(() => ({ quality: q, policy: QUALITY_POLICY[q] })),
}))
