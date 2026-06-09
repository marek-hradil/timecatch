import { create } from 'zustand'
import type { SampleOut, SessionResponse } from './api/types.gen'
import { annotateApiAnnotatePost, createSessionApiSessionPost } from './api/sdk.gen'
import { client } from './api/client.gen'

// Configure the API client to use the same origin (works with Vite proxy in dev
// and direct FastAPI serving in production).
client.setConfig({ baseUrl: '' })

// Read URL params synchronously at store creation so the first render is correct.
const _params = new URLSearchParams(window.location.search)

export type AppScreen =
  | 'landing'        // URL param check + Start button
  | 'instructions'   // Per-combination instructions
  | 'annotating'     // Main annotation loop
  | 'complete'       // Done screen

interface AppState {
  screen: AppScreen
  prolificPid: string
  dataset: string
  task: string
  session: SessionResponse | null
  cursor: number
  error: string | null
  loading: boolean

  startSession: () => Promise<void>
  beginAnnotating: () => void
  submitAnswer: (answer: boolean | [number, number]) => Promise<void>
}

export const useStore = create<AppState>((set, get) => ({
  screen: 'landing',
  prolificPid: _params.get('PROLIFIC_PID') ?? '',
  dataset: _params.get('dataset') ?? '',
  task: _params.get('task') ?? '',
  session: null,
  cursor: 0,
  error: null,
  loading: false,

  startSession: async () => {
    const { prolificPid, dataset, task } = get()
    set({ loading: true, error: null })
    try {
      const response = await createSessionApiSessionPost({
        body: { prolific_pid: prolificPid, dataset, task },
      })
      if (response.error) {
        const detail = (response.error as { detail?: string }).detail ?? 'Failed to start session'
        set({ error: detail, loading: false })
        return
      }
      set({ session: response.data!, cursor: 0, screen: 'instructions', loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  beginAnnotating: () => {
    set({ screen: 'annotating' })
  },

  submitAnswer: async (answer: boolean | [number, number]) => {
    const { prolificPid, dataset, task, cursor, session } = get()
    if (!session) return

    set({ loading: true })
    try {
      const response = await annotateApiAnnotatePost({
        body: {
          prolific_pid: prolificPid,
          dataset,
          task,
          sample_index: cursor,
          human_answer: answer,
        },
      })
      if (response.error) {
        set({ loading: false })
        return
      }
      if (response.data!.completed) {
        set({ screen: 'complete', loading: false })
      } else {
        set({ cursor: cursor + 1, loading: false })
      }
    } catch (e) {
      set({ loading: false })
    }
  },
}))

export function currentSample(state: AppState): SampleOut | null {
  if (!state.session) return null
  return state.session.samples[state.cursor] ?? null
}
