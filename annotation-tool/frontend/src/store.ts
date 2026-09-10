import { create } from 'zustand'
import type { SampleOut, SessionResponse } from './api/types.gen'
import { annotateApiAnnotatePost, createSessionApiSessionPost, sessionMetricsApiSessionMetricsPost } from './api/sdk.gen'
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
  sceneId: string
  studyId: string | undefined
  sessionId: string | undefined
  session: SessionResponse | null
  cursor: number
  answers: Record<number, boolean | [number, number]>
  error: string | null
  loading: boolean

  // Observability
  instructionsStartedAt: number | null
  sampleDisplayedAt: number | null
  currentViewerOpens: number

  startSession: () => Promise<void>
  beginAnnotating: (metrics: { videoPlayed: boolean; instructionLightboxOpens: number }) => void
  submitAnswer: (answer: boolean | [number, number]) => Promise<void>
  goBack: () => void
  goForward: () => void
  incrementViewerOpens: () => void
}

export const useStore = create<AppState>((set, get) => ({
  screen: 'landing',
  prolificPid: _params.get('PROLIFIC_PID') ?? '',
  dataset: _params.get('dataset') ?? '',
  task: _params.get('task') ?? '',
  sceneId: _params.get('SCENE_ID') ?? 'main',
  studyId: _params.get('STUDY_ID') ?? undefined,
  sessionId: _params.get('SESSION_ID') ?? undefined,
  session: null,
  cursor: 0,
  answers: {},
  error: null,
  loading: false,

  instructionsStartedAt: null,
  sampleDisplayedAt: null,
  currentViewerOpens: 0,

  startSession: async () => {
    const { prolificPid, dataset, task, sceneId, studyId, sessionId } = get()
    set({ loading: true, error: null })
    try {
      const response = await createSessionApiSessionPost({
        body: { prolific_pid: prolificPid, dataset, task, scene_id: sceneId, study_id: studyId, session_id: sessionId },
      })
      if (response.error) {
        const detail = (response.error as { detail?: string }).detail ?? 'Failed to start session'
        set({ error: detail, loading: false })
        return
      }
      set({
        session: response.data!, cursor: 0, answers: {}, screen: 'instructions', loading: false,
        instructionsStartedAt: Date.now(),
      })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  beginAnnotating: ({ videoPlayed, instructionLightboxOpens }) => {
    const { prolificPid, dataset, task, sceneId, instructionsStartedAt } = get()
    const instructionsDurationS = instructionsStartedAt ? (Date.now() - instructionsStartedAt) / 1000 : null
    sessionMetricsApiSessionMetricsPost({
      body: {
        prolific_pid: prolificPid,
        dataset,
        task,
        scene_id: sceneId,
        instructions_duration_s: instructionsDurationS,
        instructions_lightbox_opens: instructionLightboxOpens,
        video_played: videoPlayed,
      },
    }).catch(() => {})
    set({ screen: 'annotating', sampleDisplayedAt: Date.now(), currentViewerOpens: 0 })
  },

  goBack: () => {
    const { cursor } = get()
    if (cursor > 0) set({ cursor: cursor - 1, sampleDisplayedAt: Date.now(), currentViewerOpens: 0 })
  },

  goForward: () => {
    const { cursor, session, answers } = get()
    const total = session?.total ?? 0
    if (answers[cursor] !== undefined && cursor < total - 1) {
      set({ cursor: cursor + 1, sampleDisplayedAt: Date.now(), currentViewerOpens: 0 })
    }
  },

  submitAnswer: async (answer: boolean | [number, number]) => {
    const { prolificPid, dataset, task, sceneId, cursor, session, sampleDisplayedAt, currentViewerOpens } = get()
    if (!session) return

    const viewDurationS = sampleDisplayedAt ? (Date.now() - sampleDisplayedAt) / 1000 : undefined

    set({ loading: true })
    try {
      const response = await annotateApiAnnotatePost({
        body: {
          prolific_pid: prolificPid,
          dataset,
          task,
          sample_index: cursor,
          human_answer: answer,
          scene_id: sceneId,
          view_duration_s: viewDurationS,
          viewer_opens: currentViewerOpens,
        },
      })
      if (response.error) {
        set({ loading: false })
        return
      }
      const newAnswers = { ...get().answers, [cursor]: answer }
      if (response.data!.completed) {
        set({ screen: 'complete', loading: false, answers: newAnswers })
      } else {
        set({ cursor: cursor + 1, loading: false, answers: newAnswers, sampleDisplayedAt: Date.now(), currentViewerOpens: 0 })
      }
    } catch (e) {
      set({ loading: false })
    }
  },

  incrementViewerOpens: () => set(s => ({ currentViewerOpens: s.currentViewerOpens + 1 })),
}))

export function currentSample(state: AppState): SampleOut | null {
  if (!state.session) return null
  return state.session.samples[state.cursor] ?? null
}
