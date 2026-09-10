import { useStore, currentSample } from '../store'
import FrameStrip from './FrameStrip'

export default function LocalizeView() {
  const state = useStore()
  const { cursor, session, answers, loading, submitAnswer, goBack, goForward, incrementViewerOpens } = state
  const sample = currentSample(state)
  const total = session?.total ?? 0
  const prev = answers[cursor] as [number, number] | undefined

  if (!sample) return null

  const nPairs = sample.n_frames - 1

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="w-full bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        {/* Header */}
        <div className="bg-violet-600 px-8 py-4 flex items-center justify-between">
          <h2 className="text-white font-semibold">Localize: Which pair was swapped?</h2>
          <span className="text-violet-200 text-sm font-mono">
            {cursor + 1} / {total}
          </span>
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-violet-100">
          <div
            className="h-1 bg-violet-500 transition-all"
            style={{ width: `${((cursor) / total) * 100}%` }}
          />
        </div>

        <div className="px-8 py-6 space-y-6">
          <p className="text-gray-500 text-sm text-center">
            Exactly one adjacent pair has been swapped. Click which pair.
          </p>

          <FrameStrip frameUrls={sample.frame_urls} onViewerOpen={incrementViewerOpens} />

          {sample.scene_description && (
            <p className="text-center text-gray-500 text-sm">
              {sample.scene_description}
            </p>
          )}

          <div className="flex gap-3">
            {Array.from({ length: nPairs }, (_, i) => {
              const isPrev = prev !== undefined && prev[0] === i && prev[1] === i + 1
              return (
                <button
                  key={i}
                  onClick={() => submitAnswer([i, i + 1])}
                  disabled={loading}
                  className={`flex-1 py-3 px-4 font-medium rounded-xl transition-colors text-sm text-violet-800 disabled:opacity-40
                    ${isPrev
                      ? 'bg-violet-300 ring-2 ring-violet-400 ring-offset-2'
                      : 'bg-violet-100 hover:bg-violet-200 disabled:bg-violet-50'}`}
                >
                  Frame {i} ↔ {i + 1}
                </button>
              )
            })}
          </div>

          {(cursor > 0 || prev !== undefined) && (
            <div className="flex justify-between">
              {cursor > 0 ? (
                <button
                  onClick={goBack}
                  disabled={loading}
                  className="px-5 py-3 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 disabled:opacity-40 rounded-xl transition-colors"
                >
                  ← Back
                </button>
              ) : <div />}
              {prev !== undefined && (
                <button
                  onClick={goForward}
                  disabled={loading}
                  className="px-5 py-3 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 disabled:opacity-40 rounded-xl transition-colors"
                >
                  Next →
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
