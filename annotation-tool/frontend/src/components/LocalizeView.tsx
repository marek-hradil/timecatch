import { useStore, currentSample } from '../store'
import FrameStrip from './FrameStrip'

export default function LocalizeView() {
  const state = useStore()
  const { cursor, session, loading, submitAnswer } = state
  const sample = currentSample(state)
  const total = session?.total ?? 0

  if (!sample) return null

  const nPairs = sample.n_frames - 1

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="max-w-3xl w-full bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
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

          <FrameStrip frameUrls={sample.frame_urls} />

          {sample.scene_description && (
            <p className="text-center text-gray-400 text-xs italic">
              {sample.scene_description}
            </p>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {Array.from({ length: nPairs }, (_, i) => (
              <button
                key={i}
                onClick={() => submitAnswer([i, i + 1])}
                disabled={loading}
                className="py-3 px-4 bg-violet-100 hover:bg-violet-200 disabled:bg-violet-50 text-violet-800 font-medium rounded-xl transition-colors text-sm"
              >
                Frame {i} ↔ {i + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
