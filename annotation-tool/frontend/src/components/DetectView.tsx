import { useStore, currentSample } from '../store'
import FrameStrip from './FrameStrip'

export default function DetectView() {
  const state = useStore()
  const { cursor, session, answers, loading, submitAnswer, goBack, goForward, incrementViewerOpens } = state
  const sample = currentSample(state)
  const total = session?.total ?? 0
  const prev = answers[cursor]

  if (!sample) return null

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="w-full bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        {/* Header */}
        <div className="bg-indigo-600 px-8 py-4 flex items-center justify-between">
          <h2 className="text-white font-semibold">Are the frames in the correct order?</h2>
          <span className="text-indigo-200 text-sm font-mono">
            {cursor + 1} / {total}
          </span>
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-indigo-100">
          <div
            className="h-1 bg-indigo-500 transition-all"
            style={{ width: `${(cursor / total) * 100}%` }}
          />
        </div>

        <div className="px-6 py-6 space-y-6">
          <p className="text-gray-500 text-sm text-center">
            Frames are shown left→right in temporal order. Have any two <strong>adjacent</strong> frames been swapped?
          </p>

          <FrameStrip frameUrls={sample.frame_urls} onViewerOpen={incrementViewerOpens} />

          {sample.scene_description && (
            <p className="text-center text-gray-500 text-sm">
              {sample.scene_description}
            </p>
          )}

          <div className="flex gap-4">
            <button
              onClick={() => submitAnswer(false)}
              disabled={loading}
              className={`flex-1 py-4 font-semibold rounded-xl transition-colors text-base disabled:opacity-40 text-white
                ${prev === false
                  ? 'bg-green-700 ring-2 ring-green-300 ring-offset-2'
                  : 'bg-green-600 hover:bg-green-700'}`}
            >
              ✓ Yes, everything is in order
            </button>
            <button
              onClick={() => submitAnswer(true)}
              disabled={loading}
              className={`flex-1 py-4 font-semibold rounded-xl transition-colors text-base disabled:opacity-40 text-white
                ${prev === true
                  ? 'bg-red-600 ring-2 ring-red-300 ring-offset-2'
                  : 'bg-red-500 hover:bg-red-600'}`}
            >
              ✗ No, the frames are out of order
            </button>
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
