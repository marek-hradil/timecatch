import { useStore } from '../store'

const VALID_DATASETS = ['CRAFT', 'CLEVRER', 'MTL-AQA', 'drive_lm']
const VALID_TASKS = ['detect', 'localize']

export default function Landing() {
  const { prolificPid, dataset, task, loading, error, startSession } = useStore()

  const paramsValid =
    prolificPid.length > 0 &&
    VALID_DATASETS.includes(dataset) &&
    VALID_TASKS.includes(task)

  if (!paramsValid) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md text-center p-8 bg-white rounded-2xl shadow-sm border border-gray-200">
          <div className="text-4xl mb-4">🔗</div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-3">Invalid Study Link</h1>
          <p className="text-gray-500">
            Please open this study through your Prolific link. The URL must include your
            participant ID, dataset, and task.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md text-center p-8 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="text-4xl mb-4">🎬</div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">Video Frame Annotation</h1>
        <p className="text-gray-500 mb-2 text-sm">
          Dataset: <span className="font-medium text-gray-700">{dataset}</span>
          &ensp;·&ensp;
          Task: <span className="font-medium text-gray-700">{task}</span>
        </p>
        <p className="text-gray-500 mb-6 text-sm">
          You will annotate <strong>15 short video sequences</strong>. This takes about 5–10 minutes.
        </p>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        <button
          onClick={startSession}
          disabled={loading}
          className="w-full py-3 px-6 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-medium rounded-xl transition-colors"
        >
          {loading ? 'Loading…' : 'Start Study'}
        </button>
      </div>
    </div>
  )
}
