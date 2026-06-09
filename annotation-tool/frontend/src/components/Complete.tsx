import { useStore } from '../store'

export default function Complete() {
  const { session } = useStore()
  const completionUrl = session?.prolific_completion_url ?? ''

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md text-center p-8 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="text-5xl mb-4">🎉</div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-3">All done!</h1>
        <p className="text-gray-500 mb-2">
          Thank you for completing the annotation study.
        </p>
        <p className="text-gray-500 mb-8">
          Click the button below to return to Prolific and confirm your submission.
        </p>
        <a
          href={completionUrl}
          className="inline-block w-full py-3 px-6 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl transition-colors text-center"
        >
          Submit on Prolific →
        </a>
      </div>
    </div>
  )
}
