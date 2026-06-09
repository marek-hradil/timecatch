import ReactMarkdown from 'react-markdown'
import { useStore } from '../store'

export default function Instructions() {
  const { session, dataset, task, beginAnnotating } = useStore()
  const md = session?.instructions_md ?? ''

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="max-w-2xl w-full bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="bg-indigo-600 px-8 py-4">
          <p className="text-indigo-200 text-sm uppercase tracking-wide font-medium">
            {dataset} · {task}
          </p>
          <h1 className="text-white text-2xl font-semibold mt-1">Study Instructions</h1>
        </div>

        <div className="px-8 py-6 prose prose-gray max-w-none">
          <ReactMarkdown>{md}</ReactMarkdown>
        </div>

        <div className="px-8 pb-8">
          <button
            onClick={beginAnnotating}
            className="w-full py-3 px-6 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl transition-colors text-lg"
          >
            I understand — Begin
          </button>
        </div>
      </div>
    </div>
  )
}
