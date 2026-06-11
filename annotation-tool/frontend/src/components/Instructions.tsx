import ReactMarkdown from 'react-markdown'
import { useRef, useEffect } from 'react'
import { useStore } from '../store'

const VIDEO_EXT = /\.(mov|mp4|webm)$/i

function MutedVideo({ src }: { src: string }) {
  const ref = useRef<HTMLVideoElement>(null)
  useEffect(() => { if (ref.current) ref.current.muted = true }, [])
  return (
    <video ref={ref} controls muted className="w-full rounded-lg my-2">
      <source src={src} />
    </video>
  )
}

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

        <div className="px-8 py-6 space-y-3 text-gray-700 text-sm leading-relaxed">
          <ReactMarkdown
            components={{
              h1: ({ children }) => <h1 className="text-xl font-bold text-gray-900 mt-4 mb-1">{children}</h1>,
              h2: ({ children }) => <h2 className="text-lg font-semibold text-gray-900 mt-4 mb-1">{children}</h2>,
              h3: ({ children }) => <h3 className="text-base font-semibold text-gray-800 mt-3 mb-1">{children}</h3>,
              p: ({ children }) => <p className="mb-2">{children}</p>,
              strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
              ul: ({ children }) => <ul className="list-disc pl-5 space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1">{children}</ol>,
              img: ({ src, alt }) =>
                src && VIDEO_EXT.test(src) ? (
                  <MutedVideo src={src} />
                ) : (
                  <img src={src} alt={alt ?? ''} className="w-full rounded-lg my-2" />
                ),
            }}
          >
            {md}
          </ReactMarkdown>
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
