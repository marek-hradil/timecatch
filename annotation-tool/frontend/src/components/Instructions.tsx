import ReactMarkdown from 'react-markdown'
import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import { useStore } from '../store'

const VIDEO_EXT = /\.(mov|mp4|webm)$/i

function MutedVideo({ src, onPlay }: { src: string; onPlay?: () => void }) {
  const ref = useRef<HTMLVideoElement>(null)
  useEffect(() => { if (ref.current) ref.current.muted = true }, [])
  return (
    <video ref={ref} controls muted className="w-full rounded-lg my-2" onPlay={onPlay}>
      <source src={src} />
    </video>
  )
}

function Lightbox({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-red-500 hover:text-red-700 text-3xl leading-none"
      >
        ✕
      </button>
      <img
        src={src}
        alt={alt}
        className="max-w-full max-h-full rounded-lg shadow-2xl object-contain"
        onClick={e => e.stopPropagation()}
      />
    </div>
  )
}

export default function Instructions() {
  const { session, dataset, task, beginAnnotating } = useStore()
  const md = session?.instructions_md ?? ''
  const [lightbox, setLightbox] = useState<{ src: string; alt: string } | null>(null)

  // Local refs — mutations never trigger a re-render, so the video element is never destroyed.
  const videoPlayedRef = useRef(false)
  const lightboxCountRef = useRef(0)

  const handleVideoPlay = useCallback(() => { videoPlayedRef.current = true }, [])

  const openLightbox = useCallback((src: string, alt: string) => {
    lightboxCountRef.current += 1
    setLightbox({ src, alt })
  }, [])

  const handleBegin = useCallback(() => {
    beginAnnotating({ videoPlayed: videoPlayedRef.current, instructionLightboxOpens: lightboxCountRef.current })
  }, [beginAnnotating])

  const mdComponents = useMemo(() => ({
    h1: ({ children }: any) => <h1 className="text-xl font-bold text-gray-900 mt-4 mb-1">{children}</h1>,
    h2: ({ children }: any) => <h2 className="text-lg font-semibold text-gray-900 mt-4 mb-1">{children}</h2>,
    h3: ({ children }: any) => <h3 className="text-base font-semibold text-gray-800 mt-3 mb-1">{children}</h3>,
    p: ({ children }: any) => <p className="mb-2">{children}</p>,
    strong: ({ children }: any) => <strong className="font-semibold text-gray-900">{children}</strong>,
    ul: ({ children }: any) => <ul className="list-disc pl-5 space-y-1">{children}</ul>,
    ol: ({ children }: any) => <ol className="list-decimal pl-5 space-y-1">{children}</ol>,
    img: ({ src, alt }: any) =>
      src && VIDEO_EXT.test(src) ? (
        <MutedVideo src={src} onPlay={handleVideoPlay} />
      ) : (
        <img
          src={src}
          alt={alt ?? ''}
          className="w-full rounded-lg my-2 cursor-zoom-in"
          onClick={() => openLightbox(src!, alt ?? '')}
        />
      ),
  }), [openLightbox, handleVideoPlay])

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
          <ReactMarkdown components={mdComponents}>
            {md}
          </ReactMarkdown>
        </div>

        <div className="px-8 pb-8">
          <button
            onClick={handleBegin}
            className="w-full py-3 px-6 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl transition-colors text-lg"
          >
            I understand — Begin
          </button>
        </div>
      </div>

      {lightbox && (
        <Lightbox src={lightbox.src} alt={lightbox.alt} onClose={() => setLightbox(null)} />
      )}
    </div>
  )
}
