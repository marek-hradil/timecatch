import { useState, useEffect } from 'react'

interface FrameStripProps {
  frameUrls: string[]
  onViewerOpen?: () => void
}

export default function FrameStrip({ frameUrls, onViewerOpen }: FrameStripProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  useEffect(() => {
    if (activeIndex === null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setActiveIndex(null)
      if (e.key === 'ArrowLeft') setActiveIndex(i => Math.max(0, (i ?? 0) - 1))
      if (e.key === 'ArrowRight') setActiveIndex(i => Math.min(frameUrls.length - 1, (i ?? 0) + 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeIndex, frameUrls.length])

  return (
    <>
      <div className="flex gap-2 w-full">
        {frameUrls.map((url, i) => (
          <div
            key={i}
            className="flex flex-col items-center gap-1 flex-1 min-w-[100px] cursor-zoom-in group"
            onClick={() => { setActiveIndex(i); onViewerOpen?.() }}
          >
            <img
              src={url}
              alt={`Frame ${i}`}
              className="w-full h-auto rounded-lg border border-gray-200 group-hover:border-indigo-400"
            />
            <span className="text-xs text-gray-400 font-mono">{i}</span>
          </div>
        ))}
      </div>

      {activeIndex !== null && (
        <div
          className="fixed inset-0 bg-black/85 z-50 flex flex-col items-center justify-center select-none"
          onClick={() => setActiveIndex(null)}
        >
          <div
            className="relative"
            onClick={e => e.stopPropagation()}
          >
            <img
              src={frameUrls[activeIndex]}
              alt={`Frame ${activeIndex}`}
              className="w-[40vw] min-w-[300px] object-contain rounded-lg"
            />

            <button
              onClick={() => setActiveIndex(i => Math.max(0, (i ?? 0) - 1))}
              disabled={activeIndex === 0}
              className="absolute left-2 top-1/2 -translate-y-1/2 w-12 h-12 flex items-center justify-center rounded-full bg-black/40 hover:bg-black/60 disabled:opacity-20 text-white text-2xl"
            >
              ‹
            </button>

            <button
              onClick={() => setActiveIndex(i => Math.min(frameUrls.length - 1, (i ?? 0) + 1))}
              disabled={activeIndex === frameUrls.length - 1}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-12 h-12 flex items-center justify-center rounded-full bg-black/40 hover:bg-black/60 disabled:opacity-20 text-white text-2xl"
            >
              ›
            </button>
          </div>

          <div className="flex gap-1.5 mt-4">
            {frameUrls.map((_, i) => (
              <div
                key={i}
                className={`w-2 h-2 rounded-full ${i === activeIndex ? 'bg-white' : 'bg-white/25'}`}
              />
            ))}
          </div>

          <p className="text-white/40 text-xs mt-3">
            Frame {activeIndex} · ← → arrows · click outside or Esc to close
          </p>
        </div>
      )}
    </>
  )
}
