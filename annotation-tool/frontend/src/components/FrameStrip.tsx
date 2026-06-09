interface FrameStripProps {
  frameUrls: string[]
}

export default function FrameStrip({ frameUrls }: FrameStripProps) {
  return (
    <div className="flex gap-2 w-full">
      {frameUrls.map((url, i) => (
        <div key={i} className="flex flex-col items-center gap-1 flex-1 min-w-0">
          <img
            src={url}
            alt={`Frame ${i}`}
            className="w-full aspect-square object-contain rounded-lg border border-gray-200 bg-gray-50"
          />
          <span className="text-xs text-gray-400 font-mono">{i}</span>
        </div>
      ))}
    </div>
  )
}
