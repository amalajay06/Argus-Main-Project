import type { EvidenceResult } from "../lib/types";

export default function EvidenceResultCard({ result }: { result: EvidenceResult }) {
  return (
    <div className="w-full max-w-md bg-argus-panel border border-argus-border rounded-lg overflow-hidden">
      <div className="aspect-video bg-black/40 flex items-center justify-center text-argus-muted text-xs">
        {result.thumbnail_url ? (
          <img src={result.thumbnail_url} alt="Evidence thumbnail" className="w-full h-full object-cover" />
        ) : (
          "No preview available"
        )}
      </div>
      <div className="p-4 space-y-2">
        <div className="flex items-center justify-between text-xs text-argus-muted">
          <span>{result.camera}</span>
          <span>
            {result.start_timestamp} – {result.end_timestamp}
          </span>
        </div>
        <p className="text-sm text-white/90">{result.description}</p>
        <div className="flex items-center justify-between pt-1">
          <span className="text-xs text-argus-accent">
            Relevance {Math.round(result.relevance * 100)}%
          </span>
          <div className="flex gap-2">
            <button className="text-xs border border-argus-border rounded-md px-3 py-1.5 hover:border-argus-accent/50 transition-colors">
              View
            </button>
            <button className="text-xs border border-argus-accent/40 text-argus-accent rounded-md px-3 py-1.5 hover:bg-argus-accent/10 transition-colors">
              Save Finding
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
