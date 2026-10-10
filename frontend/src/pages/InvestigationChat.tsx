import { useEffect, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import ChatInput from "../components/ChatInput";
import EvidenceResultCard from "../components/EvidenceResultCard";
import { api, ApiError } from "../lib/api";
import type { ChatMessage, Evidence, EvidenceResult, Investigation } from "../lib/types";

export default function InvestigationChat() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();

  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const hasSentInitial = useRef(false);

  const loadCaseData = async (investigationId: string) => {
    const [inv, history, evidenceList] = await Promise.all([
      api.get<Investigation>(`/investigations/${investigationId}`),
      api.get<ChatMessage[]>(`/investigations/${investigationId}/messages`),
      api.get<Evidence[]>(`/investigations/${investigationId}/evidence`),
    ]);
    setInvestigation(inv);
    setMessages(history);
    setEvidence(evidenceList);
  };

  useEffect(() => {
    if (!id) return;
    api.get<Investigation[]>("/investigations").then(setInvestigations).catch(() => {});
    loadCaseData(id).catch(() => setError("Couldn't load this investigation."));
    hasSentInitial.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    const initialMessage = (location.state as { initialMessage?: string } | null)?.initialMessage;
    if (initialMessage && id && !hasSentInitial.current) {
      hasSentInitial.current = true;
      sendMessage(initialMessage);
      window.history.replaceState({}, document.title);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, location.state]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (content: string) => {
    if (!id) return;
    setSending(true);
    setError(null);
    setMessages((prev) => [
      ...prev,
      { id: `temp-${Date.now()}`, sender: "USER", content, resultPayload: null, createdAt: new Date().toISOString() },
    ]);
    try {
      const reply = await api.post<ChatMessage>(`/investigations/${id}/messages`, { content });
      setMessages((prev) => [...prev, reply]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't reach Argus. Is the backend running?");
    } finally {
      setSending(false);
    }
  };

  const handleUpload = async (file: File) => {
    if (!id) return;
    setUploading(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("sourceLabel", file.name);
    try {
      const uploaded = await api.postForm<Evidence>(`/investigations/${id}/evidence`, form);
      setEvidence((prev) => [uploaded, ...prev]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const parseResults = (payload: string | null): EvidenceResult[] => {
    if (!payload) return [];
    try {
      const parsed = JSON.parse(payload);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  if (!investigation) {
    return (
      <div className="flex">
        <Sidebar investigations={investigations} activeId={id} />
        <main className="flex-1 h-screen flex items-center justify-center text-argus-muted text-sm">
          {error ?? "Loading investigation…"}
        </main>
      </div>
    );
  }

  return (
    <div className="flex">
      <Sidebar investigations={investigations} activeId={id} />

      {/* Case info panel */}
      <aside className="w-64 shrink-0 h-screen border-r border-argus-border px-5 py-6 overflow-y-auto">
        <p className="text-xs uppercase tracking-wider text-argus-muted mb-1">Case</p>
        <h2 className="font-display text-xl text-white mb-6 leading-snug">{investigation.name}</h2>

        <div className="space-y-4 text-sm">
          <div className="flex justify-between">
            <span className="text-argus-muted">Evidence</span>
            <span>{evidence.length} sources</span>
          </div>
          <div className="flex justify-between">
            <span className="text-argus-muted">Findings</span>
            <span>{investigation.findingsCount}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-argus-muted">Storyline</span>
            <span className="text-argus-muted">Coming soon</span>
          </div>
        </div>

        <div className="h-px bg-argus-border my-6" />

        {investigation.incidentDate && (
          <div className="mb-4">
            <p className="text-xs text-argus-muted mb-0.5">Incident</p>
            <p className="text-sm">{investigation.incidentDate}</p>
          </div>
        )}
        {investigation.location && (
          <div className="mb-4">
            <p className="text-xs text-argus-muted mb-0.5">Location</p>
            <p className="text-sm">{investigation.location}</p>
          </div>
        )}

        <div className="h-px bg-argus-border my-6" />

        <p className="text-xs uppercase tracking-wider text-argus-muted mb-2">Evidence sources</p>
        <label className="block text-center text-xs border border-dashed border-argus-border rounded-md py-3 mb-3 cursor-pointer hover:border-argus-accent/50 transition-colors text-argus-muted">
          {uploading ? "Uploading…" : "+ Add CCTV / image / file"}
          <input
            type="file"
            className="hidden"
            disabled={uploading}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleUpload(file);
              e.target.value = "";
            }}
          />
        </label>
        <div className="space-y-2">
          {evidence.map((ev) => (
            <div key={ev.id} className="text-xs border border-argus-border rounded-md px-3 py-2">
              <p className="truncate text-white/80">{ev.sourceLabel || ev.originalFilename}</p>
              <p className="text-argus-muted mt-0.5">{ev.processingStatus.toLowerCase()}</p>
            </div>
          ))}
        </div>

        <button
          disabled
          title="Available once Phase 4 (Findings & Reports) is built"
          className="w-full mt-6 text-xs border border-argus-border text-argus-muted rounded-md py-2.5 cursor-not-allowed"
        >
          Generate Report
        </button>
      </aside>

      {/* Conversation */}
      <main className="flex-1 h-screen flex flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-8">
          <div className="max-w-2xl mx-auto space-y-6">
            {messages.length === 0 && (
              <p className="text-center text-argus-muted text-sm mt-20">
                Ask Argus about this case's evidence to begin.
              </p>
            )}
            {messages.map((msg) => (
              <div key={msg.id} className={msg.sender === "USER" ? "flex justify-end" : "flex flex-col items-start gap-3"}>
                {msg.sender === "USER" ? (
                  <div className="max-w-md bg-argus-accent/10 border border-argus-accent/30 rounded-lg px-4 py-2.5 text-sm">
                    {msg.content}
                  </div>
                ) : (
                  <>
                    <div className="max-w-md text-sm text-white/90">{msg.content}</div>
                    {parseResults(msg.resultPayload).map((result, i) => (
                      <EvidenceResultCard key={i} result={result} />
                    ))}
                  </>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="px-6 pb-6">
          <div className="max-w-2xl mx-auto">
            {error && <p className="text-sm text-red-400 mb-2">{error}</p>}
            <ChatInput placeholder="Ask about this case…" onSend={sendMessage} disabled={sending} />
          </div>
        </div>
      </main>
    </div>
  );
}
