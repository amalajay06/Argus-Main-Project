import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import ChatInput from "../components/ChatInput";
import { api, ApiError } from "../lib/api";
import type { Investigation } from "../lib/types";

const SUGGESTED_PROMPTS = [
  "Find everyone entering the building after 8 PM.",
  "Show vehicles leaving the area between 9 and 10 PM.",
  "Find the person carrying a backpack.",
];

export default function Home() {
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api
      .get<Investigation[]>("/investigations")
      .then(setInvestigations)
      .catch(() => setError("Couldn't load your investigations."))
      .finally(() => setLoading(false));
  }, []);

  const startInvestigation = async (message: string) => {
    setStarting(true);
    setError(null);
    try {
      const name = message.length > 60 ? `${message.slice(0, 57)}…` : message;
      const investigation = await api.post<Investigation>("/investigations", { name });
      navigate(`/investigations/${investigation.id}`, { state: { initialMessage: message } });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't start the investigation. Is the backend running?");
      setStarting(false);
    }
  };

  return (
    <div className="flex">
      <Sidebar investigations={investigations} />

      <main className="flex-1 h-screen flex flex-col items-center justify-center px-6">
        <div className="w-full max-w-2xl text-center">
          <h1 className="font-display text-4xl text-white mb-3">Good evening.</h1>
          <p className="text-argus-muted mb-10">What are you investigating?</p>

          <ChatInput
            placeholder="Start a new investigation…"
            onSend={startInvestigation}
            disabled={starting}
          />

          {error && <p className="text-sm text-red-400 mt-4">{error}</p>}

          <div className="mt-8 flex flex-wrap justify-center gap-2">
            {SUGGESTED_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                onClick={() => startInvestigation(prompt)}
                disabled={starting}
                className="text-xs text-argus-muted border border-argus-border rounded-full px-3 py-1.5 hover:border-argus-accent/50 hover:text-white/80 transition-colors disabled:opacity-40"
              >
                {prompt}
              </button>
            ))}
          </div>

          {!loading && investigations.length === 0 && (
            <p className="text-xs text-argus-muted mt-12">
              Or use{" "}
              <button
                onClick={() => navigate("/investigations/new")}
                className="text-argus-accent hover:underline"
              >
                New Investigation
              </button>{" "}
              to add case details first.
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
