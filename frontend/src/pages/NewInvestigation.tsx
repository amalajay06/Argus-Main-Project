import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import { api, ApiError } from "../lib/api";
import type { Investigation } from "../lib/types";

export default function NewInvestigation() {
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [incidentDate, setIncidentDate] = useState("");
  const [location, setLocation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    api.get<Investigation[]>("/investigations").then(setInvestigations).catch(() => {});
  }, []);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const investigation = await api.post<Investigation>("/investigations", {
        name,
        description: description || null,
        incidentDate: incidentDate || null,
        location: location || null,
      });
      navigate(`/investigations/${investigation.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't create the investigation.");
      setSubmitting(false);
    }
  };

  return (
    <div className="flex">
      <Sidebar investigations={investigations} />

      <main className="flex-1 h-screen overflow-y-auto px-6 py-16">
        <div className="max-w-lg mx-auto">
          <h1 className="font-display text-3xl text-white mb-8">New Investigation</h1>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-xs text-argus-muted mb-1.5">Investigation name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Vehicle Theft – MG Road"
                required
                className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-2.5 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors"
              />
            </div>

            <div>
              <label className="block text-xs text-argus-muted mb-1.5">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Investigation regarding a reported vehicle theft."
                rows={3}
                className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-2.5 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors resize-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-argus-muted mb-1.5">Incident date</label>
                <input
                  type="date"
                  value={incidentDate}
                  onChange={(e) => setIncidentDate(e.target.value)}
                  className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-2.5 text-sm focus:border-argus-accent transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs text-argus-muted mb-1.5">Location</label>
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="MG Road"
                  className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-2.5 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors"
                />
              </div>
            </div>

            {error && <p className="text-sm text-red-400">{error}</p>}

            <button
              type="submit"
              disabled={submitting}
              className="bg-argus-accent text-argus-black text-sm font-medium rounded-md px-6 py-2.5 hover:bg-argus-accent/90 disabled:opacity-50 transition-colors"
            >
              {submitting ? "Creating…" : "Create Investigation"}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
