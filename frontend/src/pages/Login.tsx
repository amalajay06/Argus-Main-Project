import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../lib/api";

export default function Login() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "login") {
        await login(username, password);
      } else {
        await register(username, email, password, fullName);
      }
      navigate("/");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message || "Something went wrong. Check your details and try again.");
      } else {
        setError("Couldn't reach the server. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-sm">
        <div className="text-center mb-14">
          <h1 className="font-display text-5xl tracking-wide text-white">ARGUS</h1>
          <p className="mt-3 text-sm text-argus-muted">Investigation Assistant</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === "register" && (
            <>
              <input
                type="text"
                placeholder="Full name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-3 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors"
                required
              />
              <input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-3 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors"
                required
              />
            </>
          )}

          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-3 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors"
            required
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-argus-panel border border-argus-border rounded-md px-4 py-3 text-sm placeholder:text-argus-muted focus:border-argus-accent transition-colors"
            required
            minLength={8}
          />

          {error && <p className="text-sm text-red-400">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-argus-accent text-argus-black font-medium rounded-md py-3 text-sm hover:bg-argus-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
          </button>
        </form>

        <button
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError(null);
          }}
          className="w-full text-center text-xs text-argus-muted hover:text-white/80 mt-6 transition-colors"
        >
          {mode === "login" ? "New investigator? Create an account" : "Already have an account? Log in"}
        </button>
      </div>
    </div>
  );
}
