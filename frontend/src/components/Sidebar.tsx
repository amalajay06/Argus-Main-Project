import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import type { Investigation } from "../lib/types";

interface SidebarProps {
  investigations: Investigation[];
  activeId?: string;
}

export default function Sidebar({ investigations, activeId }: SidebarProps) {
  const { username, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <aside className="w-64 shrink-0 h-screen bg-argus-panel border-r border-argus-border flex flex-col">
      <div className="px-5 pt-6 pb-4">
        <Link to="/" className="font-display text-xl tracking-wide text-white">
          ARGUS
        </Link>
      </div>

      <div className="px-4">
        <Link
          to="/investigations/new"
          className="block text-center text-sm rounded-md border border-argus-accent/40 text-argus-accent py-2 hover:bg-argus-accent/10 transition-colors"
        >
          + New Investigation
        </Link>
      </div>

      <div className="mt-6 px-4 flex-1 overflow-y-auto">
        <p className="text-xs uppercase tracking-wider text-argus-muted mb-2 px-1">Recent</p>
        <nav className="space-y-0.5">
          {investigations.length === 0 && (
            <p className="text-xs text-argus-muted px-1 py-2">No investigations yet.</p>
          )}
          {investigations.map((inv) => (
            <Link
              key={inv.id}
              to={`/investigations/${inv.id}`}
              className={`block truncate text-sm rounded-md px-3 py-2 transition-colors ${
                activeId === inv.id
                  ? "bg-argus-accent/10 text-argus-accent"
                  : "text-white/70 hover:bg-white/5 hover:text-white"
              }`}
            >
              {inv.name}
            </Link>
          ))}
        </nav>
      </div>

      <div className="px-4 py-4 border-t border-argus-border">
        <div className="flex items-center justify-between">
          <span className="text-sm text-white/70 truncate">{username}</span>
          <button
            onClick={handleLogout}
            className="text-xs text-argus-muted hover:text-white/80 transition-colors"
          >
            Log out
          </button>
        </div>
      </div>
    </aside>
  );
}
