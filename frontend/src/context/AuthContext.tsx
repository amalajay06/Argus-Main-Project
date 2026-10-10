import { createContext, useContext, useState, ReactNode } from "react";
import { api } from "../lib/api";
import { clearToken, getToken, setToken } from "../lib/api";
import type { AuthResponse } from "../lib/types";

interface AuthContextValue {
  isAuthenticated: boolean;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string, fullName: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [username, setUsername] = useState<string | null>(
    () => localStorage.getItem("argus_username")
  );

  const persistSession = (auth: AuthResponse) => {
    setToken(auth.token);
    localStorage.setItem("argus_username", auth.username);
    setUsername(auth.username);
  };

  const login = async (username: string, password: string) => {
    const auth = await api.post<AuthResponse>("/auth/login", { username, password });
    persistSession(auth);
  };

  const register = async (username: string, email: string, password: string, fullName: string) => {
    const auth = await api.post<AuthResponse>("/auth/register", { username, email, password, fullName });
    persistSession(auth);
  };

  const logout = () => {
    clearToken();
    localStorage.removeItem("argus_username");
    setUsername(null);
  };

  return (
    <AuthContext.Provider
      value={{ isAuthenticated: !!getToken(), username, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
