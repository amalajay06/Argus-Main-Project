import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { RequireAuth } from "./components/RequireAuth";
import Login from "./pages/Login";
import Home from "./pages/Home";
import NewInvestigation from "./pages/NewInvestigation";
import InvestigationChat from "./pages/InvestigationChat";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <RequireAuth>
                <Home />
              </RequireAuth>
            }
          />
          <Route
            path="/investigations/new"
            element={
              <RequireAuth>
                <NewInvestigation />
              </RequireAuth>
            }
          />
          <Route
            path="/investigations/:id"
            element={
              <RequireAuth>
                <InvestigationChat />
              </RequireAuth>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
