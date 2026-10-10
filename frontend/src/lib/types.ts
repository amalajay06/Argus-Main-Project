export interface AuthResponse {
  token: string;
  username: string;
  fullName: string | null;
}

export interface Investigation {
  id: string;
  name: string;
  description: string | null;
  incidentDate: string | null;
  location: string | null;
  status: "OPEN" | "CLOSED" | "ARCHIVED";
  createdAt: string;
  updatedAt: string;
  evidenceCount: number;
  findingsCount: number;
}

export interface EvidenceResult {
  camera: string;
  start_timestamp: string;
  end_timestamp: string;
  description: string;
  relevance: number;
  thumbnail_url: string | null;
}

export interface ChatMessage {
  id: string;
  sender: "USER" | "ARGUS";
  content: string;
  resultPayload: string | null;
  createdAt: string;
}

export interface Evidence {
  id: string;
  type: "VIDEO" | "IMAGE" | "DOCUMENT";
  originalFilename: string;
  sourceLabel: string | null;
  sha256Hash: string | null;
  sizeBytes: number | null;
  mimeType: string | null;
  processingStatus: "PENDING" | "PROCESSING" | "READY" | "FAILED";
  processingProgress: number;
  createdAt: string;
}
