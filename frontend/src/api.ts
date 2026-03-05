// Backend API URL - change this for production deployment
const API = import.meta.env.VITE_API_URL || "https://rag-6ftt.onrender.com";

async function req<T>(
  path: string,
  init?: RequestInit & { json?: unknown }
): Promise<T> {
  const { json, ...opts } = init ?? {};
  const headers: HeadersInit = { ...((opts.headers as Record<string, string>) ?? {}) };
  if (json !== undefined) {
    (headers as Record<string, string>)["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  }

  const url = `${API}${path}`;

  // #region agent log
  fetch("http://127.0.0.1:7355/ingest/c8f57fbe-6440-4313-a1ca-655dc0688a2a", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "f109f0",
    },
    body: JSON.stringify({
      sessionId: "f109f0",
      runId: "initial",
      hypothesisId: "H1",
      location: "frontend/src/api.ts:req:beforeFetch",
      message: "Request starting",
      data: { url, hasJson: json !== undefined },
      timestamp: Date.now(),
    }),
  }).catch(() => {});
  // #endregion

  try {
    const r = await fetch(url, { ...opts, headers });

    // #region agent log
    fetch("http://127.0.0.1:7355/ingest/c8f57fbe-6440-4313-a1ca-655dc0688a2a", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Debug-Session-Id": "f109f0",
      },
      body: JSON.stringify({
        sessionId: "f109f0",
        runId: "initial",
        hypothesisId: "H2",
        location: "frontend/src/api.ts:req:afterFetch",
        message: "Response received",
        data: { url, status: r.status },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion

    if (!r.ok) {
      const t = await r.text();
      let msg = t;
      try {
        const j = JSON.parse(t);
        msg = (j as any).detail ?? t;
      } catch {}

      // #region agent log
      fetch("http://127.0.0.1:7355/ingest/c8f57fbe-6440-4313-a1ca-655dc0688a2a", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Debug-Session-Id": "f109f0",
        },
        body: JSON.stringify({
          sessionId: "f109f0",
          runId: "initial",
          hypothesisId: "H3",
          location: "frontend/src/api.ts:req:errorResponse",
          message: "Non-OK response",
          data: { url, status: r.status, bodySnippet: t.slice(0, 200) },
          timestamp: Date.now(),
        }),
      }).catch(() => {});
      // #endregion

      throw new Error(msg);
    }
    if (r.status === 204) return undefined as T;
    return r.json() as Promise<T>;
  } catch (err) {
    // #region agent log
    fetch("http://127.0.0.1:7355/ingest/c8f57fbe-6440-4313-a1ca-655dc0688a2a", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Debug-Session-Id": "f109f0",
      },
      body: JSON.stringify({
        sessionId: "f109f0",
        runId: "initial",
        hypothesisId: "H4",
        location: "frontend/src/api.ts:req:catch",
        message: "Fetch threw error",
        data: {
          url,
          errorName: err instanceof Error ? err.name : typeof err,
          errorMessage: err instanceof Error ? err.message : String(err),
        },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion

    throw err;
  }
}

export interface UploadRes {
  job_id: string;
  document_id: string;
  message?: string;
}

export interface JobStatusRes {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  message?: string | null;
}

export interface SourceChunk {
  text: string;
  source?: string | null;
  score?: number | null;
}

export interface QueryRes {
  answer: string;
  sources: SourceChunk[];
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  size_bytes: number;
  uploaded_at: string;
}

export interface DocumentListRes {
  documents: DocumentInfo[];
}

export interface StreamEvent {
  type: "token" | "sources" | "done" | "error";
  content?: string;
  sources?: SourceChunk[];
  message?: string;
}

export const api = {
  upload: (file: File, signal?: AbortSignal) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<UploadRes>("/documents/upload", { method: "POST", body: fd, signal });
  },
  jobStatus: (jobId: string) =>
    req<JobStatusRes>(`/documents/jobs/${jobId}`),
  query: (question: string) =>
    req<QueryRes>("/query", { method: "POST", json: { question } }),
  
  // Streaming query - yields tokens as they arrive
  queryStream: async function* (
    question: string,
    signal?: AbortSignal
  ): AsyncGenerator<StreamEvent> {
    const response = await fetch(`${API}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal,
    });

    if (!response.ok) {
      const text = await response.text();
      let msg = text;
      try {
        const j = JSON.parse(text);
        msg = j.detail ?? text;
      } catch {}
      throw new Error(msg);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data.trim()) {
            try {
              const event: StreamEvent = JSON.parse(data);
              yield event;
            } catch {}
          }
        }
      }
    }
  },
  
  listDocuments: () => req<DocumentListRes>("/documents"),
  deleteDocument: (documentId: string) =>
    req<void>(`/documents/${documentId}`, { method: "DELETE" }),
};
