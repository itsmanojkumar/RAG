import { useState, useEffect, useRef } from "react";
import { api, type QueryRes, type SourceChunk, type DocumentInfo } from "./api";
import "./App.css";

type Tab = "upload" | "ask";

function App() {
  const [tab, setTab] = useState<Tab>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [question, setQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<QueryRes | null>(null);
  const [loading, setLoading] = useState(false);
  const [queryProgress, setQueryProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [uploadController, setUploadController] = useState<AbortController | null>(null);
  const [pollTimer, setPollTimer] = useState<number | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const streamControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const res = await api.listDocuments();
      setDocuments(res.documents);
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setUploadProgress(10);
    setUploadStatus("Uploading file...");
    setLoading(true);
    const controller = new AbortController();
    setUploadController(controller);
    try {
      const res = await api.upload(file, controller.signal);
      setUploadProgress(30);
      setUploadStatus("Processing document...");
      pollJob(res.job_id);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setUploadStatus("");
        setUploadProgress(0);
        setLoading(false);
        return;
      }
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploadStatus("");
      setUploadProgress(0);
      setLoading(false);
    }
  }

  function pollJob(jobId: string) {
    let progressVal = 30;
    const t = window.setInterval(async () => {
      try {
        const s = await api.jobStatus(jobId);
        if (s.status === "processing" && progressVal < 90) {
          progressVal += 5;
          setUploadProgress(progressVal);
        }
        if (s.status === "completed") {
          clearInterval(t);
          setPollTimer(null);
          setUploadProgress(100);
          setUploadStatus("Document ready!");
          setTimeout(() => {
            setLoading(false);
            setUploadProgress(0);
            setUploadStatus("");
            setFile(null);
            loadDocuments();
            setTab("ask");
          }, 1000);
        } else if (s.status === "failed") {
          clearInterval(t);
          setPollTimer(null);
          setError(s.message ?? "Ingestion failed");
          setUploadStatus("");
          setUploadProgress(0);
          setLoading(false);
        }
      } catch {
        clearInterval(t);
        setPollTimer(null);
        setError("Could not check job status");
        setUploadStatus("");
        setUploadProgress(0);
        setLoading(false);
      }
    }, 1500);
    setPollTimer(t);
  }

  function cancelUpload() {
    if (uploadController) uploadController.abort();
    if (pollTimer) {
      clearInterval(pollTimer);
      setPollTimer(null);
    }
    setLoading(false);
    setUploadProgress(0);
    setUploadStatus("");
  }

  async function handleDelete(documentId: string) {
    if (!window.confirm("Delete this document?")) return;
    setDeletingId(documentId);
    setError(null);
    try {
      await api.deleteDocument(documentId);
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  async function handleQuery(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setError(null);
    setQueryResult(null);
    setStreamingAnswer("");
    setLoading(true);
    setIsStreaming(true);
    setQueryProgress(20);
    
    const controller = new AbortController();
    streamControllerRef.current = controller;
    
    try {
      let answer = "";
      let sources: SourceChunk[] = [];
      setQueryProgress(40);
      
      for await (const event of api.queryStream(question.trim(), controller.signal)) {
        if (event.type === "token" && event.content) {
          answer += event.content;
          setStreamingAnswer(answer);
          setQueryProgress(60);
        } else if (event.type === "sources" && event.sources) {
          sources = event.sources;
        } else if (event.type === "error") {
          throw new Error(event.message || "Stream error");
        } else if (event.type === "done") {
          break;
        }
      }
      
      setQueryProgress(100);
      setTimeout(() => {
        setQueryResult({ answer, sources });
        setStreamingAnswer("");
        setQueryProgress(0);
      }, 200);
      
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setStreamingAnswer("");
        setQueryProgress(0);
        return;
      }
      setQueryProgress(0);
      setStreamingAnswer("");
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
      setIsStreaming(false);
      streamControllerRef.current = null;
    }
  }

  function cancelQuery() {
    streamControllerRef.current?.abort();
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <div className="logo-icon">📚</div>
          <h1>RAG Assistant</h1>
        </div>
        <p className="tagline">Upload documents and ask questions powered by AI</p>
        <nav className="tabs">
          <button className={tab === "upload" ? "active" : ""} onClick={() => setTab("upload")}>
            📤 Upload
          </button>
          <button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>
            💬 Ask
          </button>
        </nav>
      </header>

      <main className="main">
        {error && (
          <div className="banner error" role="alert">
            <span className="banner-icon">⚠️</span>
            {error}
          </div>
        )}

        {tab === "upload" && (
          <section className="card">
            <h2><span className="card-icon">📄</span> Upload Document</h2>
            <p className="muted">Supported formats: PDF, TXT (max 50 MB)</p>
            
            <form onSubmit={handleUpload}>
              <div className="file-upload">
                <label className={`file-label ${file ? "has-file" : ""}`}>
                  <input
                    type="file"
                    accept=".pdf,.txt"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    disabled={loading}
                  />
                  <span className="file-icon">{file ? "✅" : "📁"}</span>
                  <span className="file-text">
                    {file ? (
                      <>
                        <strong>{file.name}</strong>
                        <br />
                        {formatBytes(file.size)}
                      </>
                    ) : (
                      <>
                        <strong>Click to upload</strong> or drag and drop
                      </>
                    )}
                  </span>
                </label>
              </div>
              
              <div className="btn-group">
                <button type="submit" disabled={!file || loading} className="btn primary">
                  {loading ? "⏳ Processing..." : "🚀 Upload & Process"}
                </button>
                {loading && (
                  <button type="button" className="btn secondary" onClick={cancelUpload}>
                    Cancel
                  </button>
                )}
              </div>
            </form>

            {loading && uploadProgress > 0 && (
              <div className="progress-container">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                </div>
                <p className="progress-text">{uploadStatus || `${uploadProgress}%`}</p>
              </div>
            )}

            {documents.length > 0 && (
              <div className="documents-section">
                <div className="documents-header">
                  <h3>📚 Your Documents</h3>
                  <span className="doc-count">{documents.length}</span>
                </div>
                <div className="documents-list">
                  {documents.map((doc) => (
                    <div key={doc.document_id} className="doc-item">
                      <span className="doc-icon">📄</span>
                      <div className="doc-info">
                        <div className="doc-name">{doc.filename}</div>
                        <div className="doc-meta">
                          <span>{formatBytes(doc.size_bytes)}</span>
                          <span>{new Date(doc.uploaded_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                      <button
                        className="btn danger sm"
                        onClick={() => handleDelete(doc.document_id)}
                        disabled={deletingId === doc.document_id}
                      >
                        {deletingId === doc.document_id ? "..." : "🗑️"}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {documents.length === 0 && !loading && (
              <div className="empty-state">
                <div className="empty-icon">📭</div>
                <p>No documents yet. Upload one to get started!</p>
              </div>
            )}
          </section>
        )}

        {tab === "ask" && (
          <section className="card">
            <h2><span className="card-icon">💡</span> Ask a Question</h2>
            <p className="muted">Get AI-powered answers from your documents</p>
            
            <form onSubmit={handleQuery}>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What would you like to know?"
                rows={3}
                disabled={loading}
              />
              <div className="btn-group">
                <button type="submit" disabled={loading || !question.trim()} className="btn primary">
                  {loading ? "⏳ Thinking..." : "✨ Get Answer"}
                </button>
                {isStreaming && (
                  <button type="button" className="btn secondary" onClick={cancelQuery}>
                    Stop
                  </button>
                )}
              </div>
            </form>

            {loading && queryProgress > 0 && (
              <div className="progress-container">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${queryProgress}%` }} />
                </div>
                <p className="progress-text">{isStreaming ? "Generating answer..." : "Searching..."}</p>
              </div>
            )}

            {streamingAnswer && (
              <div className="result">
                <h3>💬 Answer</h3>
                <div className="answer streaming">
                  {streamingAnswer}
                  <span className="cursor">▊</span>
                </div>
              </div>
            )}

            {queryResult && !streamingAnswer && (
              <div className="result">
                <h3>💬 Answer</h3>
                <div className="answer">{queryResult.answer}</div>
                
                {queryResult.sources.length > 0 && (
                  <div className="sources-list">
                    <h3>📎 Sources ({queryResult.sources.length})</h3>
                    {queryResult.sources.map((s, i) => (
                      <SourceItem key={i} chunk={s} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {!loading && !queryResult && !streamingAnswer && documents.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">📤</div>
                <p>Upload a document first to start asking questions</p>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-links">
          <a href="/docs" target="_blank" rel="noopener noreferrer">
            📖 API Docs
          </a>
          <a href="/health" target="_blank" rel="noopener noreferrer">
            ❤️ Health
          </a>
        </div>
      </footer>
    </div>
  );
}

function SourceItem({ chunk }: { chunk: SourceChunk }) {
  const [expanded, setExpanded] = useState(false);
  
  // Extract filename from source path
  const filename = chunk.source 
    ? chunk.source.split(/[/\\]/).pop() || chunk.source
    : "Unknown source";
  
  return (
    <div className="source-item">
      <div className="source-header" onClick={() => setExpanded(!expanded)}>
        <span className="source-label">
          📄 {filename}
        </span>
        {chunk.score && (
          <span className="source-score">
            {(chunk.score * 100).toFixed(0)}% match
          </span>
        )}
      </div>
      {expanded && (
        <div className="source-content">{chunk.text}</div>
      )}
      {!expanded && (
        <div className="source-content source-preview">{chunk.text}</div>
      )}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export default App;
