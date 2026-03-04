from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import os
import uuid
import json
import io
import base64
import requests
import chromadb
import ollama
import pypdf
from docx import Document
from bs4 import BeautifulSoup
from ddgs import DDGS
from datetime import datetime, timezone

# Allow OAuth over localhost (no HTTPS required)
os.environ.setdefault('OAUTHLIB_INSECURE_TRANSPORT', '1')

# Optional Google auth — app works without it, Gmail just won't be available
try:
    from google.oauth2.credentials import Credentials as GoogleCredentials
    from google_auth_oauthlib.flow import Flow as GoogleFlow
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build as google_build
    GOOGLE_AUTH_OK = True
except ImportError:
    GOOGLE_AUTH_OK = False
    print("⚠️  Google auth not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

# ── Config / Feed file paths ──────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
CONFIG_FILE      = os.path.join(BASE_DIR, "connector_config.json")
FEED_FILE        = os.path.join(BASE_DIR, "feed_events.json")

# ── Gmail paths ───────────────────────────────────────────────
GMAIL_SCOPES     = ['https://www.googleapis.com/auth/gmail.readonly']
GMAIL_TOKEN_FILE = os.path.join(BASE_DIR, "gmail_token.json")
GMAIL_CREDS_FILE = os.path.join(BASE_DIR, "gmail_credentials.json")

# ── Connector config helpers ──────────────────────────────────

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

# ── Feed event helpers ────────────────────────────────────────

def load_feed() -> list:
    if os.path.exists(FEED_FILE):
        try:
            with open(FEED_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_feed(events: list):
    with open(FEED_FILE, "w") as f:
        json.dump(events, f, indent=2)

def log_feed_event(title: str, workspace: str, chunks: int, connector: str = "manual", source_id: str = ""):
    events = load_feed()
    events.insert(0, {
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title":     title,
        "workspace": workspace,
        "chunks":    chunks,
        "connector": connector,
        "source_id": source_id,
    })
    save_feed(events[:300])  # keep last 300 events

# ── Notion sync ───────────────────────────────────────────────

def get_notion_title(page: dict) -> str:
    props = page.get("properties", {})
    for key in ["title", "Title", "Name", "name"]:
        if key in props:
            tp = props[key]
            if tp.get("type") == "title":
                return "".join(rt.get("plain_text", "") for rt in tp.get("title", []))
    return "Untitled"

def extract_block_text(block: dict) -> str:
    btype = block.get("type", "")
    rich  = block.get(btype, {}).get("rich_text", [])
    return "".join(rt.get("plain_text", "") for rt in rich)

def get_notion_page_text(page_id: str, headers: dict) -> str:
    try:
        r = requests.get(
            f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100",
            headers=headers, timeout=12
        )
        blocks = r.json().get("results", [])
        lines  = [extract_block_text(b) for b in blocks if extract_block_text(b)]
        return "\n".join(lines)
    except Exception as e:
        print(f"Notion block fetch error: {e}")
        return ""

def sync_notion_now(notion_cfg: dict):
    token     = notion_cfg.get("token", "")
    workspace = notion_cfg.get("workspace", "Notion")
    if not token:
        return 0

    headers = {
        "Authorization":  f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type":   "application/json",
    }

    try:
        r = requests.post(
            "https://api.notion.com/v1/search",
            headers=headers,
            json={"filter": {"value": "page", "property": "object"}, "page_size": 100},
            timeout=12
        )
        pages = r.json().get("results", [])
    except Exception as e:
        print(f"Notion search error: {e}")
        return 0

    col   = get_collection(workspace)
    synced = 0

    for page in pages:
        page_id = page["id"].replace("-", "")
        title   = get_notion_title(page)
        source  = f"notion:{page_id}"

        # Skip if already indexed
        existing = col.get(where={"source": source}, include=["metadatas"])
        if existing["ids"]:
            continue

        text = get_notion_page_text(page["id"], headers)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        print(f"  📓 Notion: indexing '{title}' ({len(chunks)} chunks)")
        embed_and_store(chunks, source, col)
        log_feed_event(title, workspace, len(chunks), "notion", source)
        synced += 1

    # Update last synced timestamp
    cfg = load_config()
    if "notion" not in cfg:
        cfg["notion"] = {}
    cfg["notion"]["last_synced"] = datetime.now(timezone.utc).isoformat()
    cfg["notion"]["pages_synced"] = (cfg["notion"].get("pages_synced", 0) + synced)
    save_config(cfg)
    print(f"✅ Notion sync complete — {synced} new pages indexed")
    return synced

# ── Gmail helpers ─────────────────────────────────────────────

def get_gmail_service():
    """Return authenticated Gmail API service, refreshing token if expired."""
    if not GOOGLE_AUTH_OK or not os.path.exists(GMAIL_TOKEN_FILE):
        return None
    try:
        creds = GoogleCredentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
            with open(GMAIL_TOKEN_FILE, 'w') as f:
                f.write(creds.to_json())
        return google_build('gmail', 'v1', credentials=creds) if creds and creds.valid else None
    except Exception as e:
        print(f"Gmail auth error: {e}")
        return None

def decode_email_body(payload: dict) -> str:
    """Extract plain-text body from a Gmail message payload."""
    data = payload.get('body', {}).get('data', '')
    if data:
        return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='ignore')
    for part in payload.get('parts', []):
        if part.get('mimeType') == 'text/plain':
            data = part.get('body', {}).get('data', '')
            if data:
                return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='ignore')
    return ''

def extract_email_text(msg_data: dict) -> tuple[str, str]:
    """Return (subject, full_text) from a Gmail message."""
    headers = msg_data.get('payload', {}).get('headers', [])
    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
    sender  = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
    date    = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
    body    = decode_email_body(msg_data.get('payload', {}))
    full    = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body[:3000]}"
    return subject, full

def sync_gmail_now(gmail_cfg: dict, max_emails: int = 100) -> int:
    """Fetch and index recent inbox emails. Returns count of newly indexed emails."""
    service = get_gmail_service()
    if not service:
        return 0
    workspace = gmail_cfg.get('workspace', 'Gmail')
    col       = get_collection(workspace)
    synced    = 0
    try:
        results  = service.users().messages().list(
            userId='me', maxResults=max_emails, q='is:inbox -is:spam'
        ).execute()
        messages = results.get('messages', [])
    except Exception as e:
        print(f"Gmail list error: {e}")
        return 0
    for msg in messages:
        msg_id   = msg['id']
        source   = f"gmail:{msg_id}"
        existing = col.get(where={"source": source}, include=["metadatas"])
        if existing["ids"]:
            continue
        try:
            msg_data           = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            subject, full_text = extract_email_text(msg_data)
            if not full_text.strip():
                continue
            chunks = chunk_text(full_text)
            embed_and_store(chunks, source, col)
            log_feed_event(subject, workspace, len(chunks), "gmail", source)
            synced += 1
            print(f"  📧 Gmail: indexed '{subject[:50]}'")
        except Exception as e:
            print(f"Gmail message error {msg_id}: {e}")
    cfg = load_config()
    if "gmail" not in cfg:
        cfg["gmail"] = {}
    cfg["gmail"]["last_synced"] = datetime.now(timezone.utc).isoformat()
    save_config(cfg)
    print(f"✅ Gmail sync complete — {synced} new emails indexed")
    return synced

# ── Background polling loop ───────────────────────────────────

async def polling_loop():
    """Runs in the background, polling connectors on their configured intervals."""
    print("🔄 Connector polling started")
    while True:
        try:
            cfg = load_config()

            # Notion
            notion = cfg.get("notion", {})
            if notion.get("enabled") and notion.get("token"):
                interval_min = notion.get("poll_interval_minutes", 15)
                last_synced  = notion.get("last_synced")
                should_sync  = True
                if last_synced:
                    try:
                        last_dt  = datetime.fromisoformat(last_synced)
                        elapsed  = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                        should_sync = elapsed >= interval_min
                    except Exception:
                        pass
                if should_sync:
                    print("🔄 Polling Notion...")
                    await asyncio.to_thread(sync_notion_now, notion)

            # Gmail
            gmail = cfg.get("gmail", {})
            if gmail.get("enabled") and gmail.get("connected"):
                interval_min = gmail.get("poll_interval_minutes", 30)
                last_synced  = gmail.get("last_synced")
                should_sync  = True
                if last_synced:
                    try:
                        last_dt     = datetime.fromisoformat(last_synced)
                        elapsed     = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                        should_sync = elapsed >= interval_min
                    except Exception:
                        pass
                if should_sync:
                    print("🔄 Polling Gmail...")
                    await asyncio.to_thread(sync_gmail_now, gmail)

        except Exception as e:
            print(f"Polling loop error: {e}")

        await asyncio.sleep(60)  # check every minute

# ── App lifespan (starts background polling) ──────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(polling_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────
FRONTEND_DIR  = os.path.join(BASE_DIR, "..", "frontend")
FRONTEND_FILE = os.path.join(FRONTEND_DIR, "index.html")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)
    return {"message": "VaultMind API running. Frontend not found."}

@app.get("/manifest.json", include_in_schema=False)
async def serve_manifest():
    manifest = os.path.join(FRONTEND_DIR, "manifest.json")
    if os.path.exists(manifest):
        return FileResponse(manifest, media_type="application/manifest+json")
    return {}

# ── ChromaDB ──────────────────────────────────────────────────
chroma = chromadb.PersistentClient(path="./chroma_db")

EMBED_MODEL   = "nomic-embed-text"
DEFAULT_MODEL = "mistral"

AVAILABLE_MODELS = [
    {"id": "mistral",     "label": "Mistral 7B"},
    {"id": "llama3.2",    "label": "Llama 3.2"},
    {"id": "phi3",        "label": "Phi-3 Mini"},
    {"id": "gemma2",      "label": "Gemma 2"},
    {"id": "qwen2.5",     "label": "Qwen 2.5"},
    {"id": "deepseek-r1", "label": "DeepSeek R1"},
]

# ── Workspace helpers ─────────────────────────────────────────

def collection_name(workspace: str) -> str:
    if not workspace or workspace.strip().lower() in ("default", ""):
        return "vaultmind_docs"
    safe = workspace.strip().lower().replace(" ", "_").replace("-", "_")
    return f"vaultmind_{safe}"

def get_collection(workspace: str = "Default"):
    return chroma.get_or_create_collection(collection_name(workspace))

def workspace_from_collection(col_name: str) -> str:
    if col_name == "vaultmind_docs":
        return "Default"
    if col_name.startswith("vaultmind_"):
        return col_name[len("vaultmind_"):].replace("_", " ").title()
    return col_name

# ── Workspaces API ────────────────────────────────────────────

@app.get("/workspaces")
async def list_workspaces():
    try:
        cols  = chroma.list_collections()
        names = [workspace_from_collection(c.name) for c in cols
                 if c.name == "vaultmind_docs" or c.name.startswith("vaultmind_")]
    except Exception:
        names = []
    if not names:
        names = ["Default"]
    if "Default" in names:
        names = ["Default"] + [n for n in names if n != "Default"]
    return {"workspaces": names}

class WorkspaceCreate(BaseModel):
    name: str

@app.post("/workspaces")
async def create_workspace(data: WorkspaceCreate):
    name = data.name.strip()
    if not name:
        return {"error": "Workspace name cannot be empty"}
    get_collection(name)
    return {"message": f"Workspace '{name}' created", "name": name}

# ── Models API ────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    try:
        pulled_raw   = ollama.list()
        pulled_names = [m.model for m in pulled_raw.models]
    except Exception:
        pulled_names = []
    models = [{**m, "available": any(m["id"] in p for p in pulled_names)} for m in AVAILABLE_MODELS]
    return {"models": models, "default": DEFAULT_MODEL}

# ── Text helpers ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 150) -> list[str]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 20):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def extract_text_from_file(contents: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(contents))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(contents))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif name.endswith((".txt", ".md")):
        return contents.decode("utf-8", errors="ignore")
    elif name.endswith(".csv"):
        return contents.decode("utf-8", errors="ignore")
    return ""

def embed_and_store(chunks: list[str], source: str, col):
    for i, chunk in enumerate(chunks):
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        col.upsert(
            ids=[f"{source}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": source, "chunk": i}]
        )
        if i % 10 == 0:
            print(f"  ✓ {i}/{len(chunks)}")

# ── Upload ────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(
    file:      UploadFile = File(...),
    workspace: str        = Form(default="Default")
):
    contents = await file.read()
    text     = extract_text_from_file(contents, file.filename)
    if not text.strip():
        return {"error": f"Could not extract text from '{file.filename}'. Supported: PDF, DOCX, TXT, MD, CSV"}
    chunks = chunk_text(text)
    print(f"\n📄 [{workspace}] Indexing '{file.filename}' — {len(chunks)} chunks")
    col = get_collection(workspace)
    embed_and_store(chunks, file.filename, col)
    log_feed_event(file.filename, workspace, len(chunks), "upload")
    return {"message": f"Indexed {file.filename}", "chunks": len(chunks)}

# ── URL ingest ────────────────────────────────────────────────

class UrlIngest(BaseModel):
    url:       str
    workspace: str = "Default"

@app.post("/ingest-url")
async def ingest_url(data: UrlIngest):
    BLOCKED = ["indeed.com", "linkedin.com", "ziprecruiter.com", "glassdoor.com"]
    if any(d in data.url for d in BLOCKED):
        return {"error": "This site blocks scrapers. Try company career pages, Builtin, or Wellfound instead."}
    try:
        r = requests.get(data.url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
        })
        if r.status_code == 403:
            return {"error": "This site blocks scrapers (403). Try the company's direct careers page."}
        if r.status_code == 429:
            return {"error": "Rate limited (429). Wait a minute and try again."}
        r.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": f"Could not fetch URL: {str(e)}"}

    soup  = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else data.url
    text  = soup.get_text(separator="\n", strip=True)
    if not text.strip():
        return {"error": "No readable content found at that URL."}

    source = f"🌐 {title[:80]}"
    chunks = chunk_text(text)
    col    = get_collection(data.workspace)
    embed_and_store(chunks, source, col)
    log_feed_event(title, data.workspace, len(chunks), "url")
    return {"message": f"Indexed {title}", "chunks": len(chunks), "source": source}

# ── Files ─────────────────────────────────────────────────────

@app.get("/files")
async def list_files(workspace: str = Query(default="Default")):
    col     = get_collection(workspace)
    results = col.get(include=["metadatas"])
    files   = {}
    for meta in results["metadatas"]:
        src = meta["source"]
        files[src] = files.get(src, 0) + 1
    return {"files": [{"name": k, "chunks": v} for k, v in files.items()]}

@app.delete("/files/{filename}")
async def delete_file(filename: str, workspace: str = Query(default="Default")):
    col     = get_collection(workspace)
    results = col.get(where={"source": filename}, include=["metadatas"])
    ids     = results["ids"]
    if not ids:
        return {"error": "File not found"}
    col.delete(ids=ids)
    return {"message": f"Deleted {filename}", "chunks_removed": len(ids)}

# ── Feed API ──────────────────────────────────────────────────

@app.get("/feed")
async def get_feed(limit: int = Query(default=50)):
    """Return recent feed events from all connectors and manual uploads."""
    events = load_feed()
    return {"events": events[:limit]}

# ── Connectors API ────────────────────────────────────────────

@app.get("/connectors")
async def get_connectors():
    """Return current connector configurations (tokens masked)."""
    cfg    = load_config()
    result = {}
    for name, conf in cfg.items():
        masked = {k: ("***" if "token" in k or "key" in k or "secret" in k else v)
                  for k, v in conf.items()}
        result[name] = masked
    return {"connectors": result}

class ConnectorConfig(BaseModel):
    connector:             str
    enabled:               bool  = True
    token:                 str   = ""
    workspace:             str   = ""
    poll_interval_minutes: int   = 15

@app.post("/connectors")
async def save_connector(data: ConnectorConfig):
    """Save connector configuration."""
    cfg = load_config()
    if data.connector not in cfg:
        cfg[data.connector] = {}
    cfg[data.connector].update({
        "enabled":               data.enabled,
        "token":                 data.token,
        "workspace":             data.workspace or data.connector.title(),
        "poll_interval_minutes": data.poll_interval_minutes,
    })
    save_config(cfg)
    # Ensure workspace collection exists
    if data.workspace:
        get_collection(data.workspace)
    return {"message": f"{data.connector} connector saved"}

@app.post("/connectors/{connector}/sync")
async def manual_sync(connector: str):
    """Trigger an immediate manual sync for a connector."""
    cfg = load_config()
    if connector == "notion":
        notion_cfg = cfg.get("notion", {})
        if not notion_cfg.get("token"):
            return {"error": "Notion token not configured"}
        synced = await asyncio.to_thread(sync_notion_now, notion_cfg)
        return {"message": f"Notion sync complete — {synced} new pages indexed", "synced": synced}
    if connector == "gmail":
        gmail_cfg = cfg.get("gmail", {})
        if not gmail_cfg.get("connected"):
            return {"error": "Gmail not connected. Authorize first via ⚙️ settings."}
        synced = await asyncio.to_thread(sync_gmail_now, gmail_cfg)
        return {"message": f"Gmail sync complete — {synced} new emails indexed", "synced": synced}
    return {"error": f"Unknown connector: {connector}"}

# ── Gmail OAuth ────────────────────────────────────────────────

class GmailOAuthConfig(BaseModel):
    client_id:             str
    client_secret:         str
    workspace:             str = "Gmail"
    poll_interval_minutes: int = 30

@app.post("/auth/gmail/configure")
async def configure_gmail(data: GmailOAuthConfig):
    """Save Google OAuth client credentials so the auth flow can begin."""
    if not GOOGLE_AUTH_OK:
        return {"error": "Google auth libraries not installed. Restart backend after: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"}
    creds_data = {
        "installed": {
            "client_id":     data.client_id.strip(),
            "client_secret": data.client_secret.strip(),
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8000/auth/gmail/callback"]
        }
    }
    with open(GMAIL_CREDS_FILE, 'w') as f:
        json.dump(creds_data, f)
    cfg = load_config()
    cfg["gmail"] = cfg.get("gmail", {})
    cfg["gmail"]["workspace"]             = data.workspace
    cfg["gmail"]["poll_interval_minutes"] = data.poll_interval_minutes
    save_config(cfg)
    get_collection(data.workspace)
    return {"message": "Gmail credentials saved"}

@app.get("/auth/gmail")
async def gmail_auth_start():
    """Return the Google OAuth URL for the frontend to open."""
    if not GOOGLE_AUTH_OK:
        return {"error": "Google auth libraries not installed"}
    if not os.path.exists(GMAIL_CREDS_FILE):
        return {"error": "Save your Client ID and Secret first"}
    try:
        flow     = GoogleFlow.from_client_secrets_file(
            GMAIL_CREDS_FILE, scopes=GMAIL_SCOPES,
            redirect_uri="http://localhost:8000/auth/gmail/callback"
        )
        auth_url, _ = flow.authorization_url(
            access_type='offline', include_granted_scopes='true', prompt='consent'
        )
        return {"auth_url": auth_url}
    except Exception as e:
        return {"error": str(e)}

@app.get("/auth/gmail/callback")
async def gmail_auth_callback(code: str = Query(default=""), error: str = Query(default="")):
    """Handle Google OAuth redirect — saves token and shows a close-tab page."""
    if error:
        return HTMLResponse(f"""
<html><body style="font-family:sans-serif;background:#0f0f0f;color:#e0e0e0;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center">
<div><div style="font-size:48px">❌</div><h2>Authorization failed</h2>
<p style="color:#666">{error}</p>
<script>setTimeout(()=>window.close(),3000)</script></div></body></html>""")
    if not code or not os.path.exists(GMAIL_CREDS_FILE):
        return HTMLResponse("<p>Missing code or credentials file. Close this tab and try again.</p>")
    try:
        flow = GoogleFlow.from_client_secrets_file(
            GMAIL_CREDS_FILE, scopes=GMAIL_SCOPES,
            redirect_uri="http://localhost:8000/auth/gmail/callback"
        )
        flow.fetch_token(code=code)
        creds = flow.credentials
        with open(GMAIL_TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
        cfg = load_config()
        cfg["gmail"]              = cfg.get("gmail", {})
        cfg["gmail"]["connected"] = True
        cfg["gmail"]["enabled"]   = True
        save_config(cfg)
        return HTMLResponse("""
<html><body style="font-family:-apple-system,sans-serif;background:#0f0f0f;color:#e0e0e0;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center">
<div>
  <div style="font-size:64px">✅</div>
  <h2 style="font-weight:700;margin:16px 0 8px">Gmail connected!</h2>
  <p style="color:#666">You can close this tab.</p>
  <script>setTimeout(()=>{try{window.close()}catch(e){}},1500)</script>
</div></body></html>""")
    except Exception as e:
        return HTMLResponse(f"<p>Error: {e}<br>Close this tab and try again.</p>")

@app.get("/auth/gmail/status")
async def gmail_auth_status():
    """Check if a valid Gmail token exists."""
    if not GOOGLE_AUTH_OK or not os.path.exists(GMAIL_TOKEN_FILE):
        return {"connected": False}
    try:
        creds = GoogleCredentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
        if creds and (creds.valid or (creds.expired and creds.refresh_token)):
            return {"connected": True}
    except Exception:
        pass
    return {"connected": False}

@app.post("/auth/gmail/disconnect")
async def gmail_disconnect():
    """Remove Gmail token and mark as disconnected."""
    if os.path.exists(GMAIL_TOKEN_FILE):
        os.remove(GMAIL_TOKEN_FILE)
    if os.path.exists(GMAIL_CREDS_FILE):
        os.remove(GMAIL_CREDS_FILE)
    cfg = load_config()
    if "gmail" in cfg:
        cfg["gmail"]["connected"] = False
        cfg["gmail"]["enabled"]   = False
        save_config(cfg)
    return {"message": "Gmail disconnected"}

@app.delete("/connectors/{connector}")
async def delete_connector(connector: str):
    """Remove a connector configuration."""
    cfg = load_config()
    if connector in cfg:
        del cfg[connector]
        save_config(cfg)
    return {"message": f"{connector} connector removed"}

# ── Chat ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message:       str
    history:       list[dict] = []
    workspace:     str = "Default"
    model:         str = "mistral"
    pinned_source: str = ""   # when set, restrict retrieval to this exact source

@app.post("/chat")
async def chat(msg: ChatMessage):
    col                = get_collection(msg.workspace)
    chat_model         = msg.model or DEFAULT_MODEL
    question_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]

    # If the user clicked a specific feed item, pin retrieval to that source only
    if msg.pinned_source:
        results = col.query(
            query_embeddings=[question_embedding],
            n_results=10,
            where={"source": msg.pinned_source}
        )
        # Fall back to normal search if pinned source has no chunks
        if not results["documents"][0]:
            results = col.query(query_embeddings=[question_embedding], n_results=6)
    else:
        results = col.query(query_embeddings=[question_embedding], n_results=6)

    if not results["documents"][0]:
        def no_docs():
            yield f"data: {json.dumps({'token': 'No documents indexed yet in this workspace. Upload a file or paste a URL to get started.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    context = "\n\n---\n\n".join(results["documents"][0])
    sources  = list(set(m["source"] for m in results["metadatas"][0]))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal AI assistant. You have access ONLY to documents the user has explicitly indexed.\n\n"
                "STRICT RULES:\n"
                "1. NEVER invent, fabricate, or guess information.\n"
                "2. ONLY use information literally present in the documents below.\n"
                "3. If the answer isn't in the documents, say so clearly.\n"
                "4. Be concise and direct.\n"
                "5. Write in plain prose only. NO markdown — no bold, no headers, no bullet dashes.\n\n"
                f"INDEXED DOCUMENTS:\n{context}"
            )
        }
    ]
    for h in msg.history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": msg.message})

    def generate():
        stream = ollama.chat(model=chat_model, messages=messages, stream=True, options={"temperature": 0})
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Inbox Digest ──────────────────────────────────────────────

class DigestRequest(BaseModel):
    workspace: str = "Gmail"
    connector: str = "gmail"   # source prefix to filter on
    model:     str = "mistral"

@app.post("/digest")
async def digest(req: DigestRequest):
    """Retrieve ALL emails from a connector, group by source, and stream a ranked summary."""
    col = get_collection(req.workspace)

    # Pull every chunk in this workspace and keep only the target connector's
    all_results = col.get(include=["documents", "metadatas"])
    prefix = f"{req.connector}:"

    # Group chunks by source — first chunk of each email has the From/Date/Subject header
    emails: dict[str, list[str]] = {}
    for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
        src = meta.get("source", "")
        if src.startswith(prefix):
            if src not in emails:
                emails[src] = []
            emails[src].append(doc)

    if not emails:
        def no_emails():
            yield f"data: {json.dumps({'token': 'No Gmail emails are indexed yet. Open Settings → Gmail and click Sync Now to pull your inbox.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
        return StreamingResponse(no_emails(), media_type="text/event-stream")

    # Build compact per-email context: first chunk only (has From/Date/Subject + opening lines)
    # Cap at 40 emails to stay within local model context limits (~4k tokens)
    email_blocks = []
    for src, chunks in list(emails.items())[:40]:
        email_blocks.append(chunks[0])

    context = "\n\n---EMAIL---\n\n".join(email_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal AI assistant acting as an executive email briefer.\n"
                "The user's indexed inbox emails are below, separated by ---EMAIL---.\n\n"
                "Your job:\n"
                "1. Write a 2-sentence plain-English summary of the overall inbox (what kind of day is it?)\n"
                "2. Stack-rank every email from most to least important\n"
                "3. Format each ranked item as:\n"
                "   [PRIORITY] Subject — one sentence on what action (if any) is needed\n\n"
                "Priority levels — pick one per email:\n"
                "🔴 ACTION REQUIRED — interviews, job offers, payments due, verification codes, deadlines\n"
                "🟡 READ TODAY — newsletters, updates, replies worth reading\n"
                "⚫ LOW — promotions, marketing, automated digests, receipts\n\n"
                "Rules:\n"
                "- Be brutally concise. One line per email.\n"
                "- Put 🔴 items at the top, ⚫ at the bottom.\n"
                "- Do NOT use markdown bold or headers. Plain text only.\n"
                "- Do NOT invent content not present in the emails.\n\n"
                f"EMAILS:\n{context}"
            )
        },
        {
            "role": "user",
            "content": "Summarize my inbox and tell me what to focus on today."
        }
    ]

    chat_model = req.model or DEFAULT_MODEL

    def generate():
        stream = ollama.chat(model=chat_model, messages=messages, stream=True, options={"temperature": 0})
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': list(emails.keys())[:40]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Status / Health ───────────────────────────────────────────

@app.get("/status")
async def status(workspace: str = Query(default="Default")):
    try:
        count = get_collection(workspace).count()
    except Exception:
        count = 0
    return {"chunks_indexed": count, "status": "running", "workspace": workspace}

@app.get("/health")
async def health():
    try:
        model_names = [m.model for m in ollama.list().models]
        has_embed   = any("nomic-embed-text" in m for m in model_names)
        has_llm     = any(any(x in m for x in ["mistral", "llama3", "phi3", "gemma", "qwen", "deepseek"]) for m in model_names)
        return {"ollama": True, "embed_model": has_embed, "chat_model": has_llm, "ready": has_embed and has_llm}
    except Exception:
        return {"ollama": False, "embed_model": False, "chat_model": False, "ready": False}

# ── Query (non-streaming) ─────────────────────────────────────

class QueryMessage(BaseModel):
    message:   str
    mode:      str = "vault"
    workspace: str = "Default"
    model:     str = "mistral"

@app.post("/query")
async def query(msg: QueryMessage):
    try:
        col        = get_collection(msg.workspace)
        chat_model = msg.model or DEFAULT_MODEL
        q_emb      = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]

        RELEVANCE_THRESHOLD = 0.75
        vault_context = ""
        vault_sources = []
        v = col.query(query_embeddings=[q_emb], n_results=4, include=["documents", "metadatas", "distances"])
        if v["documents"][0]:
            rel_docs, rel_meta = [], []
            for doc, meta, dist in zip(v["documents"][0], v["metadatas"][0], v["distances"][0]):
                if dist < RELEVANCE_THRESHOLD:
                    rel_docs.append(doc); rel_meta.append(meta)
            if rel_docs:
                vault_context = "\n\n".join(rel_docs)
                vault_sources = list(set(m["source"] for m in rel_meta))

        web_context = ""
        web_sources = []
        if msg.mode == "agent":
            for hit in web_search(msg.message, 4)[:3]:
                text = smart_scrape(hit.get("href", ""), 1500)
                if text:
                    web_context += f"\n\nSource: {hit.get('title','')}\n{text}"
                    web_sources.append(hit.get("title", hit.get("href", "")))

        sections = []
        if vault_context: sections.append(f"FROM YOUR PRIVATE DOCUMENTS:\n{vault_context}")
        if web_context:   sections.append(f"FROM THE WEB:\n{web_context}")
        if not sections:
            return {"answer": "No relevant information indexed for that question.", "sources": []}

        response = ollama.chat(model=chat_model, messages=[
            {"role": "system", "content": f"Answer using ONLY the sources below. No markdown.\n\nSOURCES:\n{chr(10).join(sections)}"},
            {"role": "user", "content": msg.message}
        ], options={"temperature": 0})
        return {"answer": response["message"]["content"], "sources": vault_sources + web_sources, "mode": msg.mode}

    except Exception as e:
        return {"error": str(e), "answer": "VaultMind encountered an error. Is Ollama running?"}

# ── Agent (streaming) ─────────────────────────────────────────

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
}

def web_search(query: str, max_results: int = 6) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"Search error: {e}"); return []

def smart_scrape(url: str, max_chars: int = 2000) -> str:
    BLOCKED = ["linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com"]
    if any(d in url for d in BLOCKED): return ""
    try:
        r = requests.get(url, timeout=8, headers=BROWSER_HEADERS)
        if r.status_code != 200: return ""
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:max_chars]
    except Exception: return ""

@app.post("/agent")
async def agent(msg: ChatMessage):
    col        = get_collection(msg.workspace)
    chat_model = msg.model or DEFAULT_MODEL

    def generate():
        vault_context = ""
        vault_sources = []
        RELEVANCE_THRESHOLD = 0.75
        try:
            q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]
            v     = col.query(query_embeddings=[q_emb], n_results=4, include=["documents", "metadatas", "distances"])
            if v["documents"][0]:
                rel_docs, rel_meta = [], []
                for doc, meta, dist in zip(v["documents"][0], v["metadatas"][0], v["distances"][0]):
                    if dist < RELEVANCE_THRESHOLD:
                        rel_docs.append(doc); rel_meta.append(meta)
                if rel_docs:
                    vault_context = "\n\n".join(rel_docs)
                    vault_sources = list(set(m["source"] for m in rel_meta))
        except Exception: pass

        yield f"data: {json.dumps({'status': '🔍 Searching the web...'})}\n\n"
        search_hits = web_search(msg.message, 6)
        if not search_hits:
            yield f"data: {json.dumps({'status': '⚠️ No web results, using vault only.'})}\n\n"

        web_context = ""
        web_sources = []
        scraped     = 0
        for hit in search_hits:
            if scraped >= 3: break
            url   = hit.get("href", "")
            title = hit.get("title", url)
            yield f"data: {json.dumps({'status': f'📄 Reading: {title[:50]}...'})}\n\n"
            page_text = smart_scrape(url) or hit.get("body", "")
            if page_text:
                web_context += f"\n\nSource: {title}\nURL: {url}\n{page_text}"
                web_sources.append(f"[{title}]({url})")
                scraped += 1

        yield f"data: {json.dumps({'status': '💬 Generating answer...'})}\n\n"

        sections = []
        if vault_context: sections.append(f"FROM YOUR PRIVATE DOCUMENTS:\n{vault_context}")
        if web_context:   sections.append(f"FROM THE WEB:\n{web_context}")

        if not sections:
            yield f"data: {json.dumps({'token': 'No relevant information found.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
            return

        messages = [
            {"role": "system", "content": (
                "You are a personal AI agent. Synthesize information from both private docs and web results.\n"
                "NEVER hallucinate. Be direct. Plain prose only — no markdown.\n\n"
                f"SOURCES:\n\n{'---'.join(sections)}"
            )}
        ]
        for h in msg.history[-6:]: messages.append(h)
        messages.append({"role": "user", "content": msg.message})

        stream = ollama.chat(model=chat_model, messages=messages, stream=True, options={"temperature": 0})
        for chunk in stream:
            token = chunk["message"]["content"]
            if token: yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': vault_sources + web_sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
