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
import zipfile
import xml.etree.ElementTree as ET
import threading
import requests
import chromadb
import ollama
import pypdf
from docx import Document
from bs4 import BeautifulSoup
from ddgs import DDGS
from datetime import datetime, timezone
from urllib.parse import urlparse
from company_intel import analyze_agency_listings
from router import route_file, route_query, RouteType, IMAGE_EXTENSIONS
from vlm import extract_pdf_with_vlm, extract_image_with_vlm, get_available_vlm, vlm_available
from query_intelligence import (
    classify_query, build_prompt, get_prompt_template,
    QueryIntent, ComplexityLevel, QueryClassification,
    get_available_models as qi_get_available_models,
)
from conversation_memory import (
    store_conversation_memory, recall_relevant_memories,
    build_memory_context, get_memory_collection,
)
from privacy_firewall import sanitize_for_search as firewall_sanitize, load_config as load_firewall_config
from search_proxy import privacy_search, load_search_config
from context_fusion import fuse_contexts, build_fusion_prompt, FusedContext
from search_quality import classify_domain
from quality_gate import run_quality_gate, ConfidenceLevel
from citation_engine import cite_response, format_sources_for_frontend
from adaptive_prompts import build_adaptive_prompt, get_retry_prompt, truncate_context
from feedback_loop import (
    store_feedback, FeedbackEntry, get_feedback_summary,
    get_route_stats, get_best_route, get_routing_overrides,
    get_insights, export_training_jsonl,
)
from knowledge_graph import (
    load_graph, save_graph, add_document_to_graph,
    find_related, get_document_connections, get_graph_stats,
    build_context_from_graph, rebuild_graph as kg_rebuild,
)
from finetune_pipeline import (
    check_readiness as ft_check_readiness,
    get_status as ft_get_status,
    run_training as ft_run_training,
    get_training_history as ft_get_history,
)
from proactive_intel import (
    run_proactive_scan, get_proactive_summary,
    get_alerts, get_unread_count, mark_read, dismiss_alert, mark_all_read,
    add_folder_watch, remove_folder_watch, get_watches,
    add_topic_watch, remove_topic_watch,
    add_deadline, check_deadlines, extract_deadlines,
)
from rbac import (
    create_user, authenticate, generate_token, validate_token,
    check_permission, get_audit_log, assign_workspace,
    get_user, list_users, update_role, Role,
)
from doc_compare import compare_documents, export_comparison_markdown
from export_layer import (
    export_chat_to_memo, export_analysis_to_report,
    export_research_to_brief, list_templates,
    register_webhook, fire_webhook, generate_api_key,
)
from vertical_kit import (
    load_profile, get_active_profile, set_active_profile,
    list_profiles, create_custom_profile,
)
from photo_pipeline import process_photo, process_queue as photo_process_queue
from call_intel import process_transcript as ci_process_transcript, get_call_history
from mobile_alerts import (
    register_device as ma_register_device,
    queue_alert_for_device, get_pending_alerts as ma_get_pending,
    mark_delivered as ma_mark_delivered,
    configure_preferences as ma_configure_prefs,
)
from contact_intel import (
    add_contact, search_contacts, generate_briefing,
    log_interaction, import_contacts as ci_import_contacts,
    get_contact_history,
)
from lam import (
    run_lam_agent, load_staged, approve_staged_action,
    reject_staged_action, AUDIT_DIR
)

# Optional Pillow for EXIF
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PILLOW_OK = True
except ImportError:
    PILLOW_OK = False
    print("⚠️  Pillow not installed. Run: pip install Pillow")

# Optional watchdog for folder watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_OK = True
except ImportError:
    WATCHDOG_OK = False
    print("⚠️  watchdog not installed. Run: pip install watchdog")

IMAGE_EXTENSIONS  = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.tiff', '.tif'}
WATCHABLE_EXTS    = {'.pdf', '.docx', '.txt', '.md', '.csv'} | IMAGE_EXTENSIONS
WATCH_FOLDERS_KEY = "watch_folders"

_watcher_observer: "Observer | None" = None
_active_watchers:  dict               = {}   # folder_path → handler

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
BASE_DIR = os.path.dirname(__file__)

# DATA_DIR is configurable so the Electron app can point it at
# ~/Library/Application Support/VaultMind/data instead of the app bundle.
DATA_DIR         = os.environ.get("VAULTMIND_DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

CONFIG_FILE      = os.path.join(DATA_DIR, "connector_config.json")
FEED_FILE        = os.path.join(DATA_DIR, "feed_events.json")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# ── Gmail paths ───────────────────────────────────────────────
GMAIL_SCOPES     = ['https://www.googleapis.com/auth/gmail.readonly']
GMAIL_TOKEN_FILE = os.path.join(DATA_DIR, "gmail_token.json")
GMAIL_CREDS_FILE = os.path.join(DATA_DIR, "gmail_credentials.json")

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
    token = notion_cfg.get("token", "")
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

    col    = get_collection()
    synced = 0

    for page in pages:
        page_id = page["id"].replace("-", "")
        title   = get_notion_title(page)
        source  = f"notion:{page_id}"

        existing = col.get(where={"source": source}, include=["metadatas"])
        if existing["ids"]:
            continue

        text = get_notion_page_text(page["id"], headers)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        print(f"  📓 Notion: indexing '{title}' ({len(chunks)} chunks)")
        embed_and_store(chunks, source, col)
        log_feed_event(title, "vault", len(chunks), "notion", source)
        synced += 1

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
    """Fetch and index recent inbox emails into the single vault collection."""
    service = get_gmail_service()
    if not service:
        return 0
    col    = get_collection()
    synced = 0
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
            log_feed_event(subject, "vault", len(chunks), "gmail", source)
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
    # Start background polling
    task = asyncio.create_task(polling_loop())
    # Start any saved folder watchers
    cfg = load_config()
    for folder in cfg.get(WATCH_FOLDERS_KEY, []):
        if os.path.isdir(folder):
            start_folder_watcher(folder)
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    if _watcher_observer is not None:
        _watcher_observer.stop()
        _watcher_observer.join()

app = FastAPI(lifespan=lifespan)
print(f"\n{'='*50}")
print(f"  VaultMind v1.0.0 starting up…")
print(f"{'='*50}\n")

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
chroma = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))

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

# ── Single vault collection ───────────────────────────────────
# Everything lives in one collection — docs, emails, web pages, all of it.

def get_collection():
    return chroma.get_or_create_collection("vaultmind_vault")

# ── Models API ────────────────────────────────────────────────

@app.get("/debug-chunks")
async def debug_chunks(source: str = Query(...), limit: int = Query(default=5)):
    """Return raw chunk text for a source — for debugging what was actually indexed."""
    col = get_collection()
    results = col.get(where={"source": source}, include=["documents", "metadatas"])
    docs = results["documents"][:limit]
    return {"source": source, "total_chunks": len(results["documents"]), "sample_chunks": docs}

@app.get("/vlm-status")
async def vlm_status():
    """Returns which VLM model is available and whether VLM processing is ready."""
    model = get_available_vlm()
    return {
        "vlm_available": model is not None,
        "vlm_model": model,
        "install_hint": "ollama pull qwen2.5vl:7b" if not model else None,
    }

# ── LAM Agent Endpoints ───────────────────────────────────────

class AgentRequest(BaseModel):
    query:     str
    matter_id: str = ""
    model:     str = ""

@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    """
    LAM agent mode: plan and execute multi-step actions.
    Auto-executes low-risk tools, stages high-risk ones for review.
    """
    # First do RAG to get context
    col = get_collection()
    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=req.query)["embedding"]
        results = col.query(query_embeddings=[embedding], n_results=5)
        chunks = results["documents"][0] if results["documents"] else []
    except Exception:
        chunks = []

    result = await asyncio.to_thread(
        run_lam_agent, req.query, chunks, req.matter_id, req.model or None
    )
    return result

@app.get("/staged-actions")
async def list_staged():
    """Return all pending staged actions awaiting attorney approval."""
    actions = load_staged()
    pending = [a for a in actions if a["status"] == "pending"]
    return {"staged": pending, "count": len(pending)}

class ActionDecision(BaseModel):
    action_id: str

@app.post("/staged-actions/{action_id}/approve")
async def approve_action(action_id: str):
    result = await asyncio.to_thread(approve_staged_action, action_id)
    return result

@app.post("/staged-actions/{action_id}/reject")
async def reject_action(action_id: str):
    result = await asyncio.to_thread(reject_staged_action, action_id)
    return result

@app.get("/audit-log")
async def get_audit_log(limit: int = Query(default=50)):
    """Return recent LAM audit records."""
    import glob
    audit_files = sorted(
        glob.glob(os.path.join(AUDIT_DIR, "*.json")),
        key=os.path.getmtime,
        reverse=True
    )[:limit]
    records = []
    for f in audit_files:
        try:
            with open(f) as fp:
                records.append(json.load(fp))
        except Exception:
            pass
    return {"records": records, "total": len(audit_files)}

@app.post("/classify")
async def classify_endpoint(msg: dict):
    """Classify a query's intent and complexity without running the full chat pipeline.
    Useful for the frontend to show real-time intent badges."""
    query = msg.get("message", "")
    history = msg.get("history", [])
    if not query:
        return {"error": "No message provided"}
    result = classify_query(query=query, conversation_history=history or None)
    return {
        "intent": result.intent.value,
        "complexity": result.complexity.value,
        "confidence": result.confidence,
        "recommended_model": result.recommended_model,
        "needs_web": result.needs_web,
        "needs_vault": result.needs_vault,
        "reasoning": result.reasoning,
    }

# ── Privacy Firewall API ─────────────────────────────────────

from privacy_firewall import (
    load_config as _fw_load_config, save_config as _fw_save_config,
    load_blocklist as _fw_load_blocklist, save_blocklist as _fw_save_blocklist,
    sanitize as _fw_sanitize, get_audit_log as _fw_get_audit_log,
    scan_for_data_leakage,
)

@app.get("/firewall/config")
async def get_firewall_config():
    return _fw_load_config()

@app.post("/firewall/config")
async def update_firewall_config(config: dict):
    _fw_save_config(config)
    return {"ok": True}

@app.get("/firewall/blocklist")
async def get_firewall_blocklist():
    return {"terms": _fw_load_blocklist()}

@app.post("/firewall/blocklist")
async def update_firewall_blocklist(data: dict):
    terms = data.get("terms", [])
    _fw_save_blocklist(terms)
    return {"ok": True, "count": len(terms)}

@app.post("/firewall/test")
async def test_firewall(data: dict):
    """Test the firewall on a sample text without searching."""
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    result = _fw_sanitize(text)
    return {
        "sanitized": result.sanitized_text,
        "entities_found": result.entity_count,
        "was_modified": result.was_modified,
        "entities": [
            {"type": e.entity_type.value if hasattr(e.entity_type, "value") else e.entity_type,
             "replacement": e.replacement, "confidence": e.confidence}
            for e in result.entities_found
        ],
    }

@app.get("/firewall/audit")
async def get_firewall_audit(limit: int = 50):
    return {"entries": _fw_get_audit_log(limit)}

@app.post("/firewall/scan")
async def scan_data_leakage(data: dict):
    """AIR Blackbox-compatible data leakage scan."""
    text = data.get("text", "")
    context = data.get("context", "prompt")
    if not text:
        return {"error": "No text provided"}
    return scan_for_data_leakage(text, context)

# ── Phase 3: Quality Gate + Citation API Endpoints ────────────

class QualityCheckRequest(BaseModel):
    response: str
    question: str
    context: str = ""
    local_context: str = ""
    web_context: str = ""
    sources: list = []
    use_llm: bool = False

@app.post("/quality/check")
async def quality_check(req: QualityCheckRequest):
    """Run the quality gate on any response (for testing or external use)."""
    verdict = run_quality_gate(
        response=req.response,
        question=req.question,
        context=req.context,
        local_context=req.local_context,
        web_context=req.web_context,
        sources=req.sources,
        use_llm=req.use_llm,
    )
    return verdict.to_dict()

class CitationRequest(BaseModel):
    response: str
    local_chunks: list = []
    web_results: list = []

@app.post("/citations/generate")
async def generate_citations(req: CitationRequest):
    """Generate citations for any response (for testing or external use)."""
    cited = cite_response(req.response, req.local_chunks, req.web_results)
    return format_sources_for_frontend(cited)

# ── Phase 4: Feedback Loop Endpoints ──────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    response: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    conversation_id: str = ""
    intent: str = ""
    complexity: str = ""
    model: str = ""
    prompt_template: str = ""
    quality_score: float = 0.0
    quality_confidence: str = ""
    mode: str = ""
    sources_used: int = 0
    user_comment: str = ""

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Store user feedback on a response."""
    entry = FeedbackEntry(
        question=req.question, response=req.response, rating=req.rating,
        conversation_id=req.conversation_id, intent=req.intent,
        complexity=req.complexity, model=req.model,
        prompt_template=req.prompt_template, quality_score=req.quality_score,
        quality_confidence=req.quality_confidence, mode=req.mode,
        sources_used=req.sources_used, user_comment=req.user_comment,
    )
    feedback_id = store_feedback(entry)
    return {"status": "stored", "feedback_id": feedback_id}

@app.get("/feedback/summary")
async def feedback_summary(days: int = 30):
    """Get feedback summary for the last N days."""
    return get_feedback_summary(days=days)

@app.get("/feedback/routes")
async def feedback_routes():
    """Get route performance stats."""
    stats = get_route_stats(min_ratings=3)
    return {"routes": [{"intent": s.intent, "model": s.model, "template": s.template,
                         "total": s.total, "positive": s.positive, "negative": s.negative,
                         "success_rate": s.success_rate, "avg_quality": s.avg_quality}
                        for s in stats]}

@app.get("/feedback/overrides")
async def feedback_overrides():
    """Get routing override recommendations based on feedback."""
    return get_routing_overrides()

@app.get("/feedback/insights")
async def feedback_insights():
    """Get learning insights from feedback data."""
    return {"insights": get_insights()}

@app.get("/feedback/export")
async def feedback_export():
    """Export training data as JSONL."""
    path = export_training_jsonl()
    return FileResponse(path, media_type="application/jsonl", filename="training_data.jsonl")

# ── Phase 4: Knowledge Graph Endpoints ────────────────────────

@app.get("/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    kg = load_graph()
    return get_graph_stats(kg)

@app.get("/graph/related")
async def graph_related(q: str = Query(...), hops: int = 2, limit: int = 10):
    """Find entities related to a query."""
    kg = load_graph()
    results = find_related(kg, q, max_hops=hops, max_results=limit)
    return {"query": q, "related": results}

@app.get("/graph/document/{source}")
async def graph_document(source: str):
    """Get all entities connected to a document."""
    kg = load_graph()
    connections = get_document_connections(kg, source)
    return {"source": source, "connections": connections}

@app.post("/graph/rebuild")
async def graph_rebuild():
    """Rebuild the entire knowledge graph from ChromaDB."""
    col = get_collection()
    all_data = col.get(include=["documents", "metadatas"])
    docs = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        docs.append({
            "source": meta.get("source", "unknown"),
            "text": doc,
            "metadata": meta,
        })
    kg = kg_rebuild(docs)
    stats = get_graph_stats(kg)
    return {"status": "rebuilt", "stats": stats}

# ── Phase 4: Fine-Tuning Pipeline Endpoints ──────────────────

@app.get("/finetune/readiness")
async def finetune_readiness():
    """Check if the system is ready for fine-tuning."""
    return ft_check_readiness()

@app.get("/finetune/status")
async def finetune_status():
    """Get current fine-tuning pipeline status."""
    return ft_get_status()

@app.post("/finetune/train")
async def finetune_train(data: dict = {}):
    """Start a fine-tuning run (long-running, blocking)."""
    result = ft_run_training(
        epochs=data.get("epochs", 3),
        batch_size=data.get("batch_size", 2),
        learning_rate=data.get("learning_rate", 2e-4),
    )
    return result

@app.get("/finetune/history")
async def finetune_history():
    """Get training run history."""
    return {"history": ft_get_history()}

# ── Phase 4: Proactive Intelligence Endpoints ─────────────────

@app.get("/proactive/alerts")
async def proactive_alerts(include_dismissed: bool = False, limit: int = 50):
    """Get active alerts."""
    return {"alerts": get_alerts(include_dismissed=include_dismissed, limit=limit)}

@app.get("/proactive/unread")
async def proactive_unread():
    """Get unread alert count."""
    return {"unread": get_unread_count()}

@app.post("/proactive/alerts/{alert_id}/read")
async def proactive_mark_read(alert_id: str):
    """Mark an alert as read."""
    mark_read(alert_id)
    return {"status": "read"}

@app.post("/proactive/alerts/{alert_id}/dismiss")
async def proactive_dismiss(alert_id: str):
    """Dismiss an alert."""
    dismiss_alert(alert_id)
    return {"status": "dismissed"}

@app.post("/proactive/alerts/read-all")
async def proactive_read_all():
    """Mark all alerts as read."""
    mark_all_read()
    return {"status": "all_read"}

@app.get("/proactive/watches")
async def proactive_watches():
    """Get all active watches."""
    return get_watches()

@app.post("/proactive/watch/folder")
async def proactive_watch_folder(data: dict):
    """Add a folder to watch."""
    return add_folder_watch(data.get("path", ""), data.get("label", ""))

@app.delete("/proactive/watch/folder")
async def proactive_unwatch_folder(data: dict):
    """Remove a folder watch."""
    return remove_folder_watch(data.get("path", ""))

@app.post("/proactive/watch/topic")
async def proactive_watch_topic(data: dict):
    """Add a topic to watch."""
    return add_topic_watch(
        data.get("topic", ""),
        data.get("search_query", ""),
        data.get("interval_hours", 24),
    )

@app.post("/proactive/deadline")
async def proactive_add_deadline(data: dict):
    """Add a deadline to track."""
    return add_deadline(
        data.get("title", ""),
        data.get("date", ""),
        data.get("source", ""),
    )

@app.post("/proactive/scan")
async def proactive_scan():
    """Run a proactive intelligence scan now."""
    return run_proactive_scan()

@app.get("/proactive/summary")
async def proactive_summary():
    """Get proactive intelligence summary (for morning briefing)."""
    return get_proactive_summary()

# ── Phase 5: RBAC Endpoints ───────────────────────────────────

@app.post("/auth/register")
async def auth_register(data: dict):
    """Create a new user account."""
    try:
        user_id = create_user(
            username=data["username"],
            password=data["password"],
            email=data.get("email", ""),
            role=data.get("role", "viewer"),
        )
        return {"status": "created", "user_id": user_id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/auth/login")
async def auth_login(data: dict):
    """Authenticate and get a JWT token."""
    user = authenticate(data["username"], data["password"])
    if user:
        token = generate_token(user["id"], user["username"], user["role"])
        return {"token": token, "user": user}
    return {"error": "Invalid credentials"}

@app.get("/auth/users")
async def auth_users():
    """List all users (admin only)."""
    return {"users": list_users()}

@app.post("/auth/role")
async def auth_update_role(data: dict):
    """Update a user's role."""
    update_role(data["user_id"], data["role"])
    return {"status": "updated"}

@app.post("/auth/workspace")
async def auth_assign_workspace(data: dict):
    """Assign a workspace to a user."""
    assign_workspace(data["user_id"], data["workspace"])
    return {"status": "assigned"}

@app.get("/auth/audit")
async def auth_audit(limit: int = 100):
    """Get access audit log."""
    return {"entries": get_audit_log(limit=limit)}

# ── Phase 5: Document Comparison Endpoints ────────────────────

class CompareRequest(BaseModel):
    text_a: str
    text_b: str
    label_a: str = "Document A"
    label_b: str = "Document B"

@app.post("/compare")
async def compare_docs(req: CompareRequest):
    """Compare two documents."""
    result = compare_documents(req.text_a, req.text_b, req.label_a, req.label_b)
    return result.to_dict() if hasattr(result, "to_dict") else result

@app.post("/compare/markdown")
async def compare_markdown(req: CompareRequest):
    """Compare two documents and return markdown diff."""
    result = compare_documents(req.text_a, req.text_b, req.label_a, req.label_b)
    md = export_comparison_markdown(result)
    return {"markdown": md}

# ── Phase 5: Export Layer Endpoints ───────────────────────────

@app.post("/export/memo")
async def export_memo(data: dict):
    """Export a chat to a structured memo."""
    result = export_chat_to_memo(
        messages=data.get("messages", []),
        title=data.get("title", ""),
        author=data.get("author", "VaultMind"),
    )
    return result

@app.post("/export/report")
async def export_report(data: dict):
    """Export analysis to a structured report."""
    result = export_analysis_to_report(
        analysis=data.get("analysis", ""),
        title=data.get("title", ""),
        sources=data.get("sources", []),
    )
    return result

@app.post("/export/brief")
async def export_brief(data: dict):
    """Export research to a brief template."""
    result = export_research_to_brief(
        question=data.get("question", ""),
        answer=data.get("answer", ""),
        sources=data.get("sources", []),
    )
    return result

@app.get("/export/templates")
async def export_templates():
    """List available export templates."""
    return {"templates": list_templates()}

@app.post("/webhooks/register")
async def webhook_register(data: dict):
    """Register a webhook."""
    return register_webhook(
        url=data["url"],
        events=data.get("events", []),
        secret=data.get("secret", ""),
    )

@app.post("/api-keys/generate")
async def apikey_generate(data: dict):
    """Generate an API key."""
    return generate_api_key(
        label=data.get("label", "default"),
        permissions=data.get("permissions", ["read"]),
    )

# ── Phase 5: Vertical Kit Endpoints ──────────────────────────

@app.get("/verticals")
async def verticals_list():
    """List available vertical profiles."""
    return {"profiles": list_profiles()}

@app.get("/verticals/active")
async def verticals_active():
    """Get the active vertical profile."""
    profile = get_active_profile()
    return profile if profile else {"profile": "general"}

@app.post("/verticals/activate")
async def verticals_activate(data: dict):
    """Set the active vertical profile."""
    set_active_profile(data["profile"])
    return {"status": "activated", "profile": data["profile"]}

@app.post("/verticals/create")
async def verticals_create(data: dict):
    """Create a custom vertical profile."""
    result = create_custom_profile(
        name=data["name"],
        base=data.get("base", "general"),
        overrides=data.get("overrides", {}),
    )
    return result

# ── Phase 6: Photo Pipeline Endpoints ─────────────────────────

@app.post("/photos/process")
async def photos_process(data: dict):
    """Process a photo into searchable knowledge."""
    result = process_photo(
        image_data=data.get("image_data"),
        filepath=data.get("filepath"),
        filename=data.get("filename", "photo.jpg"),
        document_type=data.get("document_type", "auto"),
    )
    return result

@app.post("/photos/queue/process")
async def photos_queue_process():
    """Process all queued photos."""
    results = photo_process_queue()
    return {"processed": len(results), "results": results}

# ── Phase 6: Call Intelligence Endpoints ──────────────────────

@app.post("/calls/process")
async def calls_process(data: dict):
    """Process a call transcript."""
    result = ci_process_transcript(
        transcript_text=data.get("transcript", ""),
        participants=data.get("participants", []),
        workspace=data.get("workspace", ""),
    )
    return result

@app.get("/calls/history")
async def calls_history(limit: int = 20):
    """Get call history."""
    return {"calls": get_call_history(limit=limit)}

# ── Phase 6: Mobile Alerts Endpoints ──────────────────────────

@app.post("/mobile/register")
async def mobile_register(data: dict):
    """Register a mobile device for push alerts."""
    return ma_register_device(
        device_id=data["device_id"],
        device_name=data.get("device_name", ""),
        platform=data.get("platform", "unknown"),
    )

@app.get("/mobile/alerts/{device_id}")
async def mobile_alerts(device_id: str, limit: int = 50):
    """Get pending alerts for a device."""
    return {"alerts": ma_get_pending(device_id, limit=limit)}

@app.post("/mobile/alerts/{alert_id}/delivered")
async def mobile_delivered(alert_id: str):
    """Mark a mobile alert as delivered."""
    ma_mark_delivered(alert_id)
    return {"status": "delivered"}

@app.post("/mobile/preferences")
async def mobile_preferences(data: dict):
    """Configure alert preferences for a device."""
    return ma_configure_prefs(
        device_id=data["device_id"],
        min_priority=data.get("min_priority", "medium"),
        quiet_start=data.get("quiet_start"),
        quiet_end=data.get("quiet_end"),
    )

# ── Phase 6: Contact Intelligence Endpoints ───────────────────

@app.post("/contacts")
async def contacts_add(data: dict):
    """Add a contact."""
    return add_contact(
        name=data["name"],
        phone=data.get("phone", ""),
        email=data.get("email", ""),
        company=data.get("company", ""),
        role=data.get("role", ""),
        tags=data.get("tags", []),
        workspaces=data.get("workspaces", []),
    )

@app.get("/contacts/search")
async def contacts_search(q: str = Query(...)):
    """Search contacts."""
    results = search_contacts(q)
    return {"contacts": results}

@app.get("/contacts/{contact_id}/briefing")
async def contacts_briefing(contact_id: str):
    """Generate a pre-call briefing for a contact."""
    return generate_briefing(contact_id)

@app.post("/contacts/{contact_id}/interaction")
async def contacts_interaction(contact_id: str, data: dict):
    """Log an interaction with a contact."""
    return log_interaction(
        contact_id=contact_id,
        interaction_type=data.get("type", "note"),
        summary=data.get("summary", ""),
    )

@app.post("/contacts/import")
async def contacts_import(data: dict):
    """Import contacts from JSON."""
    return ci_import_contacts(data.get("contacts", []))

@app.get("/contacts/{contact_id}/history")
async def contacts_history(contact_id: str, limit: int = 20):
    """Get interaction history for a contact."""
    return {"history": get_contact_history(contact_id, limit=limit)}

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

def chunk_text(text: str, chunk_size: int = 200) -> list:
    """Section-aware chunking with rich metadata.
    Returns list of dicts (handled by embed_and_store).
    Falls back to basic word chunking for very short texts.
    """
    words = text.split()
    # For very short text, just return a simple chunk
    if len(words) < chunk_size:
        if text.strip():
            return [{"text": text.strip(), "section_header": "", "chunk_index": 0, "char_start": 0, "char_end": len(text)}]
        return []
    # Use the smart section-aware chunker
    return chunk_text_smart(text, chunk_size=chunk_size, overlap=40)


import re as _re_chunker

def chunk_text_smart(text: str, chunk_size: int = 200, overlap: int = 40) -> list[dict]:
    """Section-aware chunking that preserves document structure.

    Returns list of dicts with keys: text, section_header, chunk_index, char_start, char_end
    This gives ChromaDB richer metadata for better retrieval.
    """
    # Split on common section headers (markdown ##, ALL CAPS lines, numbered sections)
    section_pattern = _re_chunker.compile(
        r'(?:^|\n)'                          # start of line
        r'(?:'
        r'#{1,4}\s+.+'                       # markdown headers: ## Section
        r'|[A-Z][A-Z\s]{4,}(?:\n|$)'        # ALL CAPS HEADERS
        r'|\d+\.\s+[A-Z].+'                 # numbered sections: 1. Introduction
        r'|(?:ARTICLE|SECTION|CHAPTER)\s+\d+' # legal sections
        r')',
        _re_chunker.MULTILINE
    )

    # Find all section boundaries
    section_starts = [m.start() for m in section_pattern.finditer(text)]
    if not section_starts or section_starts[0] != 0:
        section_starts.insert(0, 0)

    # Build sections with their headers
    sections = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue

        # Extract header (first line of section)
        first_line = section_text.split('\n')[0].strip().lstrip('#').strip()
        if len(first_line) > 80:
            first_line = first_line[:80]

        sections.append({"header": first_line, "text": section_text, "char_start": start})

    # If no sections detected, treat whole doc as one section
    if not sections:
        sections = [{"header": "", "text": text, "char_start": 0}]

    # Now chunk each section with overlap
    chunks = []
    chunk_idx = 0
    for section in sections:
        words = section["text"].split()
        if not words:
            continue
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            if chunk_str.strip():
                chunks.append({
                    "text": chunk_str,
                    "section_header": section["header"],
                    "chunk_index": chunk_idx,
                    "char_start": section["char_start"],
                    "char_end": section["char_start"] + len(section["text"]),
                })
                chunk_idx += 1

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

def embed_and_store(chunks, source: str, col):
    """Embed and store chunks in ChromaDB. Accepts either:
    - list[str] (legacy basic chunks)
    - list[dict] (smart chunks with metadata from chunk_text_smart)
    """
    for i, chunk in enumerate(chunks):
        # Support both old list[str] and new list[dict] format
        if isinstance(chunk, dict):
            text = chunk["text"]
            meta = {
                "source": source,
                "chunk": chunk.get("chunk_index", i),
                "section": chunk.get("section_header", ""),
                "char_start": chunk.get("char_start", 0),
                "char_end": chunk.get("char_end", 0),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            text = chunk
            meta = {"source": source, "chunk": i}

        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
        col.upsert(
            ids=[f"{source}_{i}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta]
        )
        if i % 10 == 0:
            print(f"  ✓ {i}/{len(chunks)}")

    # Phase 4: Add document to knowledge graph (non-blocking)
    try:
        kg = load_graph()
        if kg is not None:
            full_text = " ".join(
                c["text"] if isinstance(c, dict) else c for c in chunks
            )
            add_document_to_graph(kg, source, full_text)
            save_graph(kg)
    except Exception as e:
        print(f"[KnowledgeGraph] Failed to update graph (non-fatal): {e}")

# ── Photo Intelligence ────────────────────────────────────────

def extract_exif(contents: bytes, filename: str) -> str:
    """Return a text summary of EXIF metadata from an image."""
    if not PILLOW_OK:
        return f"Filename: {filename}"
    try:
        img = Image.open(io.BytesIO(contents))
        parts = [f"Filename: {filename}", f"Size: {img.size[0]}x{img.size[1]}px", f"Format: {img.format or 'unknown'}"]
        exif_data = img._getexif() if hasattr(img, '_getexif') else None
        if exif_data:
            WANTED = {'DateTime': 'Date', 'DateTimeOriginal': 'Date taken',
                      'Make': 'Camera', 'Model': 'Camera model',
                      'ImageDescription': 'Caption', 'Artist': 'Artist'}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, '')
                if tag in WANTED and value:
                    parts.append(f"{WANTED[tag]}: {str(value)[:80]}")
        return "\n".join(parts)
    except Exception as e:
        print(f"EXIF error ({filename}): {e}")
        return f"Filename: {filename}"

def caption_image_llava(contents: bytes, filename: str) -> str:
    """Use LLaVA via Ollama to generate a rich image description."""
    try:
        image_b64 = base64.b64encode(contents).decode()
        response = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": (
                    "Describe this image in detail. Include: people present (no identifying info), "
                    "location and setting, objects visible, activities, time of day if apparent, "
                    "any text visible, and notable features. "
                    "Be specific — this description will be used to search for this photo later."
                ),
                "images": [image_b64],
            }],
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"LLaVA error ({filename}): {e}")
        return "Photo description unavailable — run 'ollama pull llava' to enable AI photo captions."

# ── Apple Health XML ───────────────────────────────────────────

def parse_apple_health_xml(contents: bytes) -> str:
    """Convert Apple Health export.xml into searchable text blocks."""
    try:
        root = ET.fromstring(contents)
    except ET.ParseError as e:
        return f"Could not parse Apple Health XML: {e}"

    records_by_type: dict[str, list[str]] = {}
    for record in root.findall('.//Record'):
        rtype = (record.get('type', '')
                 .replace('HKQuantityTypeIdentifier', '')
                 .replace('HKCategoryTypeIdentifier', '')
                 .replace('HKDataType', ''))
        value = record.get('value', '')
        unit  = record.get('unit', '')
        date  = record.get('startDate', '')[:10]
        if not value or not date:
            continue
        entry = f"{date}: {value} {unit}".strip()
        records_by_type.setdefault(rtype, []).append(entry)

    workout_lines = []
    for w in root.findall('.//Workout'):
        wtype    = w.get('workoutActivityType', '').replace('HKWorkoutActivityType', '')
        duration = w.get('duration', '')
        unit     = w.get('durationUnit', 'min')
        date     = w.get('startDate', '')[:10]
        workout_lines.append(f"{date}: {wtype} {duration} {unit}")

    PRIORITY = ['HeartRate','StepCount','BodyMass','BloodPressureSystolic',
                'BloodPressureDiastolic','BloodGlucose','OxygenSaturation',
                'RespiratoryRate','ActiveEnergyBurned','DistanceWalkingRunning',
                'SleepAnalysis','BodyFatPercentage','MindfulSession']

    lines = ["Apple Health Export\n",
             f"Record types: {len(records_by_type)}",
             f"Workouts: {len(workout_lines)}\n"]

    for rtype in PRIORITY:
        if rtype in records_by_type:
            lines.append(f"\n{rtype}:")
            lines.extend(records_by_type[rtype][-100:])

    for rtype, entries in records_by_type.items():
        if rtype not in PRIORITY:
            lines.append(f"\n{rtype}:")
            lines.extend(entries[-20:])

    if workout_lines:
        lines.append("\nWorkouts:")
        lines.extend(workout_lines[-100:])

    return "\n".join(lines)

# ── Obsidian Vault (ZIP) ───────────────────────────────────────

def parse_obsidian_zip(contents: bytes) -> list[tuple[str, str, str]]:
    """Extract (source, display_name, text) tuples from a ZIP of .md files."""
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            md_files = [n for n in zf.namelist()
                        if n.endswith('.md') and '/.trash/' not in n and not n.startswith('__')]
            for name in md_files:
                try:
                    text = zf.read(name).decode('utf-8', errors='ignore').strip()
                    if len(text) < 20:
                        continue
                    display = os.path.basename(name).replace('.md', '')
                    source  = f"obsidian:{name}"
                    results.append((source, display, text))
                except Exception as e:
                    print(f"  ⚠️ Skipped {name}: {e}")
    except zipfile.BadZipFile:
        pass
    return results

# ── Folder Watcher ────────────────────────────────────────────

def _index_file_sync(path: str):
    """Index a single file into the vault using the model router (called from watchdog thread)."""
    try:
        with open(path, 'rb') as f:
            contents = f.read()
        filename = os.path.basename(path)
        ext      = os.path.splitext(filename)[1].lower()
        col      = get_collection()
        route    = route_file(filename, contents)

        if route == RouteType.VLM:
            # Vision pipeline
            vlm_model = get_available_vlm()
            if not vlm_model:
                print(f"⚠️  No VLM available for {filename} — skipping vision processing")
                return
            source = f"vlm:{filename}"
            if col.get(where={"source": source}, include=["metadatas"])["ids"]:
                return
            print(f"🔭 Watch folder VLM: {filename} → {vlm_model}")
            if ext == ".pdf":
                text = extract_pdf_with_vlm(contents, filename)
            else:
                text = extract_image_with_vlm(contents, filename)
            if text.strip():
                chunks = chunk_text(text)
                embed_and_store(chunks, source, col)
                log_feed_event(filename, "vault", len(chunks), "watch-vlm", source)
                print(f"✅ Auto-indexed via VLM: {filename}")
        elif ext in IMAGE_EXTENSIONS:
            source = f"photo:{filename}"
            if col.get(where={"source": source}, include=["metadatas"])["ids"]:
                return
            exif_text = extract_exif(contents, filename)
            caption   = caption_image_llava(contents, filename)
            text      = f"Photo: {filename}\n{exif_text}\n\nDescription: {caption}"
            chunks    = chunk_text(text)
            embed_and_store(chunks, source, col)
            log_feed_event(filename, "vault", len(chunks), "watch-photo", source)
            print(f"✅ Auto-indexed photo: {filename}")
        else:
            source = filename
            if col.get(where={"source": source}, include=["metadatas"])["ids"]:
                return
            text = extract_text_from_file(contents, filename)
            if text.strip():
                chunks = chunk_text(text)
                embed_and_store(chunks, source, col)
                log_feed_event(filename, "vault", len(chunks), "watch", source)
                print(f"✅ Auto-indexed: {filename}")
    except Exception as e:
        print(f"⚠️ Watch folder index error: {e}")

if WATCHDOG_OK:
    class VaultFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext in WATCHABLE_EXTS:
                threading.Timer(1.0, _index_file_sync, args=[event.src_path]).start()

def start_folder_watcher(folder_path: str):
    global _watcher_observer, _active_watchers
    if not WATCHDOG_OK or folder_path in _active_watchers:
        return
    if _watcher_observer is None:
        _watcher_observer = Observer()
        _watcher_observer.start()
    handler = VaultFileHandler()
    _watcher_observer.schedule(handler, folder_path, recursive=True)
    _active_watchers[folder_path] = handler
    print(f"👁️  Watching: {folder_path}")

def stop_folder_watcher(folder_path: str):
    _active_watchers.pop(folder_path, None)

# ── Upload ────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(
    file:      UploadFile = File(...),
    workspace: str        = Form(default="Default"),
    doc_type:  str        = Form(default="general"),   # "legal" triggers legal VLM prompt
):
    contents = await file.read()
    ext      = os.path.splitext(file.filename)[1].lower()

    # ── Route to correct pipeline ──────────────────────────────
    route = route_file(file.filename, contents)

    if route == RouteType.VLM:
        # Vision pipeline — scanned PDF or image
        vlm_model = get_available_vlm()
        if not vlm_model:
            return {
                "error": (
                    "No VLM model available for this file type. "
                    "Run: ollama pull qwen2.5vl:7b\n"
                    "Then restart VaultMind."
                )
            }

        print(f"\n🔭 VLM pipeline: '{file.filename}' → {vlm_model}")

        if ext == ".pdf":
            text = await asyncio.to_thread(
                extract_pdf_with_vlm, contents, file.filename, doc_type
            )
        else:
            text = await asyncio.to_thread(
                extract_image_with_vlm, contents, file.filename, doc_type
            )

        if not text.strip() or "[VLM processing failed" in text:
            return {"error": f"VLM could not extract content from '{file.filename}'."}

        chunks = chunk_text(text)
        source = f"vlm:{file.filename}"
        print(f"📄 Indexing VLM output '{file.filename}' — {len(chunks)} chunks")
        col = get_collection()
        embed_and_store(chunks, source, col)
        log_feed_event(file.filename, "vault", len(chunks), "vlm-upload")
        return {
            "message": f"Indexed {file.filename} via VLM",
            "chunks": len(chunks),
            "pipeline": "vlm",
            "model": vlm_model,
        }

    else:
        # Text pipeline — standard extraction
        text = extract_text_from_file(contents, file.filename)
        if not text.strip():
            return {"error": f"Could not extract text from '{file.filename}'. Supported: PDF, DOCX, TXT, MD, CSV"}
        chunks = chunk_text(text)
        print(f"\n📄 Indexing '{file.filename}' — {len(chunks)} chunks")
        col = get_collection()
        embed_and_store(chunks, file.filename, col)
        log_feed_event(file.filename, "vault", len(chunks), "upload")
        return {
            "message": f"Indexed {file.filename}",
            "chunks": len(chunks),
            "pipeline": "slm",
        }

# ── Photo Upload ──────────────────────────────────────────────

@app.post("/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    """Accept an image, extract EXIF metadata, caption with LLaVA, store as searchable text."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return {"error": f"Unsupported image type: {ext}. Supported: JPG, PNG, GIF, WEBP, HEIC"}

    contents = await file.read()
    source   = f"photo:{file.filename}"
    col      = get_collection()

    existing = col.get(where={"source": source}, include=["metadatas"])
    if existing["ids"]:
        return {"message": f"Already indexed: {file.filename}", "chunks": len(existing["ids"])}

    print(f"\n📸 Processing photo '{file.filename}'…")
    exif_text = extract_exif(contents, file.filename)
    caption   = await asyncio.to_thread(caption_image_llava, contents, file.filename)
    full_text = f"Photo: {file.filename}\n{exif_text}\n\nDescription: {caption}"

    chunks = chunk_text(full_text)
    embed_and_store(chunks, source, col)
    log_feed_event(file.filename, "vault", len(chunks), "photo", source)

    print(f"  ✅ Indexed photo '{file.filename}' — {len(chunks)} chunks")
    return {"message": f"Indexed {file.filename}", "chunks": len(chunks), "caption": caption[:200]}

# ── Apple Health Upload ────────────────────────────────────────

@app.post("/upload-health")
async def upload_health(file: UploadFile = File(...)):
    """Accept Apple Health export.xml and index it as searchable health records."""
    if not file.filename.lower().endswith('.xml'):
        return {"error": "Expected export.xml from Apple Health. Go to Health app → your avatar → Export All Health Data."}

    contents = await file.read()
    source   = "health:apple-health"
    col      = get_collection()

    # Replace any existing health data
    existing = col.get(where={"source": source}, include=["metadatas"])
    if existing["ids"]:
        col.delete(ids=existing["ids"])
        print(f"🗑️  Replaced existing Apple Health data ({len(existing['ids'])} chunks)")

    print(f"\n🏥 Parsing Apple Health export…")
    text = await asyncio.to_thread(parse_apple_health_xml, contents)
    if not text.strip():
        return {"error": "Could not parse Apple Health XML. Export from Health app → your avatar → Export All Health Data."}

    chunks = chunk_text(text)
    embed_and_store(chunks, source, col)
    log_feed_event("Apple Health Export", "vault", len(chunks), "health", source)
    print(f"  ✅ Indexed health data — {len(chunks)} chunks")
    return {"message": f"Indexed Apple Health data", "chunks": len(chunks)}

# ── Obsidian Vault Upload ──────────────────────────────────────

@app.post("/upload-obsidian")
async def upload_obsidian(file: UploadFile = File(...)):
    """Accept a ZIP of an Obsidian vault and index all .md notes."""
    if not file.filename.lower().endswith('.zip'):
        return {"error": "Please ZIP your Obsidian vault folder and upload the ZIP file."}

    contents     = await file.read()
    col          = get_collection()
    total_chunks = 0
    indexed      = 0

    notes = await asyncio.to_thread(parse_obsidian_zip, contents)
    if not notes:
        return {"error": "No .md files found in ZIP. Make sure you zipped an Obsidian vault folder."}

    for source, display, text in notes:
        existing = col.get(where={"source": source}, include=["metadatas"])
        if existing["ids"]:
            continue
        chunks = chunk_text(text)
        embed_and_store(chunks, source, col)
        log_feed_event(display, "vault", len(chunks), "obsidian", source)
        total_chunks += len(chunks)
        indexed      += 1

    print(f"  ✅ Indexed {indexed} Obsidian notes — {total_chunks} chunks")
    return {"message": f"Indexed {indexed} notes", "chunks": total_chunks, "files": indexed}

# ── Watch Folders ─────────────────────────────────────────────

class WatchFolderRequest(BaseModel):
    path: str

@app.get("/watch-folders")
async def list_watch_folders():
    cfg = load_config()
    return {"folders": cfg.get(WATCH_FOLDERS_KEY, []), "watchdog_available": WATCHDOG_OK}

@app.post("/watch-folders")
async def add_watch_folder(req: WatchFolderRequest):
    if not WATCHDOG_OK:
        return {"error": "watchdog not installed — run: pip install watchdog"}
    if not os.path.isdir(req.path):
        return {"error": f"Directory not found: {req.path}"}
    cfg     = load_config()
    folders = cfg.get(WATCH_FOLDERS_KEY, [])
    if req.path not in folders:
        folders.append(req.path)
        cfg[WATCH_FOLDERS_KEY] = folders
        save_config(cfg)
        start_folder_watcher(req.path)
    return {"message": f"Now watching: {req.path}", "folders": folders}

@app.delete("/watch-folders")
async def remove_watch_folder(path: str = Query(...)):
    cfg     = load_config()
    folders = cfg.get(WATCH_FOLDERS_KEY, [])
    if path in folders:
        folders.remove(path)
        cfg[WATCH_FOLDERS_KEY] = folders
        save_config(cfg)
    stop_folder_watcher(path)
    return {"message": f"Stopped watching: {path}", "folders": folders}

# ── Privacy Dashboard ─────────────────────────────────────────

@app.get("/privacy")
async def privacy_dashboard():
    """Return a breakdown of indexed data and confirm zero external data transmission."""
    col     = get_collection()
    results = col.get(include=["metadatas"])

    source_types: dict[str, int] = {}
    sources_seen: set             = set()

    for meta in results["metadatas"]:
        src = meta["source"]
        if src in sources_seen:
            continue
        sources_seen.add(src)
        if src.startswith("photo:"):        stype = "Photos"
        elif src.startswith("health:"):     stype = "Health Records"
        elif src.startswith("notion:"):     stype = "Notion Pages"
        elif src.startswith("obsidian:"):   stype = "Obsidian Notes"
        elif src.startswith("gmail:"):      stype = "Emails"
        elif src.startswith("🌐") or src.startswith("http"): stype = "Web Pages"
        else:                               stype = "Documents"
        source_types[stype] = source_types.get(stype, 0) + 1

    cfg = load_config()
    return {
        "total_chunks":         col.count(),
        "total_sources":        len(sources_seen),
        "source_breakdown":     source_types,
        "external_connections": [],
        "network_calls":        "Only during Agent mode web search or Notion sync — never for your personal data",
        "data_location":        os.path.abspath(DATA_DIR),
        "watch_folders":        cfg.get(WATCH_FOLDERS_KEY, []),
        "checked_at":           datetime.now(timezone.utc).isoformat(),
    }

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
    col    = get_collection()
    embed_and_store(chunks, source, col)
    log_feed_event(title, "vault", len(chunks), "url")
    return {"message": f"Indexed {title}", "chunks": len(chunks), "source": source}

# ── Files ─────────────────────────────────────────────────────

@app.get("/files")
async def list_files(workspace: str = Query(default="Default")):  # workspace kept for compat
    col          = get_collection()
    results      = col.get(include=["metadatas", "documents"])
    files        = {}    # source → chunk count
    first_chunks = {}    # source → first chunk text

    for meta, doc in zip(results["metadatas"], results["documents"]):
        src       = meta["source"]
        chunk_idx = meta.get("chunk", 0)
        files[src] = files.get(src, 0) + 1
        if chunk_idx == 0:
            first_chunks[src] = doc

    items = []
    for src, count in files.items():
        title = None
        first = first_chunks.get(src, "")
        if src.startswith("gmail:") and first:
            for line in first.split("\n"):
                if line.lower().startswith("subject:"):
                    title = line[8:].strip() or None
                    break
        elif src.startswith("notion:") and first:
            for line in first.split("\n"):
                if line.strip():
                    title = line.strip()[:80]
                    break
        elif src.startswith("photo:"):
            title = src[6:]  # strip "photo:" prefix
        elif src.startswith("health:"):
            title = "Apple Health Export"
        elif src.startswith("obsidian:"):
            # obsidian:path/to/Note.md → "Note"
            title = os.path.basename(src[9:]).replace('.md', '')
        items.append({"name": src, "title": title, "chunks": count})

    return {"files": items}

@app.delete("/files/{filename}")
async def delete_file(filename: str, workspace: str = Query(default="Default")):
    col     = get_collection()
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
    get_collection()  # ensure the vault collection exists
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
    cfg["gmail"]["poll_interval_minutes"] = data.poll_interval_minutes
    save_config(cfg)
    get_collection()  # ensure vault collection exists
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

SKILL_PROMPTS = {
    "nostalgic": (
        "RESPONSE STYLE — Nostalgic Skin Co. Brand Voice:\n"
        "You write copy for a men's natural bar soap brand. Voice: rugged, witty, anti-corporate, nostalgic, honest.\n"
        "Target: blue-collar workers and tech dads, ages 25-45.\n"
        "Sound like a buddy recommending something — NOT a brand selling something.\n"
        "Use concrete physical language: hands, grit, sweat, work, garage, campfire.\n"
        "Lead with humor or an unexpected hook, not product features.\n"
        "NEVER use: premium, artisanal, curated, elevated, bespoke, pampering, luxury.\n"
        "CTAs: 'Grab a bar' — NOT 'Shop Now' or 'Add to Cart'.\n"
        "Describe scents through memory/scenario, not ingredient lists.\n"
        "No markdown. Plain prose only."
    ),
    "recruiting": (
        "RESPONSE STYLE — Recruiting Expert:\n"
        "You are a specialized headhunter focused on Data Science, Data Engineering, Analytics, and Machine Learning roles.\n"
        "Help craft outreach emails, evaluate candidates, analyze job descriptions, and provide recruiting strategy.\n"
        "Be direct, actionable, and focused on revenue outcomes. No fluff."
    ),
    "airblackbox": (
        "RESPONSE STYLE — AIR Blackbox Developer:\n"
        "You are an expert developer on AIR Blackbox, an open-source EU AI Act compliance scanner for Python AI agents.\n"
        "You know: Article 9 (risk management), 10 (data governance), 11 (technical documentation), "
        "12 (logging/audit), 14 (human oversight), 15 (accuracy/robustness).\n"
        "Help with code, documentation, compliance analysis, and developer content. Be precise and technical."
    ),
}

class ChatMessage(BaseModel):
    message:       str
    history:       list[dict] = []
    workspace:     str = "Default"
    model:         str = "mistral"
    pinned_source: str = ""   # when set, restrict retrieval to this exact source
    skill:         str = ""   # optional skill context injected into system prompt
    custom_prompt: str = ""   # free-form system prompt from the prompts marketplace
    mode:          str = ""   # "agent" when called from agent toggle

EMAIL_KEYWORDS = {"email", "emails", "inbox", "gmail", "mail", "summarize my day",
                  "what did i get", "any messages", "any emails", "check my email"}


# ── Conversation persistence ──────────────────────────────────

class ConversationSave(BaseModel):
    id:       str
    title:    str = ""
    messages: list[dict] = []
    model:    str = "mistral"
    skill:    str = ""

@app.get("/conversations")
async def list_conversations():
    """Return all saved conversations sorted by last modified (newest first)."""
    convos = []
    for fname in os.listdir(CONVERSATIONS_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(CONVERSATIONS_DIR, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            convos.append({
                "id":        data.get("id", fname.replace(".json", "")),
                "title":     data.get("title", "Untitled"),
                "model":     data.get("model", "mistral"),
                "skill":     data.get("skill", ""),
                "count":     len(data.get("messages", [])),
                "updated_at": os.path.getmtime(fpath),
            })
        except Exception:
            continue
    convos.sort(key=lambda c: c["updated_at"], reverse=True)
    return {"conversations": convos}

@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Load a single conversation by ID."""
    fpath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if not os.path.exists(fpath):
        return {"error": "Conversation not found"}
    with open(fpath, "r") as f:
        return json.load(f)

@app.post("/conversations")
async def save_conversation(conv: ConversationSave):
    """Save or update a conversation and store a memory summary."""
    fpath = os.path.join(CONVERSATIONS_DIR, f"{conv.id}.json")
    data = {
        "id":       conv.id,
        "title":    conv.title,
        "messages": conv.messages,
        "model":    conv.model,
        "skill":    conv.skill,
    }
    with open(fpath, "w") as f:
        json.dump(data, f, indent=2)

    # Store conversation memory in background (don't block the save)
    if conv.messages and len(conv.messages) >= 4:
        try:
            import threading
            def _store_memory():
                try:
                    client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma"))
                    store_conversation_memory(
                        chroma_client=client,
                        conversation_id=conv.id,
                        summary="",  # auto-generate
                        messages=conv.messages,
                        model=conv.model or "mistral",
                    )
                except Exception as e:
                    print(f"[ConversationMemory] Background store failed: {e}")
            threading.Thread(target=_store_memory, daemon=True).start()
        except Exception as e:
            print(f"[ConversationMemory] Failed to start memory thread: {e}")

    return {"ok": True}

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    fpath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
    return {"ok": True}


import re as _re

# ── URL detection in user messages ──────────────────────────────
_URL_RE = _re.compile(r'https?://[^\s<>"\']+', _re.IGNORECASE)

def _extract_urls(text: str) -> list[str]:
    """Pull any URLs the user pasted into their message."""
    return _URL_RE.findall(text)

# Keywords that signal the user wants real web results, not doc search
# These must be EXPLICIT web intent -- not words that could appear in vault queries
_WEB_INTENT_PATTERNS = [
    r"\bfind\b.*\b(companies|jobs|hiring|openings|positions|listings)\b",
    r"\bsearch\b.*\b(the web|online|google)\b",
    r"\b(trending|breaking news)\b",
    r"\bpull\b.*\b(links|urls|recs|listings|results)\b",
    r"\b(who is hiring|companies hiring|jobs in|openings in|hiring in)\b",
    r"\b(give me|show me|list)\b.*\b(urls|links|websites)\b",
    r"\breal urls?\b",
    r"\b(salary|compensation|pay range|glassdoor)\b.*\b(for|at|in)\b",
    r"\bweb search\b",
    r"\bsearch the internet\b",
    r"\blook up online\b",
    r"\bgoogle\b",
    r"https?://",  # If user pastes a URL, always do web mode
]
_WEB_INTENT_RE = _re.compile("|".join(_WEB_INTENT_PATTERNS), _re.IGNORECASE)

def _looks_like_web_search(query: str) -> bool:
    """Heuristic: does this query want live web data?"""
    return bool(_WEB_INTENT_RE.search(query))

# ── Structured job listing extraction ───────────────────────────
def _extract_job_listings(html_content: bytes, base_url: str) -> list[dict]:
    """Parse a job listing page and extract structured entries.
    Returns list of {title, company, location, url}."""
    soup = BeautifulSoup(html_content, "html.parser")
    listings = []
    seen = set()
    base_domain = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc

    # Strategy 1: Look for links that point to individual job pages
    # Common patterns: /job/, /jobs/, /position/, /opening/, /career/
    job_link_re = _re.compile(r'/(job|jobs|position|opening|career|role|apply|req|posting)s?/', _re.IGNORECASE)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href == "#":
            continue

        # Make absolute URL
        if href.startswith("/"):
            href = base_domain + href
        elif not href.startswith("http"):
            continue

        # Skip if blocked or already seen
        if _is_blocked_url(href) or href in seen:
            continue

        # Check if this looks like a job link
        is_job_link = bool(job_link_re.search(href))
        link_text = a_tag.get_text(strip=True)

        if not link_text or len(link_text) < 5 or len(link_text) > 200:
            continue

        # Look for company name in nearby elements
        company = ""
        parent = a_tag.parent
        if parent:
            # Check siblings and parent for company-like text
            for sibling in parent.find_all(string=True):
                txt = sibling.strip()
                if txt and txt != link_text and 5 < len(txt) < 80:
                    company = txt
                    break

        # Look for location
        location = ""
        loc_el = parent.find(string=_re.compile(r'(CA|California|Irvine|Remote|San|Los|New York)', _re.IGNORECASE)) if parent else None
        if loc_el:
            location = loc_el.strip()

        if is_job_link or (link_text and any(kw in link_text.lower() for kw in ["engineer", "developer", "analyst", "manager", "scientist", "designer", "lead", "senior", "junior", "intern"])):
            seen.add(href)
            listings.append({
                "title": link_text,
                "company": company,
                "location": location,
                "url": href,
            })

    # Strategy 2: Look for common job card patterns (divs with class containing 'job', 'posting', 'listing')
    if len(listings) < 3:
        card_re = _re.compile(r'(job|posting|listing|position|opening|result|card)', _re.IGNORECASE)
        for div in soup.find_all(["div", "li", "article"], class_=card_re):
            link = div.find("a", href=True)
            if not link:
                continue
            href = link["href"].strip()
            if href.startswith("/"):
                href = base_domain + href
            elif not href.startswith("http"):
                continue
            if href in seen or _is_blocked_url(href):
                continue
            title = link.get_text(strip=True)
            if not title or len(title) < 5:
                continue

            # Try to find company name in the card
            company = ""
            company_el = div.find(class_=_re.compile(r'(company|employer|org)', _re.IGNORECASE))
            if company_el:
                company = company_el.get_text(strip=True)

            seen.add(href)
            listings.append({
                "title": title,
                "company": company,
                "location": "",
                "url": href,
            })

    return listings[:30]  # Cap at 30

def _extract_companies_from_text(text: str) -> list[str]:
    """Extract company names from scraped text using heuristics.
    Looks for patterns like 'at CompanyName', 'Company: X', capitalized names near job keywords."""
    companies = set()

    # Pattern: "at <Company>"
    for m in _re.finditer(r'\bat\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:\s*[-–—|•]\s*|\s+in\s+|\s*\n)', text):
        name = m.group(1).strip().rstrip(".,")
        if 2 < len(name) < 60 and not any(w in name.lower() for w in ["the", "this", "that", "your", "our"]):
            companies.add(name)

    # Pattern: "Company: <Name>" or "Employer: <Name>"
    for m in _re.finditer(r'(?:company|employer|client|organization|posted by)[:\s]+([A-Z][A-Za-z0-9\s&.,]+?)(?:\s*[-–—|•]\s*|\s*\n)', text, _re.IGNORECASE):
        name = m.group(1).strip().rstrip(".,")
        if 2 < len(name) < 60:
            companies.add(name)

    return list(companies)[:20]

# ── Staffing Agency Intelligence ────────────────────────────────
STAFFING_AGENCY_DOMAINS = {
    "motionrecruitment.com", "cybercoders.com", "jobot.com",
    "roberthalf.com", "randstad.com", "hays.com", "adecco.com",
    "insightglobal.com", "teksystems.com", "kforce.com",
    "aerotek.com", "actalentservices.com", "dice.com",
    "appleone.com", "spherion.com", "massgenie.com",
    "rht.com", "nesco.com", "collabera.com",
}

def _is_staffing_agency_url(url: str) -> bool:
    """Detect if a URL belongs to a known staffing/recruiting agency."""
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        return any(agency in domain for agency in STAFFING_AGENCY_DOMAINS)
    except Exception:
        return False

def _scrape_agency_listing_page(url: str) -> list[dict]:
    """Scrape a staffing agency job board page and extract individual job URLs.
    Returns list of {url, title, location, job_type, pay}."""
    try:
        r = requests.get(url, timeout=12, headers=BROWSER_HEADERS, stream=True)
        if r.status_code != 200:
            return []
        raw = b""
        for chunk in r.iter_content(chunk_size=8192):
            raw += chunk
            if len(raw) > 800_000:
                break
        soup = BeautifulSoup(raw, "html.parser")
        base = urlparse(url).scheme + "://" + urlparse(url).netloc
        listings = []
        seen = set()

        # Find all internal links that look like individual job pages
        # Pattern: path with job ID (number) at the end
        job_url_re = _re.compile(r'/\d{4,}$')  # Ends with 4+ digit ID
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href == "#":
                continue
            # Make absolute
            if href.startswith("/"):
                href = base + href
            elif not href.startswith("http"):
                continue
            # Must be same domain and have a job ID pattern
            if urlparse(href).netloc != urlparse(url).netloc:
                continue
            if not job_url_re.search(href) and "/job/" not in href:
                continue
            if href in seen:
                continue
            seen.add(href)
            title = a_tag.get_text(strip=True)[:200]
            if not title or len(title) < 5:
                continue
            listings.append({"url": href, "title": title, "description": ""})

        return listings[:25]  # Cap at 25
    except Exception as e:
        print(f"Agency listing scrape error: {e}")
        return []

def _deep_scrape_job_pages(job_urls: list[dict], max_pages: int = 15) -> list[dict]:
    """Scrape individual job detail pages and extract full descriptions.
    Uses JSON-LD structured data first (most job sites embed it),
    then falls back to HTML text extraction.
    Returns enriched list with description field populated."""
    results = []
    for entry in job_urls[:max_pages]:
        url = entry["url"]
        try:
            r = requests.get(url, timeout=10, headers=BROWSER_HEADERS)
            if r.status_code != 200:
                results.append(entry)
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            description = ""
            job_title = entry.get("title", "")

            # ── Strategy 1: JSON-LD (application/ld+json) ──────────
            # Most job sites (Indeed, Motion, Greenhouse, Lever, etc.)
            # embed full job data in structured JSON-LD.  This is FAR
            # more reliable than scraping the rendered HTML.
            for script_tag in soup.find_all("script", type="application/ld+json"):
                try:
                    ld_data = json.loads(script_tag.string or "{}")
                    # Handle both single object and @graph arrays
                    items = [ld_data] if isinstance(ld_data, dict) else ld_data if isinstance(ld_data, list) else []
                    if isinstance(ld_data, dict) and "@graph" in ld_data:
                        items = ld_data["@graph"]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        if item.get("@type") in ("JobPosting", "JobPosition", "Job"):
                            raw_desc = item.get("description", "")
                            # Strip HTML tags from description
                            raw_desc = _re.sub(r'<[^>]+>', ' ', raw_desc)
                            raw_desc = _re.sub(r'\s+', ' ', raw_desc).strip()
                            if len(raw_desc) > 100:
                                description = raw_desc[:5000]
                                job_title = item.get("title", job_title)
                                # Also grab salary if available
                                salary = item.get("baseSalary", {})
                                if isinstance(salary, dict) and salary.get("value"):
                                    val = salary["value"]
                                    if isinstance(val, dict):
                                        lo = val.get("minValue", "")
                                        hi = val.get("maxValue", "")
                                        unit = val.get("unitText", "YEAR")
                                        entry["salary"] = f"${lo}-${hi}/{unit}" if lo and hi else ""
                                # Grab location
                                loc = item.get("jobLocation", {})
                                if isinstance(loc, dict):
                                    addr = loc.get("address", {})
                                    if isinstance(addr, dict):
                                        city = addr.get("addressLocality", "")
                                        state = addr.get("addressRegion", "")
                                        if city:
                                            entry["location"] = f"{city}, {state}" if state else city
                                break
                    if description:
                        break
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

            # ── Strategy 2: HTML text fallback ─────────────────────
            if not description:
                # Remove non-content tags, then get text
                for tag in soup(["style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                    tag.decompose()
                # Try to find main content area first
                main = soup.find("main") or soup.find("article") or soup.find("div", class_=_re.compile(r'job|description|content|detail', _re.I))
                if main:
                    description = main.get_text(separator="\n", strip=True)[:5000]
                else:
                    description = soup.get_text(separator="\n", strip=True)[:3000]

            entry["title"] = job_title
            entry["description"] = description
            results.append(entry)
        except Exception:
            results.append(entry)
    return results

RELEVANCE_THRESHOLD = 0.65   # ChromaDB L2 distance; tuned for personal docs

@app.post("/chat")
async def chat(msg: ChatMessage):
    # ── Step 0: Query Intelligence -- classify before anything else ──
    try:
        available = qi_get_available_models()
    except Exception:
        available = []
    classification = classify_query(
        query=msg.message,
        conversation_history=msg.history or None,
        available_models=available or None,
    )
    print(f"[QueryIntel] {classification.reasoning}")
    print(f"[QueryIntel] Intent={classification.intent.value} Model={classification.recommended_model} Template={classification.prompt_template}")

    # Use the query intelligence model recommendation if the user didn't pick one
    # Phase 4: Check if feedback data suggests a better route
    feedback_override = None
    try:
        feedback_override = get_best_route(classification.intent.value, min_ratings=5)
        if feedback_override:
            print(f"[FeedbackLoop] Override available: {feedback_override['model']} ({feedback_override['success_rate']:.0%} success rate)")
    except Exception:
        pass

    if msg.model and msg.model != "mistral":
        chat_model = msg.model  # user explicitly chose a model, respect it
    elif feedback_override and feedback_override.get("success_rate", 0) > 0.7:
        chat_model = feedback_override["model"]  # feedback-learned best route
        print(f"[FeedbackLoop] Using learned route: {chat_model}")
    elif classification.recommended_model:
        chat_model = classification.recommended_model
    else:
        chat_model = msg.model or DEFAULT_MODEL

    question_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]
    col                = get_collection()

    # ── Step 0.5: Recall relevant past conversations ────────────
    memory_context = ""
    try:
        chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma"))
        memories = recall_relevant_memories(chroma_client, msg.message, n_results=3, max_age_days=90)
        memory_context = build_memory_context(memories)
        if memory_context:
            print(f"[ConversationMemory] Recalled {len(memories)} relevant past conversations")
    except Exception as e:
        print(f"[ConversationMemory] Recall failed (non-fatal): {e}")

    # ── Step 0.7: Knowledge Graph context ────────────────────────
    graph_context = ""
    try:
        kg = load_graph()
        if kg is not None and kg.number_of_nodes() > 0:
            graph_context = build_context_from_graph(kg, msg.message, max_entities=5)
            if graph_context:
                print(f"[KnowledgeGraph] Found related entities for query")
    except Exception as e:
        print(f"[KnowledgeGraph] Lookup failed (non-fatal): {e}")

    # ── Step 1: Try vault retrieval ─────────────────────────────
    if msg.pinned_source:
        # Pinned to a specific file — skip threshold, return top chunks from that file only
        results = col.query(
            query_embeddings=[question_embedding],
            n_results=12,
            where={"source": msg.pinned_source}
        )
        vault_docs  = results["documents"][0] if results["documents"][0] else []
        vault_meta  = results["metadatas"][0] if vault_docs else []
        relevant_docs = [(d, m) for d, m in zip(vault_docs, vault_meta)]
        vault_has_answer = len(relevant_docs) > 0
        use_web   = False
        use_vault = True
    else:
        results = col.query(query_embeddings=[question_embedding], n_results=8)
        vault_docs    = results["documents"][0] if results["documents"][0] else []
        vault_meta    = results["metadatas"][0] if vault_docs else []
        vault_dists   = results["distances"][0] if vault_docs else []

        def is_personal_doc(meta: dict) -> bool:
            src = meta.get("source", "")
            return not (src.startswith("🌐") or src.startswith("http://") or src.startswith("https://"))

        relevant_docs_with_dist = [
            (d, m, dist) for d, m, dist in zip(vault_docs, vault_meta, vault_dists)
            if dist < RELEVANCE_THRESHOLD and is_personal_doc(m)
        ]
        if not relevant_docs_with_dist:
            relevant_docs_with_dist = [
                (d, m, dist) for d, m, dist in zip(vault_docs, vault_meta, vault_dists)
                if dist < 0.85 and is_personal_doc(m)
            ]

        # Re-rank: boost chunks whose section header matches the query keywords
        q_words = set(msg.message.lower().split())
        def rerank_score(doc_text, meta, dist):
            score = dist  # lower is better (L2 distance)
            section = meta.get("section", "").lower()
            if section:
                overlap = len(q_words & set(section.split()))
                score -= overlap * 0.05  # boost by 0.05 per matching keyword
            return score

        relevant_docs_with_dist.sort(key=lambda x: rerank_score(x[0], x[1], x[2]))
        relevant_docs = [(d, m) for d, m, _ in relevant_docs_with_dist]
        vault_has_answer = len(relevant_docs) > 0
        has_user_urls = bool(_extract_urls(msg.message))
        wants_web     = _looks_like_web_search(msg.message) or classification.needs_web
        is_agent_mode = (msg.mode == "agent")

        if has_user_urls:
            use_web = True; use_vault = False
        elif is_agent_mode:
            # Agent mode = user explicitly chose web search
            use_web = True; use_vault = vault_has_answer
        elif wants_web and not vault_has_answer:
            # Only go to web if user explicitly asked AND vault has nothing
            use_web = True; use_vault = False
        elif wants_web and vault_has_answer:
            # User wants web but vault also has docs -- use both
            use_web = True; use_vault = True
        else:
            # DEFAULT: always use vault, never fall back to web silently
            use_web = False; use_vault = True

    skill_block = ""
    if msg.skill == "__custom__" and msg.custom_prompt:
        skill_block = f"\n\n{msg.custom_prompt}"
    elif msg.skill and msg.skill in SKILL_PROMPTS:
        skill_block = f"\n\n{SKILL_PROMPTS[msg.skill]}"

    def generate():
        sections    = []
        all_sources = []
        mode        = "vault"

        # ── Vault context ───────────────────────────────────────
        if use_vault:
            context_parts = []
            for d, m in relevant_docs:
                section_label = m.get("section", "")
                src = m.get("source", "unknown")
                if section_label:
                    context_parts.append(f"[Source: {src} | Section: {section_label}]\n{d}")
                else:
                    context_parts.append(f"[Source: {src}]\n{d}")
            context = "\n\n---\n\n".join(context_parts)
            sources = list(set(m["source"] for _, m in relevant_docs))
            sections.append(f"FROM YOUR PRIVATE DOCUMENTS:\n{context}")
            all_sources.extend(sources)

        # ── Web search ──────────────────────────────────────────
        if use_web:
            mode = "hybrid" if use_vault else "web"
            user_urls = _extract_urls(msg.message)
            structured_listings = []  # Pre-extracted job listings
            web_context = ""
            scraped = 0
            seen_domains = set()
            seen_urls = set()

            # ── Privacy Firewall: sanitize query before ANY web call ──
            firewall_cfg = load_firewall_config()
            sanitized_query, firewall_result = firewall_sanitize(msg.message, firewall_cfg)
            if firewall_result.was_modified:
                stripped_types = set()
                for e in firewall_result.entities_found:
                    t = e.entity_type
                    stripped_types.add(t.value if hasattr(t, "value") else t)
                type_list = ", ".join(sorted(stripped_types))
                fw_msg = f"Stripped {firewall_result.entity_count} entities ({type_list}) before web search"
                yield f"data: {json.dumps({'status': fw_msg})}\n\n"
                print(f"[PrivacyFirewall] Sanitized for web: \"{sanitized_query}\"")
            else:
                sanitized_query = msg.message  # No changes needed

            # ── Phase A: Scrape any URLs the user explicitly provided ──
            is_agency_mode = False
            agency_job_details = []
            if user_urls:
                yield f"data: {json.dumps({'status': '🔗 Fetching the URL you provided…'})}\n\n"
                for u_url in user_urls[:3]:  # Max 3 user URLs
                    if _is_blocked_url(u_url) or u_url in seen_urls:
                        continue
                    seen_urls.add(u_url)
                    domain = urlparse(u_url).netloc
                    seen_domains.add(domain)

                    # ── Staffing Agency Deep-Scrape Mode ──
                    if _is_staffing_agency_url(u_url):
                        is_agency_mode = True
                        print(f"[VaultMind v1.0.0] ✓ Staffing agency detected: {domain}")
                        yield f"data: {json.dumps({'status': f'🏢 Detected staffing agency: {domain}…'})}\n\n"
                        yield f"data: {json.dumps({'status': '📋 Extracting job listings…'})}\n\n"
                        agency_listings = _scrape_agency_listing_page(u_url)
                        if agency_listings:
                            yield f"data: {json.dumps({'status': f'Found {len(agency_listings)} job listings. Deep-scraping details…'})}\n\n"
                            # Use the proper deep-scrape function (JSON-LD + fallback)
                            enriched = _deep_scrape_job_pages(agency_listings, max_pages=15)
                            for listing in enriched:
                                all_sources.append(f"[{listing.get('title','')[:60]}]({listing['url']})")
                            agency_job_details = enriched
                            desc_lengths = [len(e.get('description','')) for e in enriched]
                            print(f"[VaultMind v1.0.0] Deep-scraped {len(enriched)} jobs, desc lengths: {desc_lengths[:5]}...")
                        else:
                            yield f"data: {json.dumps({'status': '⚠️ Could not extract listings from agency page.'})}\n\n"
                        continue  # Skip normal scraping for agency URLs

                    # ── Normal URL scraping ──
                    yield f"data: {json.dumps({'status': f'📄 Reading: {domain}…'})}\n\n"
                    try:
                        r = requests.get(u_url, timeout=12, headers=BROWSER_HEADERS, stream=True)
                        if r.status_code == 200:
                            ctype = r.headers.get("Content-Type", "")
                            if "html" in ctype or "text" in ctype:
                                raw = b""
                                for chunk in r.iter_content(chunk_size=8192):
                                    raw += chunk
                                    if len(raw) > 500_000:
                                        break
                                # Try structured extraction first
                                yield f"data: {json.dumps({'status': '🔎 Extracting listings…'})}\n\n"
                                listings = _extract_job_listings(raw, u_url)
                                if listings:
                                    structured_listings.extend(listings)
                                    for entry in listings:
                                        all_sources.append(f"[{entry['title']}]({entry['url']})")
                                # Also get text for context
                                soup = BeautifulSoup(raw, "html.parser")
                                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                                    tag.decompose()
                                page_text = soup.get_text(separator="\n", strip=True)[:5000]
                                if page_text:
                                    web_context += f"\n\n[User-provided URL]: {u_url}\n{page_text}"
                                    all_sources.append(f"[{domain}]({u_url})")
                                    scraped += 1
                    except Exception as e:
                        yield f"data: {json.dumps({'status': f'⚠️ Could not fetch: {domain}'})}\n\n"

            # ── Phase B: Search the web for additional results ──
            # SKIP web search when in agency mode — the deep-scrape pipeline
            # already has the real data.  Generic web search for "recruiting
            # agency …" just pollutes with Robert Half / Chegg / spam.
            if is_agency_mode and agency_job_details:
                print(f"[VaultMind v1.0.0] Agency mode active with {len(agency_job_details)} listings — skipping Phase B web search")
                search_hits = []
            else:
                yield f"data: {json.dumps({'status': '🔍 Searching the web…'})}\n\n"
                # Use sanitized query for web search (privacy firewall applied)
                search_hits = multi_search(sanitized_query, 12)

            if search_hits:
                for hit in search_hits:
                    if scraped >= 8:
                        break
                    url   = hit.get("href", "")
                    title = hit.get("title", url)
                    body  = hit.get("body", "")
                    if not url or _is_blocked_url(url) or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    domain = urlparse(url).netloc
                    # Hard dedup: max 2 pages from same domain
                    domain_count = sum(1 for d in seen_domains if d == domain)
                    if domain_count >= 2:
                        continue
                    seen_domains.add(domain)
                    yield f"data: {json.dumps({'status': f'📄 Reading: {title[:50]}…'})}\n\n"
                    try:
                        r = requests.get(url, timeout=8, headers=BROWSER_HEADERS, stream=True)
                        if r.status_code == 200:
                            ctype = r.headers.get("Content-Type", "")
                            if "html" in ctype or "text" in ctype:
                                raw = b""
                                for chunk in r.iter_content(chunk_size=8192):
                                    raw += chunk
                                    if len(raw) > 500_000:
                                        break
                                # Try structured extraction on search results too
                                listings = _extract_job_listings(raw, url)
                                if listings:
                                    structured_listings.extend(listings)
                                    for entry in listings[:5]:  # Max 5 per page
                                        if entry['url'] not in seen_urls:
                                            seen_urls.add(entry['url'])
                                            all_sources.append(f"[{entry['title']}]({entry['url']})")
                                soup = BeautifulSoup(raw, "html.parser")
                                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                                    tag.decompose()
                                page_text = soup.get_text(separator="\n", strip=True)[:3000]
                                if page_text:
                                    web_context += f"\n\n[Source #{scraped+1}]: {title}\nURL: {url}\n{page_text}"
                                    all_sources.append(f"[{title}]({url})")
                                    scraped += 1
                            else:
                                if body:
                                    web_context += f"\n\n[Source #{scraped+1}]: {title}\nURL: {url}\n{body}"
                                    all_sources.append(f"[{title}]({url})")
                                    scraped += 1
                        else:
                            if body:
                                web_context += f"\n\n[Source #{scraped+1}]: {title}\nURL: {url}\n{body}"
                                all_sources.append(f"[{title}]({url})")
                                scraped += 1
                    except Exception:
                        if body:
                            web_context += f"\n\n[Source #{scraped+1}]: {title}\nURL: {url}\n{body}"
                            all_sources.append(f"[{title}]({url})")
                            scraped += 1

            # ── Phase C: Build structured data block for the model ──

            # Agency mode: build detailed job descriptions for client ID
            if is_agency_mode and agency_job_details:
                mode = "agency"  # Special mode for agency client identification
                yield f"data: {json.dumps({'status': '🧠 Analyzing clues & identifying clients…'})}\n\n"
                # Run the Python intelligence engine — does the heavy reasoning
                intel_report = analyze_agency_listings(agency_job_details)
                print(f"[VaultMind v1.0.0] Intel report generated: {len(intel_report)} chars")
                print(f"[VaultMind v1.0.0] Report preview: {intel_report[:200]}...")
                sections.append(intel_report)

            elif structured_listings:
                # Deduplicate listings by URL
                unique_listings = []
                listing_urls = set()
                listing_companies = set()
                for entry in structured_listings:
                    url_key = entry["url"]
                    company_key = entry.get("company", "").lower().strip()
                    if url_key not in listing_urls:
                        if company_key and company_key in listing_companies:
                            continue
                        listing_urls.add(url_key)
                        if company_key:
                            listing_companies.add(company_key)
                        unique_listings.append(entry)

                listing_text = "STRUCTURED JOB LISTINGS EXTRACTED FROM WEB PAGES:\n"
                for i, entry in enumerate(unique_listings[:20], 1):
                    listing_text += f"\n{i}. Title: {entry['title']}"
                    if entry.get('company'):
                        listing_text += f"\n   Company: {entry['company']}"
                    if entry.get('location'):
                        listing_text += f"\n   Location: {entry['location']}"
                    listing_text += f"\n   Apply URL: {entry['url']}"
                sections.append(listing_text)

            if web_context:
                # Add quality tier info to web context header
                privacy_label = ""
                if firewall_result.was_modified:
                    privacy_label = " (query was sanitized by Privacy Firewall)"
                sections.append(f"FROM THE WEB{privacy_label} (real search results):\n{web_context}")

            if not web_context and not structured_listings and not agency_job_details:
                yield f"data: {json.dumps({'status': '⚠️ No web results found.'})}\n\n"

        # ── No data at all ──────────────────────────────────────
        if not sections:
            yield f"data: {json.dumps({'token': 'No relevant information found in your documents or the web. Try rephrasing your question or upload some files first.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
            return

        # Stream classification info so the UI can show what VaultMind understood
        intent_label = classification.intent.value.capitalize()
        complexity_label = classification.complexity.value.upper()
        yield f"data: {json.dumps({'status': f'🧠 {intent_label} query (complexity: {complexity_label}) -- using {chat_model}'})}\n\n"
        yield f"data: {json.dumps({'status': '💬 Generating answer...'})}\n\n"

        # ── Build system prompt ─────────────────────────────────
        if mode == "agency":
            system_prompt = (
                "You are a recruiting intelligence formatter. An automated system has already analyzed "
                "staffing agency job listings and identified likely client companies.\n\n"
                "YOUR ONLY JOB: Format the pre-analyzed report below into a clean, readable markdown response.\n\n"
                "FORMATTING RULES:\n"
                "1. Present each job as a numbered entry with the job title as a bold header\n"
                "2. Show the BEST GUESS company name prominently\n"
                "3. List the key clues that led to the identification\n"
                "4. Include the job URL exactly as provided — NEVER change or fabricate URLs\n"
                "5. Group jobs by the identified client company when multiple jobs point to the same client\n"
                "6. Add a summary at the end listing all unique client companies found\n"
                "7. DO NOT add any analysis of your own — just format what's already been analyzed\n"
                "8. DO NOT make up company names that aren't in the report\n"
                "9. If the report says 'Unknown' or 'insufficient clues', say exactly that\n\n"
                f"PRE-ANALYZED REPORT TO FORMAT:\n{'---'.join(sections)}"
                f"{skill_block}"
            )
        elif mode == "web":
            system_prompt = (
                "You are a personal AI assistant with access to REAL web search results.\n\n"
                "ABSOLUTE RULES — VIOLATION = FAILURE:\n"
                "1. ONLY use information from the sources below. If data has a 'STRUCTURED JOB LISTINGS' section, use those entries as your primary answer.\n"
                "2. NEVER fabricate or guess URLs. Copy URLs EXACTLY as they appear in 'Apply URL:' or 'URL:' fields.\n"
                "3. NEVER make up company names, job titles, or facts not in the sources.\n"
                "4. Each item MUST have a DIFFERENT URL. If two items would have the same URL, merge them or drop one.\n"
                "5. If a source is a search results page, do NOT pretend each item on that page has that URL.\n"
                "6. If you only have 5 real results, list 5. NEVER pad with invented entries.\n"
                "7. If the user asked to analyze a specific URL, focus your answer on what was found at that URL.\n"
                "8. When listing companies from a staffing agency page, identify the CLIENT companies (who the agency is hiring for), not the agency itself.\n"
                "9. Use markdown formatting. Number your results.\n\n"
                f"SEARCH RESULTS:\n{'---'.join(sections)}"
                f"{skill_block}"
            )
        elif mode == "hybrid":
            system_prompt = (
                "You are a personal AI assistant. You have access to the user's private documents AND real web search results.\n\n"
                "ABSOLUTE RULES — VIOLATION = FAILURE:\n"
                "1. ONLY use information from the sources below — never invent anything.\n"
                "2. NEVER fabricate URLs. Copy URLs EXACTLY from the sources.\n"
                "3. Each item MUST have a UNIQUE URL. Never repeat the same link.\n"
                "4. If a 'STRUCTURED JOB LISTINGS' section exists, use it as primary data.\n"
                "5. Clearly distinguish between info from private docs vs web.\n"
                "6. Be concise and direct. Use markdown.\n\n"
                f"SOURCES:\n{'---'.join(sections)}"
                f"{skill_block}"
            )
        else:
            # Phase 3: Use adaptive prompt templates (replaces basic Phase 1 templates)
            context_block = '---'.join(sections)
            intent_name = classification.intent.value  # "research", "draft", etc.
            system_prompt = build_adaptive_prompt(
                question=msg.message,
                intent=intent_name,
                context=context_block,
                memory_context=memory_context,
                model=chat_model,
                style="standard",
            )
            system_prompt += skill_block

        # Inject conversation memory if available
        if memory_context:
            system_prompt += f"\n\n{memory_context}"

        # Inject knowledge graph context if available
        if graph_context:
            system_prompt += f"\n\n{graph_context}"

        messages = [{"role": "system", "content": system_prompt}]
        for h in msg.history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": msg.message})

        # ── Stream the LLM response and collect it ──────────────
        full_response = ""
        stream = ollama.chat(model=chat_model, messages=messages, stream=True, options={"temperature": 0})
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

        # ── Phase 3: Quality Gate ──────────────────────────────
        local_ctx = ""
        web_ctx = ""
        for s in sections:
            if s.startswith("FROM YOUR PRIVATE DOCUMENTS"):
                local_ctx = s
            elif s.startswith("FROM THE WEB"):
                web_ctx = s

        try:
            verdict = run_quality_gate(
                response=full_response,
                question=msg.message,
                context="\n".join(sections),
                local_context=local_ctx,
                web_context=web_ctx,
                sources=all_sources,
                use_llm=False,  # Heuristic only for speed
            )
            quality_data = verdict.to_dict()
            yield f"data: {json.dumps({'quality': quality_data})}\n\n"
            print(f"[QualityGate] {verdict.badge_text} (score: {verdict.confidence_score:.2f})")

            # If LOW confidence and we have context, retry with strict prompt
            if verdict.confidence == ConfidenceLevel.LOW and "\n".join(sections).strip():
                yield f"data: {json.dumps({'status': 'Quality check flagged low confidence. Retrying with stricter prompt...'})}\n\n"
                intent_name_retry = classification.intent.value
                strict_prompt = get_retry_prompt(
                    question=msg.message,
                    intent=intent_name_retry,
                    context="\n".join(sections),
                    quality_verdict=quality_data,
                    model=chat_model,
                )
                retry_messages = [
                    {"role": "system", "content": strict_prompt},
                    {"role": "user", "content": msg.message},
                ]
                yield f"data: {json.dumps({'retry_start': True})}\n\n"
                full_response = ""
                retry_stream = ollama.chat(model=chat_model, messages=retry_messages, stream=True, options={"temperature": 0})
                for chunk in retry_stream:
                    token = chunk["message"]["content"]
                    if token:
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
                # Re-run quality gate on retry
                retry_verdict = run_quality_gate(
                    response=full_response,
                    question=msg.message,
                    context="\n".join(sections),
                    local_context=local_ctx,
                    web_context=web_ctx,
                    sources=all_sources,
                    use_llm=False,
                )
                quality_data = retry_verdict.to_dict()
                yield f"data: {json.dumps({'quality': quality_data})}\n\n"
                print(f"[QualityGate] Retry: {retry_verdict.badge_text} (score: {retry_verdict.confidence_score:.2f})")

        except Exception as e:
            print(f"[QualityGate] Error (non-fatal): {e}")
            quality_data = {}

        # ── Phase 3: Citation Engine ───────────────────────────
        try:
            # Build source objects for citation matching
            local_chunks = []
            web_results = []
            if use_vault and relevant_docs:
                for d, m in relevant_docs:
                    local_chunks.append({
                        "text": d,
                        "source": m.get("source", "Unknown"),
                        "section_header": m.get("section", ""),
                    })
            if use_web and web_ctx:
                for src_label in all_sources:
                    # Parse markdown link format: [title](url)
                    link_match = re.match(r'\[(.+?)\]\((.+?)\)', src_label)
                    if link_match:
                        web_results.append({
                            "title": link_match.group(1),
                            "url": link_match.group(2),
                            "snippet": "",
                            "trust_tier": "tier2",
                        })

            if local_chunks or web_results:
                cited = cite_response(full_response, local_chunks, web_results)
                citation_data = format_sources_for_frontend(cited)
                yield f"data: {json.dumps({'citations': citation_data})}\n\n"
                print(f"[CitationEngine] {cited.citation_count} citations, {cited.uncited_claims} uncited claims")
        except Exception as e:
            print(f"[CitationEngine] Error (non-fatal): {e}")

        yield f"data: {json.dumps({'done': True, 'sources': all_sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Inbox Digest ──────────────────────────────────────────────

class DigestRequest(BaseModel):
    connector: str = "gmail"   # source prefix to filter on
    model:     str = "mistral"
    workspace: str = "vault"   # kept for compat, ignored

@app.post("/digest")
async def digest(req: DigestRequest):
    """Retrieve ALL emails from the vault, group by source, stream a ranked summary."""
    col = get_collection()

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
async def status(workspace: str = Query(default="Default")):  # workspace kept for compat
    try:
        count = get_collection().count()
    except Exception:
        count = 0
    return {"chunks_indexed": count, "status": "running"}

VAULTMIND_VERSION = "1.0.0"

@app.get("/health")
async def health():
    try:
        model_names = [m.model for m in ollama.list().models]
        has_embed   = any("nomic-embed-text" in m for m in model_names)
        has_llm     = any(any(x in m for x in ["mistral", "llama3", "phi3", "gemma", "qwen", "deepseek"]) for m in model_names)
        return {"ollama": True, "embed_model": has_embed, "chat_model": has_llm, "ready": has_embed and has_llm, "version": VAULTMIND_VERSION}
    except Exception:
        return {"ollama": False, "embed_model": False, "chat_model": False, "ready": False, "version": VAULTMIND_VERSION}

# ── Query (non-streaming) ─────────────────────────────────────

class QueryMessage(BaseModel):
    message:   str
    mode:      str = "vault"
    workspace: str = "Default"
    model:     str = "mistral"

@app.post("/query")
async def query(msg: QueryMessage):
    try:
        col        = get_collection()
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

# Domains to never scrape — security risks, paywalls, bot detection
SCRAPE_BLOCKLIST = {
    "linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "pinterest.com", "reddit.com",
    "bit.ly", "t.co", "goo.gl", "tinyurl.com",   # URL shorteners
    "malware", "virus", "phishing",                 # safety keywords
}

# Trusted job board domains for direct job link searches
JOB_BOARD_DOMAINS = [
    "greenhouse.io", "lever.co", "ashbyhq.com", "jobs.lever.co",
    "boards.greenhouse.io", "careers.google.com", "amazon.jobs",
    "builtin.com", "builtinla.com", "wellfound.com", "ycombinator.com",
]

# Detect if query is about jobs/hiring
_JOB_INTENT_RE = _re.compile(
    r"\b(hiring|jobs?|openings?|positions?|roles?|careers?|recrui|software engineer|data engineer|data scientist)\b",
    _re.IGNORECASE,
)

def _is_blocked_url(url: str) -> bool:
    """Check if a URL matches the blocklist."""
    url_lower = url.lower()
    return any(d in url_lower for d in SCRAPE_BLOCKLIST)

def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Single DuckDuckGo search."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"Search error: {e}"); return []

def multi_search(query: str, max_total: int = 12) -> list[dict]:
    """Run multiple targeted searches for richer results.
    For job queries, adds targeted job board searches.
    Deduplicates by URL."""
    seen_urls = set()
    all_hits  = []

    def _add_hits(hits):
        for h in hits:
            url = h.get("href", "")
            if url and url not in seen_urls and not _is_blocked_url(url):
                seen_urls.add(url)
                all_hits.append(h)

    # Primary search
    _add_hits(web_search(query, 8))

    # If job-related, add targeted searches for real job boards
    if _JOB_INTENT_RE.search(query) and len(all_hits) < max_total:
        # Extract location and role keywords for targeted queries
        _add_hits(web_search(f"{query} site:greenhouse.io OR site:lever.co", 6))
        if len(all_hits) < max_total:
            _add_hits(web_search(f"{query} site:builtin.com OR site:wellfound.com", 4))

    return all_hits[:max_total]

def smart_scrape(url: str, max_chars: int = 3000) -> str:
    """Safely scrape a page with blocklist, size limit, and timeout."""
    if _is_blocked_url(url):
        return ""
    try:
        r = requests.get(url, timeout=8, headers=BROWSER_HEADERS, stream=True)
        if r.status_code != 200:
            return ""
        # Check content type — only scrape HTML
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype:
            return ""
        # Limit download size to 500KB to avoid huge pages
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 500_000:
                break
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:max_chars]
    except Exception:
        return ""

@app.post("/agent")
async def agent(msg: ChatMessage):
    """Agent mode now uses the exact same pipeline as /chat.
    The /chat endpoint already handles URL detection, staffing agency
    deep-scrape, company intelligence, web search, and vault lookup.
    Keeping a separate dumb endpoint was the root cause of agent mode
    returning generic web search results instead of real analysis."""
    msg.mode = "agent"  # Tag it so /chat knows this came from agent toggle
    print(f"[VaultMind v1.0.0] /agent endpoint → delegating to /chat (message={msg.message[:80]}...)")
    return await chat(msg)
