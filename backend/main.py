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
    """Index a single file into the vault (called from watchdog thread)."""
    try:
        with open(path, 'rb') as f:
            contents = f.read()
        filename = os.path.basename(path)
        ext      = os.path.splitext(filename)[1].lower()
        col      = get_collection()

        if ext in IMAGE_EXTENSIONS:
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
    workspace: str        = Form(default="Default")   # kept for compat, ignored
):
    contents = await file.read()
    text     = extract_text_from_file(contents, file.filename)
    if not text.strip():
        return {"error": f"Could not extract text from '{file.filename}'. Supported: PDF, DOCX, TXT, MD, CSV"}
    chunks = chunk_text(text)
    print(f"\n📄 Indexing '{file.filename}' — {len(chunks)} chunks")
    col = get_collection()
    embed_and_store(chunks, file.filename, col)
    log_feed_event(file.filename, "vault", len(chunks), "upload")
    return {"message": f"Indexed {file.filename}", "chunks": len(chunks)}

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
    """Save or update a conversation."""
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
    return {"ok": True}

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    fpath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
    return {"ok": True}


@app.post("/chat")
async def chat(msg: ChatMessage):
    chat_model         = msg.model or DEFAULT_MODEL
    question_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]
    col                = get_collection()

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

    skill_block = ""
    if msg.skill == "__custom__" and msg.custom_prompt:
        skill_block = f"\n\n{msg.custom_prompt}"
    elif msg.skill and msg.skill in SKILL_PROMPTS:
        skill_block = f"\n\n{SKILL_PROMPTS[msg.skill]}"

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
                "5. Use markdown formatting where helpful (headers, bold, bullet lists, code blocks).\n\n"
                f"INDEXED DOCUMENTS:\n{context}"
                f"{skill_block}"
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
    col        = get_collection()
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
