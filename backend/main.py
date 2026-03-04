from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
import ollama
import pypdf
import json
import io
import requests
from docx import Document
from bs4 import BeautifulSoup
from ddgs import DDGS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("vaultmind_docs")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "mistral"   # better instruction-following than llama3.2, less hallucination


def chunk_text(text: str, chunk_size: int = 150) -> list[str]:
    """Split text into overlapping chunks for precise retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 20):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_text_from_file(contents: bytes, filename: str) -> str:
    """Extract plain text from any supported file type."""
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
        # Return CSV as readable text — row by row
        text = contents.decode("utf-8", errors="ignore")
        return text

    return ""


def embed_and_store(chunks: list[str], source: str):
    """Embed chunks and upsert into ChromaDB."""
    for i, chunk in enumerate(chunks):
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        collection.upsert(
            ids=[f"{source}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": source, "chunk": i}]
        )
        if i % 10 == 0:
            print(f"  ✓ {i}/{len(chunks)}")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest any supported file type."""
    contents = await file.read()
    text = extract_text_from_file(contents, file.filename)

    if not text.strip():
        return {"error": f"Could not extract text from '{file.filename}'. Supported: PDF, DOCX, TXT, MD, CSV"}

    chunks = chunk_text(text)
    print(f"\n📄 Indexing '{file.filename}' — {len(chunks)} chunks")
    embed_and_store(chunks, file.filename)
    print(f"✅ Done: '{file.filename}'")
    return {"message": f"Indexed {file.filename}", "chunks": len(chunks)}


class UrlIngest(BaseModel):
    url: str


@app.post("/ingest-url")
async def ingest_url(data: UrlIngest):
    """Scrape a URL and index its content."""
    BLOCKED_DOMAINS = ["indeed.com", "linkedin.com", "ziprecruiter.com", "glassdoor.com"]
    if any(d in data.url for d in BLOCKED_DOMAINS):
        return {
            "error": (
                f"This site blocks scrapers. Try these instead:\n"
                f"• Company career pages directly (e.g. greenhouse.io, lever.co, workday.com)\n"
                f"• Google: https://www.google.com/search?q=data+engineer+jobs+irvine+ca\n"
                f"• Builtin: https://builtin.com/jobs/data-engineer\n"
                f"• Wellfound (AngelList): https://wellfound.com/jobs"
            )
        }

    try:
        r = requests.get(
            data.url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        if r.status_code == 403:
            return {"error": "This site blocks scrapers (403 Forbidden). Try the company's direct careers page instead."}
        if r.status_code == 429:
            return {"error": "Rate limited (429). Wait a minute and try again, or use a different URL."}
        r.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The site may be slow or blocking requests."}
    except Exception as e:
        return {"error": f"Could not fetch URL: {str(e)}"}

    soup = BeautifulSoup(r.content, "html.parser")

    # Clean out noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else data.url
    text  = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        return {"error": "No readable content found at that URL."}

    # Use the page title as the source name
    source = f"🌐 {title[:80]}"
    chunks = chunk_text(text)
    print(f"\n🌐 Indexing URL '{title}' — {len(chunks)} chunks")
    embed_and_store(chunks, source)
    print(f"✅ Done: '{title}'")
    return {"message": f"Indexed {title}", "chunks": len(chunks), "source": source}


@app.get("/files")
async def list_files():
    results = collection.get(include=["metadatas"])
    files = {}
    for meta in results["metadatas"]:
        src = meta["source"]
        files[src] = files.get(src, 0) + 1
    return {"files": [{"name": k, "chunks": v} for k, v in files.items()]}


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    results = collection.get(where={"source": filename}, include=["metadatas"])
    ids = results["ids"]
    if not ids:
        return {"error": "File not found"}
    collection.delete(ids=ids)
    print(f"🗑️  Deleted '{filename}' ({len(ids)} chunks)")
    return {"message": f"Deleted {filename}", "chunks_removed": len(ids)}


class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/chat")
async def chat(msg: ChatMessage):
    question_embedding = ollama.embeddings(
        model=EMBED_MODEL, prompt=msg.message
    )["embedding"]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=6
    )

    if not results["documents"][0]:
        def no_docs():
            yield f"data: {json.dumps({'token': 'No documents indexed yet. Upload a file or paste a URL to get started.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    context = "\n\n---\n\n".join(results["documents"][0])
    sources  = list(set(m["source"] for m in results["metadatas"][0]))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal AI assistant. You have access ONLY to documents the user has explicitly indexed.\n\n"
                "STRICT RULES — follow these without exception:\n"
                "1. NEVER invent, fabricate, or guess information. No fake names, emails, phone numbers, job listings, companies, or URLs.\n"
                "2. ONLY use information that is literally present in the documents below.\n"
                "3. If the user asks for real-world data (live job listings, real candidate profiles, company contacts) that is NOT in the documents, respond with exactly this format:\n"
                "   'I don't have that data indexed. To get real results, paste the relevant URLs into VaultMind (e.g. a LinkedIn search page, a job board, a company careers page) and I can answer from that real data.'\n"
                "4. Never present strategies or instructions as if they are actual results. If you can only suggest a strategy, say clearly: 'I can suggest a strategy, but I don't have real data for this. Here is what to do to get it:'\n"
                "5. Be concise and direct. Do not pad responses.\n"
                "6. Write in plain prose only. NO markdown formatting — no bold (**text**), no headers (##), no bullet points, no dashes as list items. Just clean sentences and paragraphs.\n\n"
                f"INDEXED DOCUMENTS:\n{context}"
            )
        }
    ]
    for h in msg.history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": msg.message})

    def generate():
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0}
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/status")
async def status():
    return {"chunks_indexed": collection.count(), "status": "running"}


@app.get("/health")
async def health():
    try:
        models_raw = ollama.list()
        model_names = [m.model for m in models_raw.models]
        has_embed = any("nomic-embed-text" in m for m in model_names)
        has_llm   = any("llama3.2" in m for m in model_names)
        return {"ollama": True, "embed_model": has_embed, "chat_model": has_llm, "ready": has_embed and has_llm}
    except Exception:
        return {"ollama": False, "embed_model": False, "chat_model": False, "ready": False}


# ─────────────────────────────────────────────────────────────
#  QUERY — non-streaming endpoint for programmatic use (OpenClaw, etc.)
# ─────────────────────────────────────────────────────────────

class QueryMessage(BaseModel):
    message: str
    mode: str = "vault"   # "vault" or "agent"

@app.post("/query")
async def query(msg: QueryMessage):
    """
    Synchronous query endpoint — returns plain JSON.
    Designed for programmatic callers like OpenClaw skills, scripts, and integrations.
    mode=vault  → answers from your indexed documents only
    mode=agent  → answers from your vault + live DuckDuckGo web search
    """
    try:
        q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]

        # Vault search with relevance filtering
        vault_context = ""
        vault_sources = []
        RELEVANCE_THRESHOLD = 0.75
        v = collection.query(query_embeddings=[q_emb], n_results=4, include=["documents", "metadatas", "distances"])
        if v["documents"][0]:
            relevant_docs = []
            relevant_meta = []
            for doc, meta, dist in zip(v["documents"][0], v["metadatas"][0], v["distances"][0]):
                if dist < RELEVANCE_THRESHOLD:
                    relevant_docs.append(doc)
                    relevant_meta.append(meta)
            if relevant_docs:
                vault_context = "\n\n".join(relevant_docs)
                vault_sources = list(set(m["source"] for m in relevant_meta))

        # Web search (agent mode only)
        web_context = ""
        web_sources = []
        if msg.mode == "agent":
            results = web_search(msg.message, max_results=4)
            for r in results[:3]:
                text = smart_scrape(r.get("href", ""), max_chars=1500)
                if text:
                    web_context += f"\n\nSource: {r.get('title','')}\nURL: {r.get('href','')}\n{text}"
                    web_sources.append(r.get("title", r.get("href", "")))

        # Build context
        sections = []
        if vault_context:
            sections.append(f"FROM YOUR PRIVATE DOCUMENTS:\n{vault_context}")
        if web_context:
            sections.append(f"FROM THE WEB:\n{web_context}")

        if not sections:
            return {"answer": "I don't have any relevant information indexed for that question. Try adding documents or URLs in VaultMind first.", "sources": []}

        full_context = "\n\n---\n\n".join(sections)

        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a personal AI assistant. Answer the question using ONLY the sources below.\n"
                        "NEVER invent information. If the answer isn't in the sources, say so clearly.\n"
                        "Be concise and direct. Plain prose only — no markdown formatting.\n\n"
                        f"SOURCES:\n{full_context}"
                    )
                },
                {"role": "user", "content": msg.message}
            ],
            options={"temperature": 0}
        )
        answer = response["message"]["content"]
        return {"answer": answer, "sources": vault_sources + web_sources, "mode": msg.mode}

    except Exception as e:
        return {"error": str(e), "answer": "VaultMind encountered an error. Is Ollama running?"}


# ─────────────────────────────────────────────────────────────
#  AGENT LAYER — web search + vault search combined
# ─────────────────────────────────────────────────────────────

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
}


def web_search(query: str, max_results: int = 6) -> list[dict]:
    """Search DuckDuckGo — free, no API key required."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"Search error: {e}")
        return []


def smart_scrape(url: str, max_chars: int = 2000) -> str:
    """Scrape a URL and return clean text. Returns empty string on failure."""
    BLOCKED = ["linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com"]
    if any(d in url for d in BLOCKED):
        return ""
    try:
        r = requests.get(url, timeout=8, headers=BROWSER_HEADERS)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception:
        return ""


@app.post("/agent")
async def agent(msg: ChatMessage):
    """
    Agent mode: combines your private vault with live web search.
    Streams status updates → web results → LLM answer.
    """

    def generate():
        # ── Step 1: Search the vault (only include if relevant) ──
        vault_context = ""
        vault_sources = []
        RELEVANCE_THRESHOLD = 0.75  # lower = more similar; skip vault if too far
        try:
            q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=msg.message)["embedding"]
            v = collection.query(query_embeddings=[q_emb], n_results=4, include=["documents", "metadatas", "distances"])
            if v["documents"][0]:
                relevant_docs = []
                relevant_meta = []
                for doc, meta, dist in zip(v["documents"][0], v["metadatas"][0], v["distances"][0]):
                    if dist < RELEVANCE_THRESHOLD:
                        relevant_docs.append(doc)
                        relevant_meta.append(meta)
                if relevant_docs:
                    vault_context = "\n\n".join(relevant_docs)
                    vault_sources = list(set(m["source"] for m in relevant_meta))
        except Exception:
            pass

        # ── Step 2: Search the web ────────────────────────────
        yield f"data: {json.dumps({'status': '🔍 Searching the web...'})}\n\n"

        search_hits = web_search(msg.message, max_results=6)
        if not search_hits:
            yield f"data: {json.dumps({'status': '⚠️ No web results found, using vault only.'})}\n\n"

        # ── Step 3: Scrape top results ────────────────────────
        web_context = ""
        web_sources = []
        scraped = 0

        for hit in search_hits:
            if scraped >= 3:
                break
            url   = hit.get("href", "")
            title = hit.get("title", url)
            body  = hit.get("body", "")

            yield f"data: {json.dumps({'status': f'📄 Reading: {title[:50]}...'})}\n\n"

            # Use snippet if scraping fails or is blocked
            page_text = smart_scrape(url) or body
            if page_text:
                web_context += f"\n\nSource: {title}\nURL: {url}\n{page_text}"
                web_sources.append(f"[{title}]({url})")
                scraped += 1

        yield f"data: {json.dumps({'status': '💬 Generating answer...'})}\n\n"

        # ── Step 4: Build combined context ────────────────────
        sections = []
        if vault_context:
            sections.append(f"FROM YOUR PRIVATE DOCUMENTS:\n{vault_context}")
        if web_context:
            sections.append(f"FROM THE WEB (live results):\n{web_context}")

        if not sections:
            yield f"data: {json.dumps({'token': 'No relevant information found in your vault or on the web for this query.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
            return

        full_context = "\n\n" + "\n\n---\n\n".join(sections)
        all_sources  = vault_sources + web_sources

        # ── Step 5: Stream the answer ─────────────────────────
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a personal AI agent with access to both the user's private documents and live web search results.\n"
                    "Synthesize information from BOTH sources to give the most complete, accurate answer possible.\n"
                    "Clearly distinguish when information comes from private documents vs. the web.\n"
                    "NEVER invent or hallucinate information not present in the sources below.\n"
                    "Be direct and actionable.\n"
                    "Write in plain prose only. NO markdown formatting — no bold (**text**), no headers (##), no bullet points, no dashes as list items. Just clean sentences and paragraphs.\n\n"
                    f"SOURCES:\n{full_context}"
                )
            }
        ]
        for h in msg.history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": msg.message})

        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0}
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"

        yield f"data: {json.dumps({'done': True, 'sources': all_sources})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
