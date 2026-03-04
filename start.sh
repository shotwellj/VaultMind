#!/bin/bash

# ─────────────────────────────────────────────
#  VaultMind — One-command launcher
#  Run: bash start.sh
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"
FRONTEND="$SCRIPT_DIR/frontend/index.html"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "🔒 VaultMind — Starting up..."
echo "────────────────────────────────────────"

# ── Step 1: Check Ollama is installed ────────
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌  Ollama not found.${NC}"
    echo ""
    echo "VaultMind needs Ollama to run AI locally."
    echo "Opening the download page now..."
    echo ""
    open "https://ollama.ai/download"
    echo "After installing Ollama, run this script again."
    exit 1
fi
echo -e "${GREEN}✓${NC}  Ollama found"

# ── Step 2: Start Ollama if it's not running ─
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   Starting Ollama..."
    open -a Ollama 2>/dev/null || ollama serve > /dev/null 2>&1 &
    # Wait up to 10 seconds for it to come up
    for i in {1..10}; do
        sleep 1
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then break; fi
    done
fi
echo -e "${GREEN}✓${NC}  Ollama running"

# ── Step 3: Pull models if missing ───────────
MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print(' '.join([m['name'] for m in d.get('models',[])]))" 2>/dev/null)

if echo "$MODELS" | grep -q "nomic-embed-text"; then
    echo -e "${GREEN}✓${NC}  Embedding model ready"
else
    echo -e "${YELLOW}↓${NC}  Downloading embedding model (274 MB)..."
    ollama pull nomic-embed-text
fi

if echo "$MODELS" | grep -q "llama3.2"; then
    echo -e "${GREEN}✓${NC}  Language model ready"
else
    echo -e "${YELLOW}↓${NC}  Downloading language model (~2 GB, one-time only)..."
    ollama pull llama3.2
fi

# ── Step 4: Install Python deps quietly ──────
echo "   Checking Python dependencies..."
pip3 install -r "$BACKEND/requirements.txt" -q --break-system-packages 2>/dev/null
echo -e "${GREEN}✓${NC}  Dependencies ready"

# ── Step 5: Free port 8000 if something's on it
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# ── Step 6: Start the backend ─────────────────
echo "   Starting backend..."
cd "$BACKEND"
uvicorn main:app --port 8000 > "$SCRIPT_DIR/vaultmind.log" 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
for i in {1..15}; do
    sleep 1
    if curl -s http://localhost:8000/status > /dev/null 2>&1; then break; fi
done

if ! curl -s http://localhost:8000/status > /dev/null 2>&1; then
    echo -e "${RED}❌  Backend failed to start. Check vaultmind.log for details.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC}  Backend running on localhost:8000"

# ── Step 7: Open the UI ───────────────────────
echo ""
echo -e "${GREEN}✅  VaultMind is ready!${NC}"
echo "────────────────────────────────────────"
echo ""
open "$FRONTEND"

# Keep running — catch Ctrl+C to shut everything down cleanly
echo "Press Ctrl+C to stop VaultMind."
echo ""
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID 2>/dev/null; exit 0" INT
wait $BACKEND_PID
