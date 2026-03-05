/**
 * VaultMind — Electron Main Process
 *
 * What this file does:
 *  1. On first launch: finds Python 3, creates a virtualenv, installs deps
 *  2. Spawns the FastAPI backend (uvicorn) as a child process
 *  3. Waits for the backend to be ready
 *  4. Opens the app in a native Mac window (BrowserWindow → localhost:8000)
 *  5. Cleans up the backend process when the app quits
 */

const { app, BrowserWindow, dialog, shell } = require('electron');
const { spawn, execSync }                    = require('child_process');
const path                                   = require('path');
const http                                   = require('http');
const https                                  = require('https');
const fs                                     = require('fs');

// ── Constants ────────────────────────────────────────────────────────────────

const PORT        = 8000;
const BACKEND_DIR = path.join(__dirname, 'backend');

// User data dir: ~/Library/Application Support/VaultMind  (persists across app updates)
const DATA_DIR    = path.join(app.getPath('userData'), 'data');

// Python virtual environment lives in user data so it also survives app updates
const VENV_DIR    = path.join(app.getPath('userData'), 'venv');
const VENV_PYTHON = path.join(VENV_DIR, 'bin', 'python3');
const VENV_PIP    = path.join(VENV_DIR, 'bin', 'pip3');

// Ollama management — bundled binary lives in user data dir
const OLLAMA_DIR   = path.join(app.getPath('userData'), 'ollama');
const OLLAMA_BIN   = process.platform === 'win32'
                     ? path.join(OLLAMA_DIR, 'ollama.exe')
                     : path.join(OLLAMA_DIR, 'ollama');
const OLLAMA_PORT  = 11434;
const OLLAMA_VER   = '0.6.2';    // pinned version to download
const REQUIRED_MODELS = ['nomic-embed-text', 'mistral'];

// ── State ────────────────────────────────────────────────────────────────────

let mainWindow    = null;
let loadingWindow = null;
let backendProc   = null;
let ollamaProc    = null;   // only set if WE spawned ollama (not system)

// ── Python detection ─────────────────────────────────────────────────────────

/**
 * Try common Python 3 locations on macOS.
 * Returns the path to python3 if found, or null.
 */
function findPython() {
  const candidates = [
    '/opt/homebrew/bin/python3',  // Apple Silicon Homebrew
    '/usr/local/bin/python3',     // Intel Homebrew
    '/usr/bin/python3',           // Xcode Command Line Tools
    'python3',                    // PATH fallback
  ];
  for (const p of candidates) {
    try {
      execSync(`"${p}" --version 2>&1`, { stdio: 'ignore' });
      return p;
    } catch (_) { /* try next */ }
  }
  return null;
}

// ── Ollama management ────────────────────────────────────────────────────────

/**
 * Check if Ollama is already running (system install or previous session).
 */
function isOllamaRunning() {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${OLLAMA_PORT}/api/version`, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.setTimeout(2000);
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

/**
 * Find an existing Ollama binary on the system.
 * Checks common install locations, then our bundled copy.
 */
function findOllama() {
  const candidates = process.platform === 'win32'
    ? ['ollama', OLLAMA_BIN]
    : [
        '/usr/local/bin/ollama',
        '/opt/homebrew/bin/ollama',
        path.join(process.env.HOME || '', '.ollama', 'ollama'),
        OLLAMA_BIN,                // our downloaded copy
        'ollama',                  // PATH fallback
      ];

  for (const p of candidates) {
    try {
      execSync(`"${p}" --version 2>&1`, { stdio: 'ignore', timeout: 5000 });
      return p;
    } catch (_) { /* try next */ }
  }
  return null;
}

/**
 * Download the Ollama CLI binary to our app data directory.
 * Shows progress in the loading window.
 */
function downloadOllama() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(OLLAMA_DIR)) fs.mkdirSync(OLLAMA_DIR, { recursive: true });

    const platform = process.platform === 'win32' ? 'windows-amd64.exe' : 'darwin';
    const url = `https://github.com/ollama/ollama/releases/download/v${OLLAMA_VER}/ollama-${platform}`;

    setLoadingStatus('Downloading Ollama AI engine…', 15);

    // Follow redirects (GitHub releases redirect to CDN)
    const download = (url, redirects = 0) => {
      if (redirects > 5) return reject(new Error('Too many redirects downloading Ollama'));

      const proto = url.startsWith('https') ? https : http;
      proto.get(url, { headers: { 'User-Agent': 'VaultMind-App' } }, (res) => {
        // Handle redirect
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return download(res.headers.location, redirects + 1);
        }

        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`Download failed: HTTP ${res.statusCode}`));
        }

        const totalBytes = parseInt(res.headers['content-length'] || '0', 10);
        let downloaded = 0;
        const tmpPath = OLLAMA_BIN + '.tmp';
        const file = fs.createWriteStream(tmpPath);

        res.on('data', (chunk) => {
          downloaded += chunk.length;
          file.write(chunk);
          if (totalBytes > 0) {
            const pct = Math.round((downloaded / totalBytes) * 40) + 15; // 15-55%
            const mb = (downloaded / 1024 / 1024).toFixed(0);
            const totalMb = (totalBytes / 1024 / 1024).toFixed(0);
            setLoadingStatus(`Downloading Ollama… ${mb}/${totalMb} MB`, pct);
          }
        });

        res.on('end', () => {
          file.end(() => {
            // Rename tmp → final, set executable permission
            fs.renameSync(tmpPath, OLLAMA_BIN);
            if (process.platform !== 'win32') {
              fs.chmodSync(OLLAMA_BIN, 0o755);
            }
            resolve(OLLAMA_BIN);
          });
        });

        res.on('error', (err) => {
          file.destroy();
          try { fs.unlinkSync(tmpPath); } catch (_) {}
          reject(err);
        });
      }).on('error', reject);
    };

    download(url);
  });
}

/**
 * Start `ollama serve` as a background process.
 * Returns a promise that resolves when the server is accepting connections.
 */
function startOllamaServer(ollamaBin) {
  return new Promise((resolve, reject) => {
    setLoadingStatus('Starting Ollama AI engine…', 58);

    ollamaProc = spawn(ollamaBin, ['serve'], {
      stdio: 'pipe',
      env: {
        ...process.env,
        OLLAMA_HOST: `127.0.0.1:${OLLAMA_PORT}`,
      },
    });

    ollamaProc.stderr.on('data', d => console.log('[ollama]', d.toString().trimEnd()));
    ollamaProc.on('error', err => console.error('[ollama] spawn error:', err.message));

    // Poll until server is ready
    let attempts = 0;
    const maxAttempts = 30;
    const check = () => {
      isOllamaRunning().then(running => {
        if (running) return resolve();
        attempts++;
        if (attempts >= maxAttempts) return reject(new Error('Ollama server did not start in time.'));
        setTimeout(check, 1000);
      });
    };
    setTimeout(check, 1500);
  });
}

/**
 * Pull a model via Ollama API, showing progress in the loading window.
 */
function pullModel(modelName, baseProgress, progressRange) {
  return new Promise((resolve, reject) => {
    const postData = JSON.stringify({ name: modelName });

    const req = http.request({
      hostname: '127.0.0.1',
      port: OLLAMA_PORT,
      path: '/api/pull',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    }, (res) => {
      let buf = '';
      res.on('data', (chunk) => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.status === 'success') {
              setLoadingStatus(`✓ ${modelName} ready`, baseProgress + progressRange);
              return; // resolve will fire on 'end'
            }
            // Show download progress
            if (data.total && data.completed) {
              const modelPct = data.completed / data.total;
              const pct = baseProgress + Math.round(modelPct * progressRange);
              const mb = (data.completed / 1024 / 1024).toFixed(0);
              const totalMb = (data.total / 1024 / 1024).toFixed(0);
              setLoadingStatus(`Downloading ${modelName}… ${mb}/${totalMb} MB`, pct);
            } else if (data.status) {
              setLoadingStatus(`${modelName}: ${data.status}`, baseProgress);
            }
          } catch (_) {}
        }
      });
      res.on('end', resolve);
      res.on('error', reject);
    });

    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

/**
 * Ensure required models are available — pulls any that are missing.
 */
async function ensureModels(ollamaBin) {
  for (let i = 0; i < REQUIRED_MODELS.length; i++) {
    const model = REQUIRED_MODELS[i];
    const baseProgress = 60 + (i * 15);  // 60-75 for first model, 75-90 for second

    // Check if model already exists
    try {
      execSync(`"${ollamaBin}" list 2>&1`, { timeout: 10000 })
        .toString();
      const listOutput = execSync(`"${ollamaBin}" list 2>&1`, { timeout: 10000 }).toString();
      if (listOutput.includes(model)) {
        setLoadingStatus(`✓ ${model} ready`, baseProgress + 15);
        continue;
      }
    } catch (_) { /* can't check — just try pulling */ }

    setLoadingStatus(`Checking ${model}…`, baseProgress);
    await pullModel(model, baseProgress, 15);
  }
}

/**
 * Master function: ensure Ollama is running + models are available.
 * 1. Check if Ollama is already running (system install) → use it
 * 2. Find Ollama binary → start server
 * 3. No binary found → download it → start server
 * 4. Pull required models
 */
async function ensureOllama() {
  // Already running? Great, skip everything.
  if (await isOllamaRunning()) {
    setLoadingStatus('Ollama detected ✓', 58);
    const bin = findOllama() || 'ollama';
    await ensureModels(bin);
    return;
  }

  // Find or download binary
  let ollamaBin = findOllama();
  if (!ollamaBin) {
    setLoadingStatus('Ollama not found — downloading…', 12);
    ollamaBin = await downloadOllama();
  }

  // Start the server
  await startOllamaServer(ollamaBin);

  // Pull models
  await ensureModels(ollamaBin);
}

/** Kill ollama server if we started it. */
function killOllama() {
  if (ollamaProc) {
    ollamaProc.kill('SIGTERM');
    ollamaProc = null;
  }
}

// ── Backend health check ─────────────────────────────────────────────────────

/**
 * Polls GET /health every second until the backend responds 200,
 * or rejects after `maxRetries` attempts.
 */
function waitForBackend(maxRetries = 45) {
  return new Promise((resolve, reject) => {
    let attempts = 0;

    const check = () => {
      const req = http.get(`http://127.0.0.1:${PORT}/health`, (res) => {
        res.resume(); // discard body
        if (res.statusCode === 200) return resolve();
        retry();
      });
      req.setTimeout(1500);
      req.on('error',   retry);
      req.on('timeout', () => { req.destroy(); retry(); });
    };

    const retry = () => {
      attempts++;
      if (attempts >= maxRetries) {
        reject(new Error('VaultMind backend took too long to start.\n\nCheck that Ollama is installed: https://ollama.ai/download'));
      } else {
        setTimeout(check, 1000);
      }
    };

    setTimeout(check, 800); // small initial delay
  });
}

// ── Loading window ───────────────────────────────────────────────────────────

/** Shows a minimal splash screen while we set up / start the backend. */
function showLoadingWindow() {
  loadingWindow = new BrowserWindow({
    width:     380,
    height:    220,
    frame:     false,
    resizable: false,
    center:    true,
    alwaysOnTop: true,
    backgroundColor: '#0f0f0f',
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });

  const html = `<!DOCTYPE html><html><head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:#0f0f0f; color:#e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; height:100vh; gap:14px;
    -webkit-app-region: drag;
  }
  .icon  { font-size:40px; }
  .title { font-size:18px; font-weight:700; color:#fff; }
  .msg   { font-size:12px; color:#555; min-height:16px; }
  .track { width:200px; height:2px; background:#1a1a1a; border-radius:2px; overflow:hidden; }
  .bar   { height:100%; background:#6366f1; width:10%; border-radius:2px; transition:width .5s ease; }
</style></head><body>
  <div class="icon">🔒</div>
  <div class="title">VaultMind</div>
  <div class="msg" id="msg">Starting up…</div>
  <div class="track"><div class="bar" id="bar"></div></div>
</body></html>`;

  loadingWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
}

/** Update the splash message and progress bar. */
function setLoadingStatus(msg, pct) {
  if (!loadingWindow || loadingWindow.isDestroyed()) return;
  loadingWindow.webContents.executeJavaScript(`
    document.getElementById('msg').textContent = ${JSON.stringify(msg)};
    document.getElementById('bar').style.width = '${Math.min(100, pct)}%';
  `).catch(() => {});
}

// ── First-run setup ──────────────────────────────────────────────────────────

/** Runs a shell command and returns a promise. */
function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: 'pipe', ...opts });
    let stderr = '';
    if (proc.stderr) proc.stderr.on('data', d => { stderr += d.toString(); });
    proc.on('close', code => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} failed:\n${stderr.slice(-600)}`));
    });
    proc.on('error', err => reject(new Error(`Could not run ${cmd}: ${err.message}`)));
  });
}

/**
 * Creates a Python venv and installs requirements if one doesn't exist yet.
 * Only runs on first launch — subsequent launches skip straight to backend start.
 */
async function ensureVenv(python) {
  if (fs.existsSync(VENV_PYTHON)) return; // already set up

  setLoadingStatus('First time setup — this takes about a minute…', 15);

  // Create the virtual environment
  try {
    await run(python, ['-m', 'venv', VENV_DIR]);
  } catch (err) {
    throw new Error(
      `Could not create Python environment.\n\n${err.message}\n\n` +
      'Make sure Python 3.10+ is installed: https://www.python.org/downloads/'
    );
  }

  setLoadingStatus('Installing Python dependencies…', 35);

  // Install requirements into the venv
  try {
    await run(VENV_PIP, [
      'install', '-r', path.join(BACKEND_DIR, 'requirements.txt'), '--quiet'
    ]);
  } catch (err) {
    // Clean up broken venv so next launch tries again
    try { fs.rmSync(VENV_DIR, { recursive: true, force: true }); } catch (_) {}
    throw new Error(`Failed to install dependencies:\n\n${err.message}`);
  }
}

// ── Backend ──────────────────────────────────────────────────────────────────

/** Spawns the FastAPI backend as a child process. */
function startBackend() {
  // Ensure the user data directory exists
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

  backendProc = spawn(
    VENV_PYTHON,
    ['-m', 'uvicorn', 'main:app',
     '--host', '127.0.0.1',
     '--port', String(PORT),
     '--log-level', 'warning'],
    {
      cwd: BACKEND_DIR,
      env: {
        ...process.env,
        VAULTMIND_DATA_DIR: DATA_DIR, // tells backend where to store all data
      },
    }
  );

  backendProc.stderr.on('data', d => console.log('[backend]', d.toString().trimEnd()));
  backendProc.on('error', err  => console.error('[backend] spawn error:', err.message));
  backendProc.on('exit',  code => {
    if (code !== null && code !== 0 && code !== undefined) {
      console.error('[backend] exited with code', code);
    }
  });
}

/** Kill the backend cleanly. */
function killBackend() {
  if (backendProc) {
    backendProc.kill('SIGTERM');
    backendProc = null;
  }
}

// ── Main window ──────────────────────────────────────────────────────────────

async function createMainWindow() {
  mainWindow = new BrowserWindow({
    width:    1280,
    height:   820,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',  // native Mac traffic lights, no title text
    backgroundColor: '#0f0f0f',
    show: false,                   // show only after page loads (prevents white flash)
    webPreferences: {
      nodeIntegration:  false,
      contextIsolation: true,
    },
  });

  // External links (Google OAuth popup, ollama.ai, etc.) → open in Safari/Chrome
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Prevent navigating away from the app inside the window
  mainWindow.webContents.on('will-navigate', (e, url) => {
    const isLocal = url.startsWith(`http://127.0.0.1:${PORT}`)
                 || url.startsWith(`http://localhost:${PORT}`);
    if (!isLocal) {
      e.preventDefault();
      shell.openExternal(url);
    }
  });

  // Close loading splash, show main window
  mainWindow.once('ready-to-show', () => {
    if (loadingWindow && !loadingWindow.isDestroyed()) loadingWindow.close();
    mainWindow.show();
  });

  await mainWindow.loadURL(`http://127.0.0.1:${PORT}`);
}

// ── Auto-update check ────────────────────────────────────────────────────────

/**
 * Silently checks GitHub Releases for a newer version.
 * If found, shows a dialog offering to open the download page.
 * Works on Mac (unsigned) and Windows — no code signing required.
 */
function checkForUpdates() {
  const options = {
    hostname: 'api.github.com',
    path:     '/repos/airblackbox/VaultMind/releases/latest',
    headers:  { 'User-Agent': 'VaultMind-App' },
    timeout:  8000,
  };

  const req = https.get(options, (res) => {
    let body = '';
    res.on('data', chunk => { body += chunk; });
    res.on('end', () => {
      try {
        const data          = JSON.parse(body);
        const latestTag     = (data.tag_name  || '').replace(/^v/, '');  // e.g. "0.2.0"
        const currentVer    = app.getVersion();                           // from package.json

        if (!latestTag || latestTag === currentVer) return; // already up to date

        // Simple semver comparison — just compare the strings numerically
        const newer = latestTag.split('.').map(Number);
        const curr  = currentVer.split('.').map(Number);
        const isNewer = newer.some((n, i) => n > (curr[i] || 0));
        if (!isNewer) return;

        // Show non-blocking update dialog
        dialog.showMessageBox(mainWindow, {
          type:    'info',
          title:   'Update Available',
          message: `VaultMind v${latestTag} is ready`,
          detail:  `You're on v${currentVer}. Download the latest version from GitHub?`,
          buttons: ['Download Update', 'Later'],
          defaultId: 0,
          cancelId:  1,
        }).then(({ response }) => {
          if (response === 0) {
            shell.openExternal(`https://github.com/airblackbox/VaultMind/releases/tag/v${latestTag}`);
          }
        });

      } catch (_) { /* ignore parse errors */ }
    });
  });

  req.on('error',   () => { /* ignore network errors — no internet, no update check */ });
  req.on('timeout', () => { req.destroy(); });
}

// ── App initialization ───────────────────────────────────────────────────────

async function initialize() {
  showLoadingWindow();

  try {
    // Step 1: Ensure Ollama is running + models are pulled
    setLoadingStatus('Checking AI engine…', 5);
    await ensureOllama();

    // Step 2: Find Python
    setLoadingStatus('Checking Python…', 90);
    const python = findPython();
    if (!python) {
      throw new Error(
        'Python 3 is required but was not found.\n\n' +
        'Install it via Homebrew:  brew install python3\n' +
        'or download from:  https://www.python.org/downloads/'
      );
    }

    // Step 3: Set up venv (skipped on every launch except the first)
    await ensureVenv(python);

    // Step 4: Start the backend
    setLoadingStatus('Starting VaultMind…', 94);
    startBackend();

    // Step 5: Wait for it to be ready
    setLoadingStatus('Loading AI backend…', 96);
    await waitForBackend();

    // Step 6: Open the app
    setLoadingStatus('Ready!', 100);
    await createMainWindow();

    // Check for updates in the background — 5 second delay so it doesn't
    // interrupt the initial load experience
    setTimeout(checkForUpdates, 5000);

  } catch (err) {
    if (loadingWindow && !loadingWindow.isDestroyed()) loadingWindow.close();
    dialog.showErrorBox('VaultMind could not start', err.message);
    app.quit();
  }
}

// ── Lifecycle hooks ──────────────────────────────────────────────────────────

app.whenReady().then(initialize);

// macOS: re-open window when dock icon is clicked and no windows are open
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    // Backend is already running — just open a new window
    createMainWindow().catch(console.error);
  }
});

// Kill backend + ollama when all windows are closed (non-macOS quits the app too)
app.on('window-all-closed', () => {
  killBackend();
  killOllama();
  if (process.platform !== 'darwin') app.quit();
});

// Always kill backend + ollama before quitting (catches Cmd+Q, dock quit, etc.)
app.on('before-quit', () => {
  killBackend();
  killOllama();
});
