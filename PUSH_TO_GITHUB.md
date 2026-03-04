# Push VaultMind to GitHub

## Step 1 — Create the repo on GitHub
Go to: https://github.com/organizations/air-blackbox/repositories/new

Settings:
- Repository name: `vaultmind`
- Description: `Chat with your documents using local LLMs. Nothing leaves your machine.`
- Public ✅
- Do NOT initialize with README (we have one)

## Step 2 — Push from your Desktop

Paste this in terminal:

```bash
cd ~/Desktop/vaultmind
git init
git add .
git commit -m "Initial commit — VaultMind POC"
git branch -M main
git remote add origin https://github.com/air-blackbox/vaultmind.git
git push -u origin main
```

## Step 3 — Add a .gitignore first (before pushing)

Run this BEFORE the git commands above:

```bash
cat > ~/Desktop/vaultmind/.gitignore << 'EOF'
# Don't commit the vector database (user data)
backend/chroma_db/

# Python
__pycache__/
*.pyc
.env

# Mac
.DS_Store
EOF
```

So the full order is:
1. Create repo on GitHub (Step 1)
2. Add .gitignore (Step 3 commands)
3. Push (Step 2 commands)
