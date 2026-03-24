#!/bin/bash
# Deploy Live Assistant to GitHub + GitHub Pages
set -e

cd "$(dirname "$0")"

REPO_NAME="live-assistant"

echo ""
echo "Deploy Live Assistant to GitHub"
echo "==============================="
echo ""

# Check gh CLI
if ! command -v gh &>/dev/null; then
  echo "❌ GitHub CLI (gh) not found. Install: brew install gh"
  exit 1
fi

# Check auth
if ! gh auth status &>/dev/null 2>&1; then
  echo "❌ Not logged in to GitHub. Run: gh auth login"
  exit 1
fi

USERNAME=$(gh api user -q .login)
echo "GitHub user: $USERNAME"

# Init git if needed
if [ ! -d .git ]; then
  git init
  git branch -M main
fi

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.license
uploads/
screenshots/
sessions/
.env
EOF

# Create GitHub repo
if ! gh repo view "$USERNAME/$REPO_NAME" &>/dev/null 2>&1; then
  echo "Creating repository..."
  gh repo create "$REPO_NAME" --public --description "Real-time AI interview coach" --source . --remote origin
else
  echo "Repository already exists"
  git remote set-url origin "https://github.com/$USERNAME/$REPO_NAME.git" 2>/dev/null || \
  git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git" 2>/dev/null || true
fi

# Commit and push
git add -A
git commit -m "Live Assistant v2 — real-time interview AI coach" 2>/dev/null || echo "Nothing to commit"
git push -u origin main

# Enable GitHub Pages
echo ""
echo "Enabling GitHub Pages..."
gh api repos/$USERNAME/$REPO_NAME/pages -X POST -f source='{"branch":"main","path":"/docs"}' 2>/dev/null || \
gh api repos/$USERNAME/$REPO_NAME/pages -X PUT -f source='{"branch":"main","path":"/docs"}' 2>/dev/null || true

echo ""
echo "✅ Deployed!"
echo ""
echo "  Repository: https://github.com/$USERNAME/$REPO_NAME"
echo "  Landing page: https://$USERNAME.github.io/$REPO_NAME/"
echo "  Releases: https://github.com/$USERNAME/$REPO_NAME/releases"
echo ""
echo "Next steps:"
echo "  1. Update docs/index.html with your GitHub username"
echo "  2. Set up Stripe and update the payment link"
echo "  3. Create a release: gh release create v1.0 --notes 'Initial release'"
echo ""
