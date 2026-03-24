#!/bin/bash
# Live Assistant — One-click installer for macOS
set -e

echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║   Live Assistant — Installing...  ║"
echo "  ╚═══════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install from https://python.org"
  exit 1
fi
echo "✓ Python $(python3 --version | cut -d' ' -f2)"

# Check Homebrew
if ! command -v brew &>/dev/null; then
  echo "❌ Homebrew not found. Install from https://brew.sh"
  exit 1
fi
echo "✓ Homebrew"

# Install system deps
echo ""
echo "Installing system dependencies..."
brew install portaudio tesseract 2>/dev/null || true

# Check Ollama
if ! command -v ollama &>/dev/null; then
  echo ""
  echo "⚠️  Ollama not found. Install from https://ollama.com"
  echo "   Then run: ollama pull gemma3:4b"
  echo ""
else
  echo "✓ Ollama"
  if ! ollama list 2>/dev/null | grep -q "gemma3"; then
    echo "  Pulling gemma3:4b model (this may take a few minutes)..."
    ollama pull gemma3:4b
  fi
  echo "✓ gemma3:4b model"
fi

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install faster-whisper sounddevice numpy rich fastapi uvicorn python-multipart pymupdf python-docx pytesseract Pillow anthropic 2>&1 | tail -1

echo ""
echo "  ✅ Installation complete!"
echo ""
echo "  To start: python3 web_app.py"
echo "  Then open: http://localhost:8765"
echo ""
