<div align="center">

# Live Assistant

### Your real-time AI interview coach

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)

**Transcribes conversations in real-time. Detects interview questions. Generates spoken answers and working code — all running locally on your machine.**

[Get Started](#-quick-start) · [Features](#-features) · [Pro Version](#-free-vs-pro) · [Landing Page](https://cherishchencherish.github.io/live-assistant/)

</div>

---

## How It Works

```
🎙 Audio In ──→ Whisper (local) ──→ Transcript ──→ Question Detector ──→ Ollama (local) ──→ Response
   mic/zoom        transcription       real-time       2-stage filter       code + verbal      display
```

Everything runs **locally**. No data leaves your machine. No API costs.

---

## Screenshots

> Coming soon — run the app and see for yourself!

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/CherishChenCherish/live-assistant.git
cd live-assistant
./install.sh
```

Or manually:

```bash
# System deps
brew install portaudio tesseract

# Ollama (for AI responses)
# Download from https://ollama.com, then:
ollama pull gemma3:4b

# Python packages
pip install faster-whisper sounddevice numpy rich fastapi uvicorn \
  python-multipart pymupdf python-docx pytesseract Pillow
```

### 2. Launch

```bash
python3 web_app.py
```

Opens at **http://localhost:8765** — select your audio device, upload your resume, and hit Start.

---

## Features

### Free Tier

| Feature | Description |
|---------|-------------|
| **Real-time Transcription** | Whisper-powered, runs locally, auto-detects language |
| **Meeting App Detection** | Auto-detects Zoom, Teams, Google Meet, WeMeet and selects the right audio device |
| **Speaking Pace Monitor** | Shows your WPM — warns if you're speaking too fast or too slow |
| **Answer Timer** | Visual countdown — green → yellow → red to keep answers under 2 minutes |
| **Session Recording** | Auto-saves transcripts and Q&A to `sessions/` on exit |
| **Manual Question Input** | Type or paste questions directly into the toolbar |

### Pro Tier

| Feature | Description |
|---------|-------------|
| **AI Response Generation** | Detects real interview questions and generates natural spoken English answers |
| **Technical Code Answers** | For coding questions: working code + verbal explanation + complexity analysis |
| **Context Upload** | Upload resume, JD, interviewer info (.pdf .txt .md .docx) — AI personalizes answers |
| **Follow-up Predictions** | Predicts the interviewer's likely next question |
| **Keyword Hints** | Shows key talking points instead of full answers — speak more naturally |
| **Screenshot OCR** | `Ctrl+V` a screenshot of a coding problem — OCR extracts text and generates a solution |
| **Screen Monitoring** | Auto-captures screen periodically to detect new questions |
| **SOS Button** | Press `S` when stuck — shows a natural stalling phrase to buy thinking time |
| **Regenerate** | Don't like an answer? Regenerate with a different angle |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume listening |
| `R` | Regenerate last response |
| `S` | SOS — show a stalling phrase |
| `Ctrl+V` | Paste screenshot for OCR |

---

## Smart Question Detection

Not every sentence triggers a response. The 2-stage detection system filters out:

```
Stage 1 (Fast)                          Stage 2 (LLM)
─────────────                          ──────────────
✓ "Tell me about yourself"  → strong ──→ Ollama confirms → ✅ Generate
✓ "What's your approach?"   → weak   ──→ Ollama confirms → ✅ Generate
✗ "What we do here is..."   → blocked                     → ❌ Skip
✗ "You know, the thing..."  → blocked                     → ❌ Skip
✗ "There's a question of..."→ blocked                     → ❌ Skip
```

Lecture content, rhetorical questions, and self-talk are filtered out. Only real questions directed at you trigger AI responses.

---

## Technical Interview Mode

When a coding question is detected, the response includes three sections:

```
┌─────────────────────────────────────────────┐
│ 🏷 TECHNICAL                                │
│                                             │
│ 📌 Key Points                               │
│ ┌─────┐ ┌──────────┐ ┌───────┐             │
│ │ O(n) │ │ hash map │ │ edge  │             │
│ └─────┘ └──────────┘ │ cases │             │
│                       └───────┘             │
│ 🗣 Say This                                 │
│ "I'd use a hash map approach here..."       │
│                                             │
│ 💻 Code                                     │
│ ┌─────────────────────────────────────────┐ │
│ │ def two_sum(nums, target):              │ │
│ │     seen = {}                           │ │
│ │     for i, n in enumerate(nums):        │ │
│ │         comp = target - n               │ │
│ │         if comp in seen:                │ │
│ │             return [seen[comp], i]      │ │
│ │         seen[n] = i                     │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ 🧠 Why This Works                           │
│ "O(n) time, O(n) space. Single pass..."     │
└─────────────────────────────────────────────┘
```

---

## Context Upload

Upload before your interview for personalized answers:

| File | Purpose |
|------|---------|
| `resume.pdf` | Your background, skills, experiences |
| `jd.txt` | Job description — AI tailors answers to the role |
| `interviewer.md` | Interviewer's LinkedIn/background — AI adjusts tone |
| Free text | "Technical SWE interview, focus on system design" |

Supports: `.pdf` `.txt` `.md` `.docx`

---

## Free vs Pro

|  | Free | Pro |
|--|------|-----|
| Real-time transcription | ✅ | ✅ |
| Meeting app detection | ✅ | ✅ |
| Speaking pace monitor | ✅ | ✅ |
| Answer timer | ✅ | ✅ |
| Session recording | ✅ | ✅ |
| AI response generation | ❌ | ✅ |
| Technical code answers | ❌ | ✅ |
| Context upload | ❌ | ✅ |
| Follow-up predictions | ❌ | ✅ |
| Screenshot OCR | ❌ | ✅ |
| Screen monitoring | ❌ | ✅ |
| SOS button | ❌ | ✅ |
| **Price** | **Free** | **$29/year** |

Get a Pro activation code at the [landing page](https://cherishchencherish.github.io/live-assistant/).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser UI                           │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Live Transcript  │  │  Response Cards                  │ │
│  │  (auto-scroll)    │  │  • Verbal / Code / Explain       │ │
│  │                   │  │  • Keywords / Follow-ups          │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  Toolbar: manual input / screenshot / shortcuts          ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────┬───────────────────────────────┘
                              │ WebSocket
┌─────────────────────────────┴───────────────────────────────┐
│                     FastAPI Server                           │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────┐ │
│  │  Audio    │ │  Whisper     │ │ Question │ │  Ollama    │ │
│  │  Capture  │→│  Transcribe  │→│ Detector │→│  Response  │ │
│  │ sounddevice│ │ faster-whisper│ │ 2-stage  │ │ gemma3:4b  │ │
│  └──────────┘ └──────────────┘ └──────────┘ └────────────┘ │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────────────┐  │
│  │ Context      │ │  OCR       │ │  License             │  │
│  │ Loader       │ │ tesseract  │ │  HMAC-SHA256 codes   │  │
│  └──────────────┘ └────────────┘ └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
live-assistant/
├── web_app.py              # FastAPI server (main entry point)
├── templates/index.html    # Web UI
├── context_loader.py       # Load .pdf/.txt/.md/.docx
├── question_detector.py    # 2-stage question detection
├── responder.py            # Ollama prompt engineering
├── license.py              # Activation code system
├── generate_codes.py       # Generate codes for customers
├── ui.py                   # Terminal TUI (alternative)
├── live_assistant.py       # Terminal version (alternative)
├── install.sh              # One-click installer
├── docs/index.html         # GitHub Pages landing page
├── uploads/                # User-uploaded context files
├── sessions/               # Auto-saved transcripts
└── screenshots/            # Captured screenshots
```

---

## Requirements

- **macOS** (uses CoreAudio for device detection)
- **Python 3.10+**
- **Ollama** with `gemma3:4b` model
- **Homebrew** (for portaudio, tesseract)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Cherish Chen](https://github.com/CherishChenCherish)**

*Yale School of Management*

</div>
