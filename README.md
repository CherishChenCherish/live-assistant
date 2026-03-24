# Live Interview / Lecture Assistant

Real-time audio transcription + AI response generation for interviews and lectures.

## Quick Start

```bash
# 1. List audio devices (find BlackHole or your mic)
python3 live_assistant.py --list-devices

# 2. Interview mode (transcribe + auto-generate responses)
export ANTHROPIC_API_KEY='sk-ant-...'
python3 live_assistant.py --device <DEVICE_ID>

# 3. Lecture mode (transcription only)
python3 live_assistant.py --device <DEVICE_ID> --no-respond
```

## Setup System Audio Capture (BlackHole)

To capture what you HEAR (Zoom/Teams/browser audio):

```bash
# Install BlackHole (needs sudo password)
brew install --cask blackhole-2ch
```

Then in macOS **Audio MIDI Setup** (search in Spotlight):
1. Click `+` bottom-left -> "Create Multi-Output Device"
2. Check both your speakers/headphones AND BlackHole 2ch
3. Set this Multi-Output as your system output in System Preferences -> Sound

Now BlackHole captures a copy of your system audio.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | system default | Audio input device ID |
| `--model` | `small` | Whisper model: tiny/base/small/medium/large-v3 |
| `--no-respond` | off | Disable AI response generation |
| `--chunk` | 4 | Seconds per audio chunk |
| `--threshold` | 0.005 | Silence detection threshold |
| `--api-key` | env var | Anthropic API key |
| `--output` | ./sessions | Session save directory |
