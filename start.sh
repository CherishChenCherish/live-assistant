#!/bin/bash
# Live Assistant v2 - Quick Start
cd "$(dirname "$0")"

case "${1:-help}" in
    interview)
        shift
        echo "Starting Interview Mode..."
        python3 live_assistant.py "$@"
        ;;
    lecture)
        shift
        echo "Starting Lecture Mode (transcription only)..."
        python3 live_assistant.py --no-respond "$@"
        ;;
    devices)
        python3 live_assistant.py --list-devices
        ;;
    *)
        echo "Live Assistant v2"
        echo ""
        echo "Usage: ./start.sh <mode> [options]"
        echo ""
        echo "Modes:"
        echo "  interview   Transcribe + generate responses"
        echo "  lecture     Transcription only"
        echo "  devices    List audio devices"
        echo ""
        echo "Examples:"
        echo "  ./start.sh interview --device 4"
        echo "  ./start.sh interview --device 2 --context resume.pdf jd.txt"
        echo "  ./start.sh interview --device 4 --context-text 'PM interview at Google'"
        echo "  ./start.sh lecture --device 4"
        echo ""
        echo "Keys during session:"
        echo "  Space  Pause/Resume"
        echo "  N/->   Skip to next question"
        echo "  P/<-   Previous Q&A"
        echo "  R      Regenerate response"
        echo "  H      Toggle history view"
        echo "  Q      Quit and save"
        ;;
esac
