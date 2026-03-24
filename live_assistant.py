#!/usr/bin/env python3
"""
Live Interview / Lecture Assistant v2
- Real-time audio transcription (faster-whisper, local)
- Smart question detection (2-stage: heuristic + LLM filter)
- Context-aware response generation (Ollama, local & free)
- Interactive TUI with keyboard controls
- Background materials support (resume, JD, etc.)
"""

import argparse
import json
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rich.live import Live

from context_loader import context_summary, load_context
from question_detector import QuestionDetector
from responder import generate_response, regenerate_response
from ui import AppState, KeyReader, LiveUI

# --- Audio Config ---
SAMPLE_RATE = 16000
CHANNELS = 1


class TranscriptBuffer:
    def __init__(self, max_lines=100):
        self.lines = []
        self.max_lines = max_lines
        self.lock = threading.Lock()

    def add(self, text, is_question=False):
        with self.lock:
            self.lines.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": text,
                "is_question": is_question,
            })
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]

    def get_recent(self, n=15):
        with self.lock:
            return list(self.lines[-n:])

    def get_all(self):
        with self.lock:
            return list(self.lines)


class ResponseBuffer:
    def __init__(self):
        self.responses = []
        self.lock = threading.Lock()
        self.current_index = -1  # -1 = latest

    def add(self, question, response):
        with self.lock:
            self.responses.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "response": response,
            })
            self.current_index = -1

    def update_latest(self, response):
        with self.lock:
            if self.responses:
                self.responses[-1]["response"] = response

    def get_all(self):
        with self.lock:
            return list(self.responses)

    def get_current(self):
        with self.lock:
            if not self.responses:
                return None
            return self.responses[self.current_index]

    def navigate(self, direction):
        """direction: -1 for prev, +1 for next"""
        with self.lock:
            if not self.responses:
                return
            if direction == -1:
                if abs(self.current_index) < len(self.responses):
                    self.current_index -= 1
            elif direction == 1:
                if self.current_index < -1:
                    self.current_index += 1


def list_audio_devices():
    from rich.console import Console
    console = Console()
    console.print("\n[bold]Available audio input devices:[/bold]\n")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            name = d["name"]
            marker = ""
            if "blackhole" in name.lower():
                marker = " [yellow]<-- BlackHole[/yellow]"
            elif any(k in name.lower() for k in ("teams", "wemeet", "zoom")):
                marker = " [cyan]<-- Meeting App[/cyan]"
            elif i == sd.default.device[0]:
                marker = " [green]<-- Default Mic[/green]"
            console.print(f"  [{i}] {name} (in: {d['max_input_channels']}ch){marker}")
    console.print()


def audio_capture_thread(audio_queue, device_id, stop_event, chunk_duration, silence_threshold):
    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    buffer = np.zeros(0, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        audio = indata[:, 0].copy()
        buffer = np.concatenate([buffer, audio])

        while len(buffer) >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            rms = np.sqrt(np.mean(chunk**2))
            if rms > silence_threshold:
                audio_queue.put(("audio", chunk))
            else:
                audio_queue.put(("silence", None))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=device_id,
            callback=callback,
            blocksize=int(SAMPLE_RATE * 0.5),
        ):
            while not stop_event.is_set():
                stop_event.wait(0.1)
    except Exception as e:
        print(f"Audio capture error: {e}", file=sys.stderr)


def transcription_thread(
    audio_queue, transcript, response_buf, whisper_model,
    question_detector, ollama_model, context_materials,
    stop_event, pause_event, ui,
):
    """Transcribe audio, detect questions, generate responses."""
    silence_count = 0

    while not stop_event.is_set():
        # Check pause
        if pause_event.is_set():
            time.sleep(0.2)
            continue

        try:
            msg_type, data = audio_queue.get(timeout=1.0)
        except queue.Empty:
            # On idle, flush the question detector buffer
            question = question_detector.flush()
            if question:
                _handle_question(question, transcript, response_buf, ollama_model,
                                 context_materials, ui)
            continue

        if msg_type == "silence":
            silence_count += 1
            # After enough silence, flush detector
            if silence_count >= 2:
                question = question_detector.flush()
                if question:
                    _handle_question(question, transcript, response_buf, ollama_model,
                                     context_materials, ui)
                silence_count = 0
            continue

        silence_count = 0
        chunk = data

        try:
            segments, _ = whisper_model.transcribe(
                chunk, beam_size=3, language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            )
            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts).strip()
            if not full_text or len(full_text) < 3:
                continue

            # Add to transcript (not yet marked as question)
            transcript.add(full_text, is_question=False)

            # Feed to question detector
            question = question_detector.feed(full_text)
            if question:
                _handle_question(question, transcript, response_buf, ollama_model,
                                 context_materials, ui)

        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)


def _handle_question(question, transcript, response_buf, ollama_model,
                     context_materials, ui):
    """Mark question in transcript, generate response."""
    # Mark it in transcript
    transcript.add(f"[QUESTION] {question}", is_question=True)

    # Update UI state
    ui.state = AppState.GENERATING

    # Generate response
    response = generate_response(
        ollama_model=ollama_model,
        question=question,
        conversation_lines=transcript.get_recent(10),
        context_materials=context_materials,
        previous_responses=response_buf.get_all(),
    )

    response_buf.add(question, response)
    ui.state = AppState.SHOWING_RESPONSE


def save_session(transcript, response_buf, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / f"transcript_{timestamp}.txt", "w") as f:
        for line in transcript.get_all():
            marker = " [Q]" if line["is_question"] else ""
            f.write(f"[{line['time']}]{marker} {line['text']}\n")

    responses = response_buf.get_all()
    if responses:
        with open(out / f"responses_{timestamp}.json", "w") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Live Interview/Lecture Assistant v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --device 4                              # Use Teams Audio
  %(prog)s --device 2 --context resume.pdf jd.txt  # Load background files
  %(prog)s --device 2 --context-text "PM interview at Google"
  %(prog)s --list-devices                           # Show audio devices
        """,
    )
    parser.add_argument("--device", type=int, default=None, help="Audio input device ID")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument(
        "--whisper", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--llm", default="gemma3:4b",
        help="Ollama model (default: gemma3:4b)",
    )
    parser.add_argument(
        "--context", nargs="+", default=None,
        help="Background files: resume, JD, etc. (.txt .pdf .md .docx)",
    )
    parser.add_argument(
        "--context-text", action="append", default=None,
        help="Additional context as text (can use multiple times)",
    )
    parser.add_argument("--no-respond", action="store_true", help="Transcription only")
    parser.add_argument("--output", default="./sessions", help="Session save directory")
    parser.add_argument("--chunk", type=int, default=4, help="Audio chunk seconds (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.003, help="Silence threshold (default: 0.003)")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    from rich.console import Console
    console = Console()

    # Load context materials
    context_materials = ""
    ctx_summary = "None"
    if args.context or args.context_text:
        console.print("[cyan]Loading context materials...[/cyan]")
        context_materials = load_context(args.context, args.context_text)
        ctx_summary = context_summary(args.context, args.context_text)
        console.print(f"[green]Loaded: {ctx_summary}[/green]")

    auto_respond = not args.no_respond

    # Check Ollama
    if auto_respond:
        import subprocess
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if args.llm.split(":")[0] not in r.stdout:
                console.print(f"[yellow]Pulling model '{args.llm}'...[/yellow]")
                subprocess.run(["ollama", "pull", args.llm], timeout=300)
        except FileNotFoundError:
            console.print("[red]Ollama not found! Install from https://ollama.com[/red]")
            return
        except Exception as e:
            console.print(f"[yellow]Ollama warning: {e}[/yellow]")

    # Load Whisper
    console.print(f"[cyan]Loading Whisper '{args.whisper}' model...[/cyan]")
    whisper_model = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    console.print("[green]Model loaded![/green]")

    # Device info
    if args.device is not None:
        dev_info = sd.query_devices(args.device)
        console.print(f"[cyan]Audio: [{args.device}] {dev_info['name']}[/cyan]")
    else:
        list_audio_devices()
        console.print("[yellow]No --device specified, using default mic.[/yellow]")

    # Initialize
    audio_queue = queue.Queue(maxsize=30)
    transcript = TranscriptBuffer()
    response_buf = ResponseBuffer()
    stop_event = threading.Event()
    pause_event = threading.Event()
    question_detector = QuestionDetector(ollama_model=args.llm)

    ui = LiveUI(context_summary=ctx_summary)
    if not auto_respond:
        ui.state = AppState.LISTENING

    # Start threads
    capture_t = threading.Thread(
        target=audio_capture_thread,
        args=(audio_queue, args.device, stop_event, args.chunk, args.threshold),
        daemon=True,
    )

    transcribe_t = threading.Thread(
        target=transcription_thread,
        args=(audio_queue, transcript, response_buf, whisper_model,
              question_detector, args.llm, context_materials,
              stop_event, pause_event, ui),
        daemon=True,
    )

    capture_t.start()
    transcribe_t.start()

    # Keyboard reader
    key_reader = KeyReader()
    key_reader.start()

    mode = f"[green]Interview Mode (LLM: {args.llm})[/green]" if auto_respond else "[yellow]Lecture Mode[/yellow]"
    console.print(f"\n{mode}")
    console.print("[dim]Keys: [Space] Pause  [N] Next  [P] Prev  [R] Regen  [H] History  [Q] Quit[/dim]\n")

    # Main loop
    try:
        with Live(
            ui.build_layout(transcript.get_recent(), response_buf.get_all()),
            refresh_per_second=2,
            console=console,
        ) as live:
            while not stop_event.is_set():
                # Handle keyboard
                key = key_reader.get_key()
                if key:
                    action = ui.handle_key(key)

                    if action == "quit":
                        stop_event.set()
                        break
                    elif action == "pause":
                        pause_event.set()
                    elif action == "resume":
                        pause_event.clear()
                    elif action == "next":
                        pass  # Just switch state back to listening
                    elif action == "prev":
                        response_buf.navigate(-1)
                    elif action == "regenerate":
                        current = response_buf.get_current()
                        if current and auto_respond:
                            ui.state = AppState.GENERATING
                            # Regenerate in background
                            def _regen(cur=current):
                                new_resp = regenerate_response(
                                    ollama_model=args.llm,
                                    question=cur["question"],
                                    conversation_lines=transcript.get_recent(10),
                                    context_materials=context_materials,
                                    previous_responses=response_buf.get_all(),
                                    previous_attempt=cur["response"],
                                )
                                response_buf.update_latest(new_resp)
                                ui.state = AppState.SHOWING_RESPONSE
                            threading.Thread(target=_regen, daemon=True).start()

                # Update display
                live.update(ui.build_layout(
                    transcript.get_recent(),
                    response_buf.get_all(),
                    response_buf.current_index,
                ))
                time.sleep(0.3)

    except KeyboardInterrupt:
        stop_event.set()

    key_reader.stop()
    out = save_session(transcript, response_buf, args.output)
    console.print(f"\n[green]Session saved to {out}/[/green]")


if __name__ == "__main__":
    main()
