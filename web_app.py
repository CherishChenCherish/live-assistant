#!/usr/bin/env python3
"""Live Assistant v2 — Web App (FastAPI + WebSocket)."""

import asyncio
import base64
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel

from context_loader import load_context, context_summary
from license import activate, deactivate, get_license_status, has_feature
from question_detector import QuestionDetector
from responder import generate_response, regenerate_response, detect_question_type


def generate_followups(ollama_model: str, question: str, response_verbal: str) -> list[str]:
    """Predict likely follow-up questions the interviewer might ask."""
    prompt = (
        f'Based on this interview Q&A, predict 2-3 likely follow-up questions '
        f'the interviewer might ask next. Output ONLY the questions, one per line.\n\n'
        f'Q: {question}\nA: {response_verbal}\n\nFollow-up questions:'
    )
    try:
        result = subprocess.run(
            ["ollama", "run", ollama_model, prompt],
            capture_output=True, text=True, timeout=20,
        )
        lines = [l.strip().lstrip('0123456789.-) ') for l in result.stdout.strip().split('\n') if l.strip()]
        return lines[:3]
    except Exception:
        return []


def generate_keywords(ollama_model: str, question: str, context: str) -> list[str]:
    """Generate keyword hints instead of full answer."""
    prompt = (
        f'For this interview question, give 5-8 KEY TALKING POINTS as short phrases '
        f'(2-4 words each). No full sentences. One per line.\n\n'
        f'Background: {context[:500]}\n\nQ: {question}\n\nKey points:'
    )
    try:
        result = subprocess.run(
            ["ollama", "run", ollama_model, prompt],
            capture_output=True, text=True, timeout=15,
        )
        lines = [l.strip().lstrip('0123456789.-•) ') for l in result.stdout.strip().split('\n') if l.strip()]
        return lines[:8]
    except Exception:
        return []

app = FastAPI(title="Live Assistant")
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
whisper_model = None
active_session = None


def ocr_image(image_bytes: bytes) -> str:
    """Extract text from image using tesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[OCR Error: {e}]"


def screen_capture() -> bytes | None:
    """Capture screen on macOS using screencapture."""
    tmp = "/tmp/live_assistant_screen.png"
    try:
        subprocess.run(["screencapture", "-x", tmp], timeout=5, check=True)
        with open(tmp, "rb") as f:
            return f.read()
    except Exception:
        return None


class Session:
    def __init__(self, device_id, context_materials, context_label, ollama_model, whisper_size):
        self.device_id = device_id
        self.context_materials = context_materials
        self.context_label = context_label
        self.ollama_model = ollama_model

        self.transcript = []
        self.responses = []
        self.lock = threading.Lock()

        self.audio_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.question_detector = QuestionDetector(ollama_model=ollama_model)
        self.ws_queue = asyncio.Queue()

        self.screen_monitor_active = False
        self._capture_thread = None
        self._transcribe_thread = None
        self._screen_thread = None

    def start(self):
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._capture_thread.start()
        self._transcribe_thread.start()

    def stop(self):
        self.stop_event.set()
        self.screen_monitor_active = False
        self._save_session()

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            return False
        else:
            self.pause_event.set()
            return True

    def toggle_screen_monitor(self):
        self.screen_monitor_active = not self.screen_monitor_active
        if self.screen_monitor_active and (self._screen_thread is None or not self._screen_thread.is_alive()):
            self._screen_thread = threading.Thread(target=self._screen_monitor_loop, daemon=True)
            self._screen_thread.start()
        return self.screen_monitor_active

    def process_screenshot(self, image_bytes: bytes):
        """OCR a screenshot and treat extracted text as a question."""
        text = ocr_image(image_bytes)
        if text and len(text) > 10:
            # Save screenshot
            ts = datetime.now().strftime("%H%M%S")
            (SCREENSHOTS_DIR / f"screen_{ts}.png").write_bytes(image_bytes)

            self._send({"type": "ocr_result", "text": text})

            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": f"[SCREENSHOT] {text[:200]}",
                "is_question": True,
            }
            with self.lock:
                self.transcript.append(entry)
            self._send({"type": "transcript", **entry})

            # Generate response for the extracted question
            self._handle_question(text)

    def _screen_monitor_loop(self):
        """Periodically capture screen and OCR for new content."""
        last_text = ""
        while not self.stop_event.is_set() and self.screen_monitor_active:
            img_bytes = screen_capture()
            if img_bytes:
                text = ocr_image(img_bytes)
                # Only process if content changed significantly
                if text and len(text) > 20 and text != last_text:
                    similarity = _text_similarity(text, last_text)
                    if similarity < 0.8:  # content changed
                        last_text = text
                        self._send({"type": "screen_update", "text": text[:500]})
                        # Check if it looks like a new question/problem
                        from question_detector import fast_filter
                        if fast_filter(text):
                            self._handle_question(text)
            time.sleep(5)  # check every 5 seconds

    def _send(self, msg: dict):
        try:
            self.ws_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def _capture_loop(self):
        chunk_samples = int(SAMPLE_RATE * 4)
        buffer = np.zeros(0, dtype=np.float32)

        def callback(indata, frames, time_info, status):
            nonlocal buffer
            audio = indata[:, 0].copy()
            buffer = np.concatenate([buffer, audio])
            while len(buffer) >= chunk_samples:
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]
                rms = np.sqrt(np.mean(chunk**2))
                if rms > 0.003:
                    self.audio_queue.put(("audio", chunk))
                else:
                    self.audio_queue.put(("silence", None))

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                device=self.device_id, callback=callback,
                blocksize=int(SAMPLE_RATE * 0.5),
            ):
                while not self.stop_event.is_set():
                    self.stop_event.wait(0.1)
        except Exception as e:
            self._send({"type": "error", "message": f"Audio error: {e}"})

    def _transcribe_loop(self):
        global whisper_model
        silence_count = 0
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.2)
                continue
            try:
                msg_type, data = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                q = self.question_detector.flush()
                if q:
                    self._handle_question(q)
                continue

            if msg_type == "silence":
                silence_count += 1
                if silence_count >= 2:
                    q = self.question_detector.flush()
                    if q:
                        self._handle_question(q)
                    silence_count = 0
                continue

            silence_count = 0
            try:
                segments, _ = whisper_model.transcribe(
                    data, beam_size=3, language=None,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                if not text or len(text) < 3:
                    continue
                entry = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "text": text,
                    "is_question": False,
                }
                with self.lock:
                    self.transcript.append(entry)
                self._send({"type": "transcript", **entry})
                q = self.question_detector.feed(text)
                if q:
                    self._handle_question(q)
            except Exception as e:
                self._send({"type": "error", "message": f"Transcription error: {e}"})

    def _handle_question(self, question):
        # Truncate overly long questions (max ~50 words)
        words = question.split()
        if len(words) > 50:
            question = " ".join(words[:50])

        q_entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": f"[QUESTION] {question}",
            "is_question": True,
        }
        with self.lock:
            self.transcript.append(q_entry)
        self._send({"type": "transcript", **q_entry})

        # Check license — free tier only gets transcription
        if not has_feature("ai_responses"):
            self._send({"type": "paywall", "question": question})
            return

        q_type = detect_question_type(question)
        self._send({"type": "status", "state": "generating", "question_type": q_type})

        with self.lock:
            recent = list(self.transcript[-10:])
            prev = list(self.responses)

        result = generate_response(
            ollama_model=self.ollama_model,
            question=question,
            conversation_lines=recent,
            context_materials=self.context_materials,
            previous_responses=prev,
        )

        # Generate follow-up predictions & keyword hints in parallel
        followups = []
        keywords = []

        def _get_followups():
            nonlocal followups
            followups = generate_followups(self.ollama_model, question, result.get("verbal", ""))

        def _get_keywords():
            nonlocal keywords
            keywords = generate_keywords(self.ollama_model, question, self.context_materials)

        t1 = threading.Thread(target=_get_followups)
        t2 = threading.Thread(target=_get_keywords)
        t1.start(); t2.start()
        t1.join(timeout=20); t2.join(timeout=15)

        r_entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            **result,
            "followups": followups,
            "keywords": keywords,
        }
        with self.lock:
            self.responses.append(r_entry)
        self._send({"type": "response", **r_entry})
        self._send({"type": "status", "state": "listening"})

    def do_regenerate(self, index=-1):
        with self.lock:
            if not self.responses:
                return
            r = self.responses[index]
            recent = list(self.transcript[-10:])
            prev = list(self.responses)

        self._send({"type": "status", "state": "generating"})

        result = regenerate_response(
            ollama_model=self.ollama_model,
            question=r["question"],
            conversation_lines=recent,
            context_materials=self.context_materials,
            previous_responses=prev,
            previous_attempt=r.get("verbal", ""),
        )

        with self.lock:
            self.responses[index].update(result)

        self._send({
            "type": "regenerated",
            "index": index,
            "question": r["question"],
            **result,
        })
        self._send({"type": "status", "state": "listening"})

    def _save_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with self.lock:
            if self.transcript:
                with open(SESSIONS_DIR / f"transcript_{ts}.txt", "w") as f:
                    for line in self.transcript:
                        marker = " [Q]" if line["is_question"] else ""
                        f.write(f"[{line['time']}]{marker} {line['text']}\n")
            if self.responses:
                with open(SESSIONS_DIR / f"responses_{ts}.json", "w") as f:
                    json.dump(self.responses, f, indent=2, ensure_ascii=False)


def _text_similarity(a: str, b: str) -> float:
    """Simple word overlap similarity."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "templates" / "index.html").read_text()


def detect_active_meeting_app() -> str | None:
    """Detect which meeting app is currently running on macOS."""
    try:
        r = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get name of every process whose background only is false'],
            capture_output=True, text=True, timeout=5,
        )
        apps = r.stdout.lower()
        if "zoom" in apps:
            return "Zoom"
        if "teams" in apps or "microsoft teams" in apps:
            return "Teams"
        if "wemeet" in apps or "voov" in apps:
            return "WeMeet"
        if "webex" in apps:
            return "Webex"
        if "slack" in apps:
            return "Slack"
        if "google meet" in apps or "chrome" in apps:
            return "Google Meet (Chrome)"
        if "facetime" in apps:
            return "FaceTime"
    except Exception:
        pass
    return None


@app.get("/api/devices")
async def get_devices():
    devices = sd.query_devices()
    active_app = detect_active_meeting_app()

    result = []
    recommended = None
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            tag = ""
            name = d["name"]
            name_lower = name.lower()
            if "blackhole" in name_lower:
                tag = "BlackHole"
            elif "teams" in name_lower:
                tag = "Teams"
                if active_app and "teams" in active_app.lower():
                    recommended = i
            elif "wemeet" in name_lower:
                tag = "WeMeet"
                if active_app and "wemeet" in active_app.lower():
                    recommended = i
            elif "zoom" in name_lower:
                tag = "Zoom"
                if active_app and "zoom" in active_app.lower():
                    recommended = i
            elif i == sd.default.device[0]:
                tag = "Default"
            result.append({"id": i, "name": name, "tag": tag})

    return {
        "devices": result,
        "active_meeting_app": active_app,
        "recommended_device": recommended,
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "path": str(dest)}


@app.get("/api/uploads")
async def list_uploads():
    return [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]


@app.delete("/api/uploads/{filename}")
async def delete_upload(filename: str):
    path = UPLOAD_DIR / filename
    if path.exists():
        path.unlink()
    return {"ok": True}


@app.post("/api/screenshot")
async def upload_screenshot(file: UploadFile):
    """Handle screenshot upload, OCR it, return text."""
    data = await file.read()
    text = ocr_image(data)
    # Save
    ts = datetime.now().strftime("%H%M%S")
    (SCREENSHOTS_DIR / f"upload_{ts}.png").write_bytes(data)
    # If session active, process as question
    if active_session:
        active_session.process_screenshot(data)
    return {"text": text}


@app.get("/api/license")
async def get_license():
    return get_license_status()


@app.post("/api/license/activate")
async def activate_license(data: dict):
    code = data.get("code", "")
    result = activate(code)
    return result


@app.post("/api/license/deactivate")
async def deactivate_license():
    deactivate()
    return {"ok": True}


SOS_PHRASES = [
    "That's a great question — let me think about the best way to frame this.",
    "I want to make sure I give you a thorough answer. Let me organize my thoughts for a moment.",
    "There are a few angles I could approach this from. Let me take a second to pick the most relevant one.",
    "That's something I've thought about quite a bit. Let me share the most impactful example.",
    "I appreciate that question. To give you the most useful answer, let me think about which experience best illustrates this.",
]


@app.get("/api/sos")
async def get_sos():
    if not has_feature("sos_button"):
        return {"error": "pro_only", "phrase": "Upgrade to Pro for SOS phrases"}
    import random
    return {"phrase": random.choice(SOS_PHRASES)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_session, whisper_model
    await ws.accept()

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "start":
                device_id = msg.get("device")
                context_files = msg.get("context_files", [])
                context_text = msg.get("context_text", "")
                ollama_model = msg.get("llm", "gemma3:4b")
                whisper_size = msg.get("whisper", "small")

                if whisper_model is None:
                    await ws.send_json({"type": "status", "state": "loading_model"})
                    whisper_model = WhisperModel(whisper_size, device="cpu", compute_type="int8")

                file_paths = [str(UPLOAD_DIR / f) for f in context_files if (UPLOAD_DIR / f).exists()]
                text_snippets = [context_text] if context_text else None
                ctx = load_context(file_paths or None, text_snippets)
                ctx_label = context_summary(file_paths or None, text_snippets)

                if active_session:
                    active_session.stop()

                active_session = Session(device_id, ctx, ctx_label, ollama_model, whisper_size)
                active_session.start()
                await ws.send_json({"type": "status", "state": "listening"})
                await ws.send_json({"type": "context", "label": ctx_label})
                asyncio.create_task(_forward_messages(ws, active_session))

            elif action == "stop":
                if active_session:
                    active_session.stop()
                    active_session = None
                await ws.send_json({"type": "status", "state": "stopped"})

            elif action == "pause":
                if active_session:
                    paused = active_session.toggle_pause()
                    await ws.send_json({"type": "status", "state": "paused" if paused else "listening"})

            elif action == "regenerate":
                if active_session:
                    idx = msg.get("index", -1)
                    threading.Thread(target=active_session.do_regenerate, args=(idx,), daemon=True).start()

            elif action == "screenshot":
                # Base64 encoded screenshot from clipboard paste
                img_data = base64.b64decode(msg.get("data", ""))
                if active_session and img_data:
                    threading.Thread(target=active_session.process_screenshot, args=(img_data,), daemon=True).start()

            elif action == "screen_monitor":
                if active_session:
                    is_on = active_session.toggle_screen_monitor()
                    await ws.send_json({"type": "screen_monitor", "active": is_on})

            elif action == "manual_question":
                # User typed a question manually
                question = msg.get("text", "")
                if active_session and question:
                    threading.Thread(target=active_session._handle_question, args=(question,), daemon=True).start()

    except WebSocketDisconnect:
        if active_session:
            active_session.stop()
            active_session = None


async def _forward_messages(ws: WebSocket, session: Session):
    while not session.stop_event.is_set():
        try:
            msg = await asyncio.wait_for(session.ws_queue.get(), timeout=0.5)
            await ws.send_json(msg)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


if __name__ == "__main__":
    print("\n  Live Assistant — http://localhost:8765\n")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
