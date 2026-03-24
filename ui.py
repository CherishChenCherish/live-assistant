"""Interactive TUI for Live Assistant with keyboard controls."""

import sys
import termios
import threading
import tty
from enum import Enum

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class AppState(Enum):
    LISTENING = "listening"
    PAUSED = "paused"
    GENERATING = "generating"
    SHOWING_RESPONSE = "showing_response"
    HISTORY = "history"


class KeyReader:
    """Non-blocking keyboard reader for macOS/Linux."""

    def __init__(self):
        self._stop = threading.Event()
        self._key_queue = []
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_key(self) -> str | None:
        with self._lock:
            if self._key_queue:
                return self._key_queue.pop(0)
        return None

    def _read_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop.is_set():
                # Use select to check if input is available
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":  # Escape sequence (arrow keys)
                        ch2 = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.05)[0] else ""
                        ch3 = sys.stdin.read(1) if ch2 and select.select([sys.stdin], [], [], 0.05)[0] else ""
                        if ch2 == "[":
                            if ch3 == "C":  # Right arrow
                                with self._lock:
                                    self._key_queue.append("right")
                            elif ch3 == "D":  # Left arrow
                                with self._lock:
                                    self._key_queue.append("left")
                        continue
                    with self._lock:
                        self._key_queue.append(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class LiveUI:
    """Interactive TUI manager."""

    def __init__(self, context_summary: str = "None"):
        self.console = Console()
        self.state = AppState.LISTENING
        self.context_summary = context_summary
        self.history_index = -1  # -1 = latest
        self.key_reader = KeyReader()

    def build_layout(self, transcript_lines, responses, current_response_idx=-1):
        """Build the full TUI layout."""
        layout = Layout()

        # Header
        header = self._build_header()

        # Transcript panel
        transcript_panel = self._build_transcript(transcript_lines)

        # Response panel (or history view)
        if self.state == AppState.HISTORY:
            response_panel = self._build_history(responses)
        else:
            response_panel = self._build_response(responses, current_response_idx)

        # Status bar
        status_bar = self._build_status_bar()

        layout.split_column(
            Layout(header, name="header", size=3),
            Layout(transcript_panel, name="transcript", ratio=2),
            Layout(response_panel, name="response", ratio=2),
            Layout(status_bar, name="status", size=3),
        )
        return layout

    def _build_header(self) -> Panel:
        header_text = Text()
        header_text.append("LIVE ASSISTANT", style="bold white")
        header_text.append("  |  ", style="dim")
        header_text.append(f"State: {self.state.value.upper()}", style="bold cyan")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Context: {self.context_summary}", style="yellow")
        return Panel(header_text, style="blue")

    def _build_transcript(self, lines: list[dict]) -> Panel:
        text = Text()
        display_lines = lines[-18:] if lines else []

        if not display_lines:
            text.append("Waiting for audio...", style="dim italic")
        else:
            for line in display_lines:
                time_str = f"[{line['time']}] "
                if line.get("is_question"):
                    text.append(time_str, style="dim")
                    text.append(f">> {line['text']}\n", style="bold yellow")
                else:
                    text.append(time_str, style="dim")
                    text.append(f"{line['text']}\n", style="white")

        return Panel(
            text,
            title="[bold cyan]Live Transcript[/bold cyan]",
            border_style="cyan",
        )

    def _build_response(self, responses: list[dict], idx: int = -1) -> Panel:
        text = Text()

        if not responses:
            if self.state == AppState.GENERATING:
                text.append("Generating response...", style="bold yellow italic")
            else:
                text.append("Waiting for questions...\n", style="dim italic")
                text.append(
                    "Only real interview questions will trigger responses.",
                    style="dim",
                )
            return Panel(
                text,
                title="[bold green]Suggested Response[/bold green]",
                border_style="green",
            )

        # Show the response at given index
        r = responses[idx] if abs(idx) <= len(responses) else responses[-1]

        text.append(f"[{r['time']}] ", style="dim")
        text.append("Q: ", style="bold yellow")
        text.append(f"{r['question']}\n\n", style="yellow")
        text.append("SAY THIS:\n", style="bold green")
        text.append(f"{r['response']}", style="bold white")

        nav_info = ""
        if len(responses) > 1:
            actual_idx = idx if idx >= 0 else len(responses) + idx
            nav_info = f" ({actual_idx + 1}/{len(responses)})"

        return Panel(
            text,
            title=f"[bold green]Suggested Response{nav_info}[/bold green]",
            border_style="green",
        )

    def _build_history(self, responses: list[dict]) -> Panel:
        text = Text()

        if not responses:
            text.append("No responses yet.", style="dim italic")
        else:
            for i, r in enumerate(responses):
                is_selected = (i == self.history_index) or (
                    self.history_index == -1 and i == len(responses) - 1
                )
                prefix = ">> " if is_selected else "   "
                style = "bold" if is_selected else "dim"

                text.append(f"{prefix}[{r['time']}] ", style="dim")
                text.append(f"Q: {r['question'][:60]}", style=f"yellow {style}")
                text.append("\n")

                if is_selected:
                    text.append(f"      SAY: {r['response']}\n\n", style="bold white")

        return Panel(
            text,
            title="[bold magenta]Q&A History (arrow keys to navigate)[/bold magenta]",
            border_style="magenta",
        )

    def _build_status_bar(self) -> Panel:
        text = Text()
        keys = [
            ("Space", "Pause/Resume"),
            ("N/->", "Next"),
            ("P/<-", "Prev"),
            ("R", "Regenerate"),
            ("H", "History"),
            ("Q", "Quit"),
        ]
        for i, (key, desc) in enumerate(keys):
            if i > 0:
                text.append("  |  ", style="dim")
            text.append(f"[{key}]", style="bold cyan")
            text.append(f" {desc}", style="white")

        return Panel(text, style="dim")

    def handle_key(self, key: str) -> str | None:
        """Process a keypress. Returns action string or None.

        Actions: 'quit', 'pause', 'resume', 'next', 'prev', 'regenerate', 'history_toggle'
        """
        if key in ("q", "Q", "\x03"):  # q or Ctrl+C
            return "quit"

        if key == " ":
            if self.state == AppState.PAUSED:
                self.state = AppState.LISTENING
                return "resume"
            elif self.state in (AppState.LISTENING, AppState.SHOWING_RESPONSE):
                self.state = AppState.PAUSED
                return "pause"

        if key in ("n", "right"):
            if self.state == AppState.HISTORY:
                self.history_index = min(self.history_index + 1, -1) if self.history_index < -1 else -1
                return None
            self.state = AppState.LISTENING
            return "next"

        if key in ("p", "left"):
            if self.state == AppState.HISTORY:
                self.history_index -= 1
                return None
            return "prev"

        if key in ("r", "R"):
            return "regenerate"

        if key in ("h", "H"):
            if self.state == AppState.HISTORY:
                self.state = AppState.SHOWING_RESPONSE
                return "history_toggle"
            else:
                self.state = AppState.HISTORY
                self.history_index = -1
                return "history_toggle"

        return None
