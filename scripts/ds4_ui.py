#!/usr/bin/env python3
"""DS4 terminal UI with live DFlash counters and runtime fallbacks.

Pure Python standard library: no external TUI dependency.
"""

from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(
    os.environ.get("DS4_CONTINUAL_ROOT", str(Path.home() / "ds4_lab/continual"))
).expanduser()
STATE = ROOT / "state"
BUFFER_INDEX = ROOT / "buffer/index.jsonl"
RUNTIME_PIDS = STATE / "runtime"
LAUNCHER = Path(
    os.environ.get("DS4_CONTINUAL_LAUNCHER", str(Path.home() / ".local/bin/ds4-continual"))
).expanduser()

LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in English unless the user explicitly requests another language.",
    "it": "Rispondi in italiano salvo richiesta esplicita dell'utente di usare un'altra lingua.",
}
LANGUAGE_ALIASES = {
    "en": "en", "english": "en",
    "it": "it", "italian": "it", "italiano": "it",
}
UI_TEXT = {
    "en": {
        "ready": "Ready",
        "startup": "DS4 UI ready. Change DFlash with /dflash on, /dflash off, or /dflash auto.",
        "fallback_startup": "Runtime fallbacks enabled: DFlash to normal generation; proxy to direct server; one restart after a server crash.",
        "busy": "A request is already running",
        "empty_answer": "[empty response]",
        "thinking_title": "THINKING",
        "thinking_empty": "No thinking stream received",
        "unknown": "Unknown command: {command}. Use /help",
        "help": "/dflash on  /dflash off  /dflash auto\n/language  /language en  /language it  /reset\n/fallback on|off  /recover on|off|now  /stats  /status\n/buffer  /training  /clearstats  /quit",
        "language_status": "Current language: {language}. Choices: en, it. Usage: /language en|it",
        "language_set": "Language set to English.",
        "language_invalid": "Unsupported language. Use /language en or /language it.",
        "dflash_status": "DFlash configured={configured}; effective={effective}",
        "dflash_set": "DFlash set to {configured}; effective={effective}",
        "dflash_usage": "Usage: /dflash on|off|auto",
        "fallback": "DFlash fallback to normal generation: {state}",
        "recover": "Runtime auto-recovery: {state}",
        "recover_busy": "Cannot recover while a request is running",
        "recover_usage": "Auto-recovery={state}; usage: /recover on|off|now",
        "reset": "Conversation reset",
        "cleared": "Counters reset",
        "buffer": "Replay buffer: {count} samples",
        "no_training": "No continual training recorded",
        "recovered": "Runtime recovered",
        "recovery_failed": "Recovery failed",
        "request_failed": "Request failed after recovery: {error}",
        "non_json": "non-JSON response: {text}",
        "non_object": "JSON response is not an object",
        "you": "YOU",
        "footer": "PgUp/PgDn scroll | Ctrl-C exits | Enter sends",
        "stats_turns": "turns",
        "stats_normal": "normal",
    },
    "it": {
        "ready": "Pronto",
        "startup": "UI DS4 pronta. Cambia DFlash con /dflash on, /dflash off o /dflash auto.",
        "fallback_startup": "Fallback runtime attivi: DFlash verso generazione normale; proxy verso server diretto; un riavvio dopo un crash server.",
        "busy": "Una richiesta è già in corso",
        "empty_answer": "[risposta vuota]",
        "thinking_title": "RAGIONAMENTO",
        "thinking_empty": "Nessun flusso di ragionamento ricevuto",
        "unknown": "Comando sconosciuto: {command}. Usa /help",
        "help": "/dflash on  /dflash off  /dflash auto\n/language  /language en  /language it  /reset\n/fallback on|off  /recover on|off|now  /stats  /status\n/buffer  /training  /clearstats  /quit",
        "language_status": "Lingua corrente: {language}. Scelte: en, it. Uso: /language en|it",
        "language_set": "Lingua impostata su italiano.",
        "language_invalid": "Lingua non supportata. Usa /language en o /language it.",
        "dflash_status": "DFlash configurato={configured}; effettivo={effective}",
        "dflash_set": "DFlash impostato su {configured}; effettivo={effective}",
        "dflash_usage": "Uso: /dflash on|off|auto",
        "fallback": "Fallback DFlash verso generazione normale: {state}",
        "recover": "Auto-recovery runtime: {state}",
        "recover_busy": "Impossibile recuperare durante una richiesta",
        "recover_usage": "Auto-recovery={state}; uso: /recover on|off|now",
        "reset": "Conversazione azzerata",
        "cleared": "Counter azzerati",
        "buffer": "Replay buffer: {count} campioni",
        "no_training": "Nessun training continual registrato",
        "recovered": "Runtime recuperato",
        "recovery_failed": "Recovery fallita",
        "request_failed": "Richiesta fallita dopo recovery: {error}",
        "non_json": "risposta non JSON: {text}",
        "non_object": "risposta JSON non-oggetto",
        "you": "TU",
        "footer": "PgUp/PgDn scorre | Ctrl-C esce | Enter invia",
        "stats_turns": "turni",
        "stats_normal": "normali",
    },
}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def normalize_language(value: str) -> str | None:
    return LANGUAGE_ALIASES.get(value.strip().lower())


def ui_text(locale: str, key: str, **values: Any) -> str:
    return UI_TEXT[locale][key].format(**values)


def build_request_messages(
    history: list[dict[str, str]], language: str
) -> list[dict[str, str]]:
    instructions = set(LANGUAGE_INSTRUCTIONS.values())
    cleaned = [
        dict(message) for message in history
        if not (message.get("role") == "system" and message.get("content") in instructions)
    ]
    insert_at = 1 if cleaned and cleaned[0].get("role") == "system" else 0
    cleaned.insert(insert_at, {"role": "system", "content": LANGUAGE_INSTRUCTIONS[language]})
    return cleaned


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def prompt_metrics(response: dict[str, Any]) -> tuple[int, float | None]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    runtime = response.get("ds4_runtime") if isinstance(response.get("ds4_runtime"), dict) else {}
    prompt_value = runtime.get("prompt_token_count", usage.get("prompt_tokens", 0))
    prompt_number = _finite_number(prompt_value)
    prompt_tokens = int(prompt_number) if prompt_number is not None and prompt_number >= 0 else 0
    timing = response.get("timing_ms") if isinstance(response.get("timing_ms"), dict) else {}
    milliseconds = _finite_number(
        runtime.get("prompt_processing_ms", response.get("prompt_processing_ms", timing.get("prompt_processing_ms")))
    )
    if milliseconds is None or milliseconds <= 0:
        milliseconds = None
    return prompt_tokens, milliseconds


def format_prefill_metric(prompt_tokens: Any, prompt_processing_ms: Any) -> str:
    token_number = _finite_number(prompt_tokens)
    tokens = int(token_number) if token_number is not None and token_number >= 0 else 0
    milliseconds = _finite_number(prompt_processing_ms)
    if milliseconds is None or milliseconds <= 0:
        return f"PREFILL {tokens} tok N/A"
    return f"PREFILL {tokens} tok {tokens / (milliseconds / 1000.0):.2f} tok/s"


@dataclass
class AssistantContent:
    thinking: str = ""
    final: str = ""


class ThinkingParser:
    """Small streaming state machine that never exposes raw thinking tags."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.state = "unknown"
        self.carry = ""
        self.tentative: list[str] = []
        self.thinking: list[str] = []
        self.final: list[str] = []
        self.structured: list[str] = []

    def feed_structured(self, value: Any) -> None:
        if isinstance(value, str) and value:
            self.structured.append(value)

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        data = self.carry + chunk
        self.carry = ""
        keep = max(len(self.OPEN), len(self.CLOSE)) - 1
        while data:
            open_at, close_at = data.find(self.OPEN), data.find(self.CLOSE)
            positions = [(pos, tag) for pos, tag in ((open_at, self.OPEN), (close_at, self.CLOSE)) if pos >= 0]
            if not positions:
                if len(data) <= keep:
                    self.carry = data
                else:
                    safe, self.carry = data[:-keep], data[-keep:]
                    self._append_text(safe)
                return
            pos, tag = min(positions, key=lambda item: item[0])
            self._append_text(data[:pos])
            data = data[pos + len(tag):]
            if tag == self.OPEN:
                if self.state == "unknown":
                    self.final.extend(self.tentative)
                    self.tentative.clear()
                self.state = "thinking"
            elif self.state == "unknown":
                self.thinking.extend(self.tentative)
                self.tentative.clear()
                self.state = "final"
            else:
                self.state = "final"

    def _append_text(self, text: str) -> None:
        if not text:
            return
        if self.state == "thinking":
            self.thinking.append(text)
        elif self.state == "final":
            self.final.append(text)
        else:
            self.tentative.append(text)

    def finish(self) -> AssistantContent:
        self._append_text(self.carry)
        self.carry = ""
        if self.state == "unknown":
            self.final.extend(self.tentative)
        elif self.state == "thinking":
            self.thinking.extend(self.tentative)
        self.tentative.clear()
        thinking = "".join(self.structured) if self.structured else "".join(self.thinking)
        return AssistantContent(thinking=thinking.strip(), final="".join(self.final).strip())

    def snapshot(self) -> AssistantContent:
        """Return safe display text without flushing a possible partial tag."""
        thinking = "".join(self.structured) if self.structured else "".join(self.thinking)
        return AssistantContent(thinking=thinking.strip(), final="".join(self.final).strip())


def _structured_reasoning(response: dict[str, Any]) -> list[str]:
    values: list[str] = []
    containers: list[dict[str, Any]] = [response]
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                containers.append(choice)
                for key in ("message", "delta"):
                    if isinstance(choice.get(key), dict):
                        containers.append(choice[key])
    for container in containers:
        for key in ("reasoning", "reasoning_content", "thinking"):
            if isinstance(container.get(key), str) and container[key]:
                values.append(container[key])
    return values


def parse_assistant_responses(responses: list[dict[str, Any]]) -> AssistantContent:
    parser = ThinkingParser()
    for response in responses:
        for reasoning in _structured_reasoning(response):
            parser.feed_structured(reasoning)
        parser.feed(extract_text(response))
    return parser.finish()


def thinking_box_lines(content: str, language: str, width: int) -> list[str]:
    title = ui_text(language, "thinking_title")
    body = ANSI_ESCAPE_RE.sub("", content).strip() or ui_text(language, "thinking_empty")
    inner = max(12, width - 4)
    top_label = f" {title} "
    top = "╭" + top_label + "─" * max(0, inner - len(top_label) + 1) + "╮"
    lines = [top[:width]]
    for paragraph in body.splitlines() or [""]:
        remaining = paragraph
        if not remaining:
            lines.append("│ " + " " * inner + " │")
        while remaining:
            piece, remaining = remaining[:inner], remaining[inner:]
            lines.append("│ " + piece.ljust(inner) + " │")
    lines.append("╰" + "─" * (inner + 2) + "╯")
    return [line[:width] for line in lines]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def safe_addstr(win: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        height, width = win.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        room = max(0, width - x - 1)
        win.addnstr(y, x, text, room, attr)
    except curses.error:
        pass


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 3600.0,
) -> tuple[dict[str, Any], float]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        text = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}") from exc
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

    elapsed = time.perf_counter() - started
    try:
        value = json.loads(raw)
    except Exception as exc:
        text = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"non-JSON response: {text[:1200]}") from exc

    if not isinstance(value, dict):
        raise RuntimeError("JSON response is not an object")
    return value, elapsed


def http_json_stream(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    on_chunk: Any = None,
) -> tuple[list[dict[str, Any]], float]:
    """Read OpenAI-compatible SSE without ever exposing an incomplete data frame."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    responses: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data_lines: list[str] = []
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line:
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    continue
                if not data_lines:
                    continue
                data = "\n".join(data_lines)
                data_lines.clear()
                if data == "[DONE]":
                    break
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise RuntimeError("stream event is not a JSON object")
                responses.append(value)
                if on_chunk is not None:
                    on_chunk(value)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}") from exc
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
    return responses, time.perf_counter() - started


def merge_stream_metadata(responses: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for response in responses:
        for key in ("usage", "ds4_runtime", "timing_ms", "prompt_processing_ms"):
            if key in response and response[key] is not None:
                merged[key] = response[key]
    return merged


def tcp_ready(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pid_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def count_buffer() -> int:
    try:
        with BUFFER_INDEX.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def active_checkpoint_name() -> str:
    link = ROOT / "active/checkpoint"
    try:
        return link.resolve().name
    except Exception:
        return "unknown"


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return delta["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    for key in ("text", "content"):
        if isinstance(response.get(key), str):
            return response[key]
    return ""


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


@dataclass
class TurnStats:
    mode: str = "none"
    fallback: str = ""
    prompt_tokens: int = 0
    prefill_seed_tokens: int = 0
    prompt_processing_ms: float | None = None
    committed_tokens: int = 0
    generation_wall_s: float = 0.0
    accepted: int = 0
    proposed: int = 0
    tried: int = 0
    cycles: int = 0
    total_wall_s: float = 0.0
    error: str = ""

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def gen_tps(self) -> float:
        return (
            self.committed_tokens / self.generation_wall_s
            if self.generation_wall_s > 0
            else 0.0
        )

    @property
    def prefill_tps(self) -> float | None:
        if self.prompt_processing_ms is None or self.prompt_processing_ms <= 0:
            return None
        return self.prompt_tokens / (self.prompt_processing_ms / 1000.0)

    @property
    def prefill_display(self) -> str:
        return format_prefill_metric(self.prompt_tokens, self.prompt_processing_ms)


@dataclass
class TotalStats:
    turns: int = 0
    fallbacks: int = 0
    dflash_turns: int = 0
    normal_turns: int = 0
    prompt_tokens: int = 0
    prefill_seed_tokens: int = 0
    prompt_processing_ms: float = 0.0
    prompt_metric_turns: int = 0
    prompt_metric_tokens: int = 0
    committed_tokens: int = 0
    generation_wall_s: float = 0.0
    accepted: int = 0
    proposed: int = 0
    tried: int = 0
    cycles: int = 0
    errors: int = 0

    def add(self, turn: TurnStats) -> None:
        self.turns += 1
        if turn.fallback:
            self.fallbacks += 1
        if turn.mode == "dflash":
            self.dflash_turns += 1
        else:
            self.normal_turns += 1
        self.prompt_tokens += turn.prompt_tokens
        self.prefill_seed_tokens += turn.prefill_seed_tokens
        if turn.prompt_processing_ms is not None and turn.prompt_processing_ms > 0:
            self.prompt_processing_ms += turn.prompt_processing_ms
            self.prompt_metric_turns += 1
            self.prompt_metric_tokens += turn.prompt_tokens
        self.committed_tokens += turn.committed_tokens
        self.generation_wall_s += turn.generation_wall_s
        self.accepted += turn.accepted
        self.proposed += turn.proposed
        self.tried += turn.tried
        self.cycles += turn.cycles
        if turn.error:
            self.errors += 1

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def gen_tps(self) -> float:
        return (
            self.committed_tokens / self.generation_wall_s
            if self.generation_wall_s > 0
            else 0.0
        )

    @property
    def prefill_tps(self) -> float | None:
        if self.prompt_processing_ms <= 0:
            return None
        return self.prompt_metric_tokens / (self.prompt_processing_ms / 1000.0)

    @property
    def prefill_display(self) -> str:
        return format_prefill_metric(
            self.prompt_metric_tokens if self.prompt_metric_turns else self.prompt_tokens,
            self.prompt_processing_ms if self.prompt_metric_turns else None,
        )


class DS4TUI:
    def __init__(self, stdscr: curses.window, args: argparse.Namespace) -> None:
        self.stdscr = stdscr
        self.args = args
        self.base_server = args.server.rstrip("/")
        self.base_proxy = args.proxy.rstrip("/")
        self.sidecar_health = args.sidecar_health
        self.sidecar_generate = args.sidecar_generate
        self.language = normalize_language(args.language) or "en"
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": args.system,
            }
        ]
        self.transcript: list[tuple[str, str]] = []
        self.input_buffer = ""
        self.scroll = 0
        self.running = True
        self.busy = False
        self.phase = "IDLE"
        self.dflash_mode = args.dflash
        self.fallback_enabled = True
        self.auto_recover = True
        self.last = TurnStats()
        self.total = TotalStats()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.health: dict[str, Any] = {}
        self.buffer_samples = count_buffer()
        self.last_training: dict[str, Any] = {}
        self.status_message = ui_text(self.language, "ready")
        self.status_time = time.time()
        self._last_health_poll = 0.0
        self.color_enabled = False
        self.live_content: AssistantContent | None = None

    def add_line(self, role: str, text: str) -> None:
        self.transcript.append((role, text))
        self.scroll = 0

    def add_thinking(self, text: str) -> None:
        self.add_line(f"THINKING:{self.language}", text)

    def outgoing_messages(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        return build_request_messages(history, self.language)

    def set_status(self, text: str) -> None:
        self.status_message = text
        self.status_time = time.time()

    def component_state(self) -> dict[str, bool]:
        return {
            "sidecar": pid_alive(RUNTIME_PIDS / "sidecar.pid") and tcp_ready("127.0.0.1", 8091),
            "server": pid_alive(RUNTIME_PIDS / "server.pid") and tcp_ready("127.0.0.1", 8080),
            "collector": pid_alive(RUNTIME_PIDS / "collector.pid"),
            "proxy": pid_alive(RUNTIME_PIDS / "proxy.pid") and tcp_ready("127.0.0.1", 8081),
        }

    def load_training_state(self) -> dict[str, Any]:
        candidates = [
            STATE / "last_training.json",
            STATE / "training_state.json",
            ROOT / "training/state.json",
        ]
        for path in candidates:
            value = json_file(path)
            if value:
                return value
        return {}

    def poll_health(self) -> None:
        now = time.time()
        if now - self._last_health_poll < 1.5:
            return
        self._last_health_poll = now
        self.health = self.component_state()
        self.buffer_samples = count_buffer()
        self.last_training = self.load_training_state()

    def dflash_available(self) -> bool:
        return bool(self.health.get("server") and self.health.get("sidecar"))

    def effective_mode(self) -> str:
        if self.dflash_mode == "off":
            return "normal"
        if self.dflash_mode == "auto" and not self.dflash_available():
            return "normal"
        return "dflash"

    def normal_chat(self, payload: dict[str, Any]) -> tuple[AssistantContent, TurnStats]:
        started = time.perf_counter()
        endpoints: list[tuple[str, str]] = []
        if tcp_ready("127.0.0.1", 8081):
            endpoints.append(("proxy", self.base_proxy + "/v1/chat/completions"))
        endpoints.append(("server", self.base_server + "/v1/chat/completions"))

        errors: list[str] = []
        for name, endpoint in endpoints:
            try:
                if payload.get("stream"):
                    parser = ThinkingParser()

                    def consume(response_chunk: dict[str, Any]) -> None:
                        for reasoning in _structured_reasoning(response_chunk):
                            parser.feed_structured(reasoning)
                        parser.feed(extract_text(response_chunk))
                        self.events.put(("stream", parser.snapshot()))

                    responses, wall = http_json_stream(
                        endpoint, payload, self.args.timeout, consume
                    )
                    response = merge_stream_metadata(responses)
                    answer = parser.finish()
                else:
                    response, wall = http_json("POST", endpoint, payload, self.args.timeout)
                    answer = parse_assistant_responses([response])
                usage = response.get("usage")
                if not isinstance(usage, dict):
                    usage = {}
                prompt_tokens, prompt_processing_ms = prompt_metrics(response)
                stats = TurnStats(
                    mode="normal",
                    prompt_tokens=prompt_tokens,
                    prompt_processing_ms=prompt_processing_ms,
                    committed_tokens=as_int(usage.get("completion_tokens")),
                    generation_wall_s=wall,
                    total_wall_s=time.perf_counter() - started,
                )
                return answer, stats
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        raise RuntimeError(" | ".join(errors))

    def dflash_chat(self, payload: dict[str, Any]) -> tuple[AssistantContent, TurnStats]:
        started = time.perf_counter()
        reset, _ = http_json(
            "POST",
            self.base_server + "/v1/ds4/deepspec_reset_sample",
            {},
            30,
        )
        if reset.get("ok") is not True:
            raise RuntimeError(f"DeepSpec reset failed: {reset}")

        prefill_payload = dict(payload)
        prefill_payload["max_tokens"] = self.args.prefill_tokens
        prefill_payload["stream"] = False

        self.events.put(("phase", "PREFILL"))
        prefill, prefill_wall = http_json(
            "POST",
            self.base_server + "/v1/chat/completions",
            prefill_payload,
            self.args.timeout,
        )
        if "error" in prefill:
            raise RuntimeError(f"prefill failed: {prefill['error']}")

        self.events.put(("phase", "DFLASH"))
        dflash, generation_wall = http_json(
            "POST",
            self.base_server + "/v1/ds4/deepspec_generate_dflash",
            {
                "max_tokens": max(1, self.args.max_tokens - self.args.prefill_tokens),
                "temperature": self.args.temperature,
                "seed": self.args.seed,
                "spec_top_k": self.args.top_k,
                "sidecar_url": self.sidecar_generate,
            },
            self.args.timeout,
        )
        if dflash.get("ok") is not True:
            raise RuntimeError(f"DFlash failed: {dflash}")

        usage = prefill.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        answer = parse_assistant_responses([prefill, dflash])
        prompt_tokens, prompt_processing_ms = prompt_metrics(prefill)
        stats = TurnStats(
            mode="dflash",
            prompt_tokens=prompt_tokens,
            prefill_seed_tokens=as_int(usage.get("completion_tokens")),
            prompt_processing_ms=prompt_processing_ms,
            committed_tokens=as_int(dflash.get("committed_tokens")),
            generation_wall_s=generation_wall,
            accepted=as_int(dflash.get("accepted_draft_tokens")),
            proposed=as_int(dflash.get("total_draft_token_count")),
            tried=as_int(dflash.get("total_tried_tokens")),
            cycles=as_int(dflash.get("cycles")),
            total_wall_s=time.perf_counter() - started,
        )
        return answer, stats

    def recover_stack(self) -> bool:
        if not LAUNCHER.is_file():
            return False
        self.events.put(("phase", "RECOVERY"))
        try:
            process = subprocess.run(
                [str(LAUNCHER), "restart"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
            self.events.put(("runtime_log", process.stdout[-4000:]))
            if process.returncode != 0:
                return False
            deadline = time.time() + 180
            while time.time() < deadline:
                if tcp_ready("127.0.0.1", 8080) and tcp_ready("127.0.0.1", 8081):
                    return True
                time.sleep(1)
            return False
        except Exception as exc:
            self.events.put(("runtime_log", f"recovery exception: {exc}"))
            return False

    def send_worker(self, prompt: str) -> None:
        tentative = self.messages + [{"role": "user", "content": prompt}]
        payload = {
            "model": "ds4",
            "messages": self.outgoing_messages(tentative),
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "seed": self.args.seed,
            "think_mode": "none",
            "stream": self.args.stream,
            "stream_options": {"include_usage": True},
            "ds4_return_runtime_metrics": True,
            "ds4_return_token_ids": True,
        }

        try:
            mode = self.effective_mode()
            if mode == "dflash":
                try:
                    answer, stats = self.dflash_chat(payload)
                except Exception as dflash_exc:
                    if not self.fallback_enabled:
                        raise
                    self.events.put(("phase", "FALLBACK"))
                    answer, stats = self.normal_chat(payload)
                    stats.fallback = str(dflash_exc)
            else:
                answer, stats = self.normal_chat(payload)

        except Exception as first_exc:
            if self.auto_recover and self.recover_stack():
                try:
                    answer, stats = self.normal_chat(payload)
                    stats.fallback = f"runtime recovery: {first_exc}"
                except Exception as second_exc:
                    self.events.put(
                        (
                            "error",
                            ui_text(self.language, "request_failed", error=second_exc),
                        )
                    )
                    return
            else:
                self.events.put(("error", str(first_exc)))
                return

        self.events.put(
            (
                "answer",
                {
                    "prompt": prompt,
                    "thinking": answer.thinking,
                    "answer": answer.final,
                    "messages": tentative,
                    "stats": asdict(stats),
                },
            )
        )

    def start_send(self, prompt: str) -> None:
        if self.busy:
            self.set_status(ui_text(self.language, "busy"))
            return
        self.busy = True
        self.phase = "START"
        self.live_content = AssistantContent()
        self.add_line(ui_text(self.language, "you"), prompt)
        thread = threading.Thread(target=self.send_worker, args=(prompt,), daemon=True)
        thread.start()

    def handle_command(self, command: str) -> None:
        parts = command.strip().split()
        head = parts[0].lower() if parts else ""

        if head in {"/quit", "/exit"}:
            self.running = False
            return

        if head == "/help":
            self.add_line("SYS", ui_text(self.language, "help"))
            return

        if head == "/language":
            if len(parts) == 1:
                self.add_line(
                    "SYS",
                    ui_text(self.language, "language_status", language=self.language.upper()),
                )
                return
            selected = normalize_language(parts[1])
            if selected is None:
                self.add_line("SYS", ui_text(self.language, "language_invalid"))
                return
            self.language = selected
            self.status_message = ui_text(self.language, "ready")
            self.add_line("SYS", ui_text(self.language, "language_set"))
            return

        if head == "/dflash":
            if len(parts) == 1:
                self.add_line(
                    "SYS",
                    ui_text(self.language, "dflash_status", configured=self.dflash_mode, effective=self.effective_mode()),
                )
            elif parts[1].lower() in {"on", "off", "auto"}:
                self.dflash_mode = parts[1].lower()
                self.add_line(
                    "SYS",
                    ui_text(self.language, "dflash_set", configured=self.dflash_mode, effective=self.effective_mode()),
                )
            else:
                self.add_line("SYS", ui_text(self.language, "dflash_usage"))
            return

        if head == "/fallback":
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                self.fallback_enabled = parts[1] == "on"
            self.add_line(
                "SYS",
                ui_text(self.language, "fallback", state="ON" if self.fallback_enabled else "OFF"),
            )
            return

        if head == "/recover":
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                self.auto_recover = parts[1] == "on"
                self.add_line(
                    "SYS",
                    ui_text(self.language, "recover", state="ON" if self.auto_recover else "OFF"),
                )
            elif len(parts) == 2 and parts[1] == "now":
                if self.busy:
                    self.add_line("SYS", ui_text(self.language, "recover_busy"))
                else:
                    self.busy = True
                    threading.Thread(target=self.manual_recover_worker, daemon=True).start()
            else:
                self.add_line(
                    "SYS",
                    ui_text(self.language, "recover_usage", state="ON" if self.auto_recover else "OFF"),
                )
            return

        if head == "/reset":
            self.messages = [{"role": "system", "content": self.args.system}]
            self.transcript.clear()
            self.add_line("SYS", ui_text(self.language, "reset"))
            return

        if head == "/clearstats":
            self.total = TotalStats()
            self.last = TurnStats()
            self.add_line("SYS", ui_text(self.language, "cleared"))
            return

        if head == "/stats":
            self.add_line(
                "STATS",
                (
                    f"{ui_text(self.language, 'stats_turns')}={self.total.turns} "
                    f"dflash={self.total.dflash_turns} "
                    f"{ui_text(self.language, 'stats_normal')}={self.total.normal_turns} "
                    f"fallback={self.total.fallbacks}\n"
                    f"{self.total.prefill_display}\n"
                    f"gen={self.total.committed_tokens} tok @ {self.total.gen_tps:.2f} tok/s\n"
                    f"accepted={self.total.accepted}/{self.total.proposed} "
                    f"({self.total.acceptance:.1%}) tried={self.total.tried} "
                    f"cycles={self.total.cycles} errors={self.total.errors}"
                ),
            )
            return

        if head == "/status":
            states = self.component_state()
            self.add_line(
                "SYS",
                " ".join(f"{name}={'UP' if up else 'DOWN'}" for name, up in states.items()),
            )
            return

        if head == "/buffer":
            self.add_line("SYS", ui_text(self.language, "buffer", count=count_buffer()))
            return

        if head == "/training":
            training = self.load_training_state()
            if training:
                self.add_line("TRAIN", json.dumps(training, ensure_ascii=False, indent=2))
            else:
                self.add_line("TRAIN", ui_text(self.language, "no_training"))
            return

        self.add_line("SYS", ui_text(self.language, "unknown", command=command))

    def manual_recover_worker(self) -> None:
        ok = self.recover_stack()
        self.events.put(("recovered", ok))

    def process_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return

            if kind == "phase":
                self.phase = str(payload)
            elif kind == "runtime_log":
                if payload:
                    self.add_line("RUNTIME", str(payload))
            elif kind == "recovered":
                self.busy = False
                self.phase = "IDLE"
                self.add_line(
                    "SYS",
                    ui_text(self.language, "recovered" if payload else "recovery_failed"),
                )
            elif kind == "error":
                self.busy = False
                self.phase = "ERROR"
                self.live_content = None
                self.total.errors += 1
                self.add_line("ERR", str(payload))
            elif kind == "stream":
                self.live_content = payload
            elif kind == "answer":
                self.busy = False
                self.phase = "IDLE"
                self.live_content = None
                data = payload
                stats = TurnStats(**data["stats"])
                self.last = stats
                self.total.add(stats)
                self.add_thinking(data.get("thinking", ""))
                answer = data["answer"] or ui_text(self.language, "empty_answer")
                self.add_line("DS4", answer)
                self.messages = data["messages"] + [
                    {"role": "assistant", "content": answer}
                ]
                if stats.fallback:
                    self.add_line("FALLBACK", stats.fallback)

    def wrapped_lines(self, width: int) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        width = max(20, width)
        for role, text in self.transcript:
            if role.startswith("THINKING:"):
                box_language = role.partition(":")[2] or self.language
                lines.extend(
                    ("thinking", line)
                    for line in thinking_box_lines(str(text), box_language, width)
                )
                lines.append(("thinking", ""))
                continue
            prefix = f"{role}> "
            chunks = str(text).splitlines() or [""]
            first = True
            for chunk in chunks:
                remaining = chunk
                while remaining:
                    current_prefix = prefix if first else " " * len(prefix)
                    room = max(1, width - len(current_prefix))
                    lines.append((role, current_prefix + remaining[:room]))
                    remaining = remaining[room:]
                    first = False
                if chunk == "":
                    lines.append((role, prefix if first else ""))
                    first = False
            lines.append((role, ""))
        if self.live_content is not None:
            lines.extend(
                ("thinking", line)
                for line in thinking_box_lines(
                    self.live_content.thinking, self.language, width
                )
            )
            lines.append(("thinking", ""))
            if self.live_content.final:
                prefix = "DS4> "
                remaining = self.live_content.final
                first = True
                while remaining:
                    current_prefix = prefix if first else " " * len(prefix)
                    room = max(1, width - len(current_prefix))
                    lines.append(("DS4", current_prefix + remaining[:room]))
                    remaining = remaining[room:]
                    first = False
        return lines

    def draw(self) -> None:
        self.poll_health()
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()

        states = self.health
        dflash_label = self.dflash_mode.upper()
        effective = self.effective_mode().upper()
        train_status = str(
            self.last_training.get("status")
            or self.last_training.get("state")
            or "IDLE"
        ).upper()
        checkpoint = active_checkpoint_name()

        header1 = (
            f" DS4 | DFLASH {dflash_label}/{effective} | "
            f"LANG:{self.language.upper()} | PHASE {self.phase} | TRAIN {train_status} "
        )
        safe_addstr(self.stdscr, 0, 0, header1, curses.A_REVERSE | curses.A_BOLD)

        components = " ".join(
            f"{key[:3].upper()}:{'UP' if value else 'DOWN'}"
            for key, value in states.items()
        )
        header2 = (
            f" {components} | BUFFER:{self.buffer_samples} | CKPT:{checkpoint} "
        )
        safe_addstr(self.stdscr, 1, 0, header2, curses.A_REVERSE)

        stats1 = (
            f" LAST mode={self.last.mode or '-'} "
            f"accept={self.last.accepted}/{self.last.proposed} "
            f"({self.last.acceptance:.1%}) tried={self.last.tried} "
            f"cycles={self.last.cycles} committed={self.last.committed_tokens} "
        )
        safe_addstr(self.stdscr, 2, 0, stats1, curses.A_BOLD)

        stats2 = (
            f" {self.last.prefill_display} | "
            f"GEN {self.last.gen_tps:.2f} tok/s {self.last.generation_wall_s:.1f}s | "
            f"TOTAL turns={self.total.turns} fallback={self.total.fallbacks} "
            f"accept={self.total.acceptance:.1%} "
        )
        safe_addstr(self.stdscr, 3, 0, stats2)

        divider_y = 4
        safe_addstr(self.stdscr, divider_y, 0, "-" * max(1, width - 1))

        input_y = max(divider_y + 2, height - 3)
        status_y = max(divider_y + 1, height - 4)
        transcript_height = max(1, status_y - divider_y - 1)

        all_lines = self.wrapped_lines(max(20, width - 1))
        max_scroll = max(0, len(all_lines) - transcript_height)
        self.scroll = clamp(self.scroll, 0, max_scroll)
        end = len(all_lines) - self.scroll
        start = max(0, end - transcript_height)
        visible = all_lines[start:end]

        for index, (role, line) in enumerate(visible):
            attr = 0
            if role == "thinking":
                attr = curses.A_DIM
                if self.color_enabled:
                    attr |= curses.color_pair(1)
            safe_addstr(self.stdscr, divider_y + 1 + index, 0, line, attr)

        footer = (
            f" {self.status_message} | LANG:{self.language.upper()} | /language en|it | /dflash on|off|auto | "
            f"/fallback on|off | /recover now | /help "
        )
        safe_addstr(self.stdscr, status_y, 0, footer, curses.A_REVERSE)

        prompt = f"{ui_text(self.language, 'you')}> "
        safe_addstr(self.stdscr, input_y, 0, prompt, curses.A_BOLD)
        room = max(1, width - len(prompt) - 2)
        shown = self.input_buffer[-room:]
        safe_addstr(self.stdscr, input_y, len(prompt), shown)
        try:
            curses.setsyx(input_y, min(width - 2, len(prompt) + len(shown)))
        except curses.error:
            pass

        safe_addstr(
            self.stdscr,
            height - 1,
            0,
            f" {ui_text(self.language, 'footer')} ",
            curses.A_DIM,
        )
        self.stdscr.refresh()

    def run(self) -> int:
        curses.curs_set(1)
        if os.environ.get("NO_COLOR") is None and curses.has_colors():
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                self.color_enabled = True
            except curses.error:
                self.color_enabled = False
        self.stdscr.timeout(100)
        self.stdscr.keypad(True)

        self.add_line(
            "SYS",
            (
                ui_text(self.language, "startup")
            ),
        )
        self.add_line(
            "SYS",
            (
                ui_text(self.language, "fallback_startup")
            ),
        )

        while self.running:
            self.process_events()
            self.draw()
            try:
                key = self.stdscr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                break

            if key == curses.KEY_RESIZE:
                continue
            if key in (curses.KEY_PPAGE,):
                self.scroll += 5
                continue
            if key in (curses.KEY_NPAGE,):
                self.scroll = max(0, self.scroll - 5)
                continue
            if key in ("\n", "\r", curses.KEY_ENTER):
                text = self.input_buffer.strip()
                self.input_buffer = ""
                if not text:
                    continue
                if text.startswith("/"):
                    self.handle_command(text)
                else:
                    self.start_send(text)
                continue
            if key in ("\b", "\x7f", curses.KEY_BACKSPACE):
                self.input_buffer = self.input_buffer[:-1]
                continue
            if key == "\x03":
                break
            if isinstance(key, str) and key.isprintable():
                self.input_buffer += key

        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS4 Codex-style terminal UI")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--proxy", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--sidecar-health",
        default="http://127.0.0.1:8091/health",
    )
    parser.add_argument(
        "--sidecar-generate",
        default="http://127.0.0.1:8091/v1/ds4/dflash_propose_from_raw",
    )
    parser.add_argument("--dflash", choices=("on", "off", "auto"), default="off")
    parser.add_argument("--language", choices=("en", "it"), default="en")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0001)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=False,
        help="stream normal/fallback replies when the endpoint supports SSE",
    )
    parser.add_argument(
        "--system",
        default="Be clear, useful, and direct.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_tokens <= args.prefill_tokens:
        raise SystemExit("--max-tokens must be greater than --prefill-tokens")
    return curses.wrapper(lambda stdscr: DS4TUI(stdscr, args).run())


if __name__ == "__main__":
    raise SystemExit(main())
