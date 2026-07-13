#!/usr/bin/env python3
"""Noninteractive live TUI harness for localized OFF/ON presentation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


UI_PATH = Path(__file__).resolve().parents[1] / "scripts/ds4_ui.py"
SPEC = importlib.util.spec_from_file_location("ds4_ui_transcript", UI_PATH)
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui
SPEC.loader.exec_module(ui)


class Window:
    def getmaxyx(self): return (30, 100)
    def addnstr(self, *_args): pass
    def erase(self): pass
    def refresh(self): pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--small-gate", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    cli = parser.parse_args()
    args = argparse.Namespace(
        server="http://127.0.0.1:8080", proxy="http://127.0.0.1:8081",
        sidecar_health="http://127.0.0.1:8091/health",
        sidecar_generate="http://127.0.0.1:8091/v1/ds4/dflash_propose_from_raw",
        dflash="off", language="en", max_tokens=32, prefill_tokens=2,
        temperature=0.0, top_k=8, seed=260713, timeout=cli.timeout,
        stream=False, system="Be clear, useful, and direct.",
    )
    tui = ui.DS4TUI(Window(), args)
    tui.handle_command("/language it")
    tui.handle_command("/dflash off")
    prompt = "Rispondi soltanto con la parola OK."
    history = tui.messages + [{"role": "user", "content": prompt}]
    payload = tui.chat_payload(history)
    off_content, off_stats = tui.normal_chat(payload)
    tui.add_line("TU", prompt)
    tui.add_thinking(off_content.thinking)
    tui.add_line("DS4", off_content.final or ui.ui_text("it", "empty_answer"))
    if off_stats.token_limit_reached:
        tui.add_line("NOTICE", ui.ui_text("it", "token_limit_notice"))

    tui.handle_command("/reset")
    tui.handle_command("/dflash on")
    history = tui.messages + [{"role": "user", "content": prompt}]
    on_content, on_stats = tui.dflash_chat(tui.chat_payload(history))
    tui.add_line("TU", prompt)
    tui.add_thinking(on_content.thinking)
    tui.add_line("DS4", on_content.final or ui.ui_text("it", "empty_answer"))
    if on_stats.token_limit_reached:
        tui.add_line("NOTICE", ui.ui_text("it", "token_limit_notice"))

    small = json.loads(cli.small_gate.read_text(encoding="utf-8"))
    tui.handle_command("/max-tokens 8")
    tui.add_line("TU", "[turno bounded al limite configurato]")
    tui.add_thinking(small["normalized"]["thinking"])
    tui.add_line("DS4", small["normalized"]["final"])
    tui.add_line("NOTICE", small["limit_notice"])
    tui.handle_command("/max-tokens 32")
    tui.handle_command("/language en")

    transcript = [
        "DS4 TUI noninteractive transcript",
        "LANG:IT | MAX:32",
    ]
    transcript.extend(line for _, line in tui.wrapped_lines(96))
    transcript.extend(["LANG:EN", "NO RAW THINK MARKERS"])
    rendered = "\n".join(transcript) + "\n"
    cli.output.write_text(rendered, encoding="utf-8")
    result = {
        "pass": bool(
            off_content.final == on_content.final
            and not off_stats.token_limit_reached
            and not on_stats.token_limit_reached
            and on_stats.accepted > 0
            and "<think>" not in rendered
            and "</think>" not in rendered
        ),
        "off": {"content": off_content.__dict__, "stats": ui.asdict(off_stats)},
        "on": {"content": on_content.__dict__, "stats": ui.asdict(on_stats)},
    }
    cli.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
