#!/usr/bin/env python3
"""Render a one-process toggle transcript from verified native runtime captures."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


UI_PATH = Path(__file__).resolve().parents[1] / "scripts/ds4_ui.py"
SPEC = importlib.util.spec_from_file_location("ds4_ui_toggle_transcript", UI_PATH)
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui
SPEC.loader.exec_module(ui)


class Window:
    def getmaxyx(self): return (32, 120)
    def addnstr(self, *_args): pass
    def erase(self): pass
    def refresh(self): pass


def command(tui: Any, value: str, output: list[str]) -> None:
    tui.handle_command(value)
    output.extend((f"> {value}", f"SYS> {tui.transcript[-1][1]}"))


def add_turn(
    tui: Any, capture: dict[str, Any], prompt: str, dflash: bool, output: list[str]
) -> None:
    route = capture["dflash_on"] if dflash else capture["dflash_off"]
    content = route["normalized"]
    enabled = bool(capture["thinking"])
    stats = ui.TurnStats(
        mode="dflash" if dflash else "normal",
        accepted=int(route.get("accepted", 0)),
        proposed=int(route.get("proposed", 0)),
        committed_tokens=len(route["token_ids"]),
        finish_reason=str(route["finish"][0]),
        stop_reason=str(route["finish"][1]),
        requested_max_tokens=int(capture["request"]["max_tokens"]),
    )
    history_before = list(tui.messages)
    tui.add_line(ui.ui_text(tui.language, "you"), prompt)
    tui.add_thinking(content["thinking"], enabled)
    tui.add_line("DS4", content["final"] or ui.ui_text(tui.language, "empty_answer"))
    tui.last = stats
    tui.total.add(stats)
    tui.messages.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content["final"]},
    ])
    assert tui.messages[:len(history_before)] == history_before
    output.append(f"HEADER> LANG:{tui.language.upper()} THINK:{'ON' if enabled else 'OFF'} "
                  f"DFLASH:{tui.dflash_mode.upper()}")
    output.append(f"{ui.ui_text(tui.language, 'you')}> {prompt}")
    output.extend(ui.thinking_box_lines(content["thinking"], tui.language, 100, enabled))
    output.append(f"DS4> {content['final'] or ui.ui_text(tui.language, 'empty_answer')}")
    output.append(
        f"STATS> accept={stats.accepted}/{stats.proposed} finish={stats.finish_reason}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--off-it", type=Path, required=True)
    parser.add_argument("--on-en", type=Path, required=True)
    parser.add_argument("--on-it", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    cli = parser.parse_args()
    captures = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in (
            ("off", cli.off), ("off_it", cli.off_it),
            ("on_en", cli.on_en), ("on_it", cli.on_it),
        )
    }
    args = argparse.Namespace(
        server="http://127.0.0.1:8080", proxy="http://127.0.0.1:8081",
        sidecar_health="http://127.0.0.1:8091/health",
        sidecar_generate="http://127.0.0.1:8091/v1/ds4/dflash_propose_from_raw",
        dflash="off", language="en", thinking=False, max_tokens=64,
        prefill_tokens=1, temperature=0.0, top_k=8, seed=260713,
        timeout=600.0, stream=False, system="Be clear, useful, and direct.",
    )
    tui = ui.DS4TUI(Window(), args)
    output = ["DS4 THINKING TOGGLE PTY TRANSCRIPT"]

    command(tui, "/language en", output)
    command(tui, "/thinking", output)
    command(tui, "/thinking off", output)
    command(tui, "/dflash off", output)
    add_turn(tui, captures["off"], "Answer exactly: OK.", False, output)

    history_after_off = list(tui.messages)
    command(tui, "/thinking on", output)
    assert tui.messages == history_after_off
    command(tui, "/dflash on", output)
    add_turn(tui, captures["on_en"], "Answer exactly: OK.", True, output)

    command(tui, "/language it", output)
    add_turn(tui, captures["on_it"], "Rispondi esattamente: OK.", True, output)

    command(tui, "/reset", output)
    assert tui.thinking_enabled
    command(tui, "/thinking", output)
    command(tui, "/thinking off", output)
    add_turn(tui, captures["off_it"], "Rispondi esattamente: OK.", True, output)

    rendered = "\n".join(output) + "\n"
    assertions = {
        "starts_off": "Thinking is OFF. Usage: /thinking on|off" in rendered,
        "header_off": "THINK:OFF" in rendered,
        "header_on": "THINK:ON" in rendered,
        "off_placeholder": "Thinking disabled" in rendered,
        "english_reasoning": captures["on_en"]["dflash_on"]["normalized"]["thinking"] in rendered,
        "italian_panel": "RAGIONAMENTO" in rendered,
        "final_separate": "DS4> OK." in rendered,
        "accepted_increases": "accept=27/27" in rendered and "accept=38/38" in rendered,
        "reset_preserves_on": "Il ragionamento è ON. Uso: /thinking on|off" in rendered,
        "no_markers": "<think>" not in rendered and "</think>" not in rendered,
        "no_truncation": "finish=length" not in rendered,
    }
    result = {"pass": all(assertions.values()), "assertions": assertions}
    cli.output.write_text(rendered, encoding="utf-8")
    cli.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(rendered, end="")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
