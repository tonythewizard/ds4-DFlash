#!/usr/bin/env python3
"""Bounded live-stack smoke for DS4 UI OFF/ON exactness and presentation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


UI_PATH = Path(__file__).resolve().parents[1] / "scripts/ds4_ui.py"
SPEC = importlib.util.spec_from_file_location("ds4_ui_runtime", UI_PATH)
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui
SPEC.loader.exec_module(ui)


def tokens(response: dict[str, Any]) -> list[int]:
    values = (response.get("ds4_runtime") or {}).get("completion_token_ids")
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise RuntimeError("native completion token IDs are missing")
    return values


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response, _ = ui.http_json("POST", url, payload, timeout)
    return response


def reset(server: str, timeout: float) -> None:
    response = post(server + "/v1/ds4/deepspec_reset_sample", {}, min(timeout, 30))
    if response.get("ok") is not True:
        raise RuntimeError(f"reset failed: {response}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--proxy", default="http://127.0.0.1:8081")
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--prefill-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    args = parser.parse_args()

    if args.max_tokens <= args.prefill_tokens:
        parser.error("--max-tokens must exceed --prefill-tokens")

    prompt = "Reply with one short sentence about deterministic testing."
    history = [
        {"role": "system", "content": "Be clear, useful, and direct."},
        {"role": "user", "content": prompt},
    ]
    messages = ui.build_request_messages(history, "en")
    language_count = sum(
        message.get("content") in ui.LANGUAGE_INSTRUCTIONS.values()
        for message in messages
    )
    payload = {
        "model": "ds4",
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 260713,
        "stream": False,
        "thinking": False,
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }

    reset(args.server, args.timeout)
    off = post(args.proxy + "/v1/chat/completions", payload, args.timeout)
    off_ids = tokens(off)
    off_content = ui.parse_assistant_responses([off])

    reset(args.server, args.timeout)
    prefill_payload = dict(payload)
    prefill_payload["max_tokens"] = args.prefill_tokens
    prefill = post(args.server + "/v1/chat/completions", prefill_payload, args.timeout)
    prefill_ids = tokens(prefill)
    prefill_content = ui.parse_assistant_responses([prefill])
    dflash = post(args.server + "/v1/ds4/deepspec_generate_dflash", {
        "max_tokens": args.max_tokens - len(prefill_ids),
        "temperature": 0.0,
        "seed": 260713,
        "spec_top_k": 8,
        "reasoning_active": bool(prefill_content.thinking and not prefill_content.final),
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }, args.timeout)
    on_ids = prefill_ids + tokens(dflash)
    on_content = ui.parse_assistant_responses([prefill, dflash])

    off_prefill = ui.format_prefill_metric(*ui.prompt_metrics(off))
    on_prefill = ui.format_prefill_metric(*ui.prompt_metrics(prefill))
    proposed = int(dflash.get("total_draft_token_count", 0))
    accepted = int(dflash.get("accepted_draft_tokens", 0))

    italian_messages = ui.build_request_messages(history, "it")
    english_messages = ui.build_request_messages(history, "en")
    result = {
        "schema": "ds4_ui_runtime_smoke_v1",
        "pass": bool(
            language_count == 1
            and off_ids == on_ids
            and off_content.final == on_content.final
            and proposed > 0
            and accepted > 0
            and "0.00 tok/s" not in off_prefill
            and "<think>" not in off_content.final
            and "</think>" not in off_content.final
            and "<think>" not in on_content.final
            and "</think>" not in on_content.final
        ),
        "english_default": True,
        "language_instruction_count": language_count,
        "english_instruction": english_messages[1],
        "italian_instruction": italian_messages[1],
        "off": {
            "token_ids": off_ids,
            "final": off_content.final,
            "thinking": off_content.thinking,
            "prefill_display": off_prefill,
            "proposed": int((off.get("ds4_runtime") or {}).get("proposed_draft_tokens", 0)),
        },
        "on": {
            "token_ids": on_ids,
            "final": on_content.final,
            "thinking": on_content.thinking,
            "prefill_display": on_prefill,
            "proposed": proposed,
            "accepted": accepted,
        },
        "final_exact": off_ids == on_ids and off_content.final == on_content.final,
        "finish_reason_off": ui.response_finish(off),
        "finish_reason_on": ui.response_finish(dflash),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "DS4 TUI bounded transcript",
        "==========================",
        "LANG:EN (default)",
        f"OFF  {off_prefill} | proposed=0",
        *ui.thinking_box_lines(off_content.thinking, "en", 72),
        f"DS4> {off_content.final}",
        f"ON   {on_prefill} | accepted={accepted}/{proposed}",
        *ui.thinking_box_lines(on_content.thinking, "en", 72),
        f"DS4> {on_content.final}",
        f"OFF/ON final exact: {result['final_exact']}",
        "SYS> Language set to Italian.",
        "LANG:IT",
        *ui.thinking_box_lines("", "it", 72),
        f"payload system> {italian_messages[1]['content']}",
        "SYS> Conversazione azzerata",
        "SYS> Language set to English.",
        "LANG:EN",
    ]
    args.transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
