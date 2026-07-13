#!/usr/bin/env python3
"""Bounded direct/OFF/ON gate for structured thinking and length stops."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


UI_PATH = Path(__file__).resolve().parents[1] / "scripts/ds4_ui.py"
SPEC = importlib.util.spec_from_file_location("ds4_ui_thinking_gate", UI_PATH)
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui
SPEC.loader.exec_module(ui)


def runtime_tokens(response: dict[str, Any]) -> list[int]:
    values = (response.get("ds4_runtime") or {}).get("completion_token_ids")
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise RuntimeError("native completion token IDs missing")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--proxy", default="http://127.0.0.1:8081")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.max_tokens <= 256 or not 1 <= args.prefill_tokens < args.max_tokens:
        parser.error("require 2 <= max_tokens <= 256 and 1 <= prefill_tokens < max_tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = [
        {"role": "system", "content": "Be clear, useful, and direct."},
        {"role": "user", "content": "Puoi spiegare meglio cosa intendi?"},
    ]
    payload = {
        "model": "ds4",
        "messages": ui.build_request_messages(history, "it"),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 260713,
        "thinking": False,
        "stream": False,
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }

    def post(url: str, body: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return ui.http_json("POST", url, body, timeout or args.timeout)[0]

    def reset() -> None:
        response = post(args.server + "/v1/ds4/deepspec_reset_sample", {}, 30)
        if response.get("ok") is not True:
            raise RuntimeError(f"reset failed: {response}")

    reset()
    direct = post(args.server + "/v1/chat/completions", payload)
    (args.output_dir / "DIRECT_RAW_RESPONSE.json").write_text(
        json.dumps({"request": payload, "response": direct}, ensure_ascii=False, indent=2) + "\n"
    )
    reset()
    off = post(args.proxy + "/v1/chat/completions", payload)
    (args.output_dir / "OFF_RAW_RESPONSE.json").write_text(
        json.dumps({"request": payload, "response": off}, ensure_ascii=False, indent=2) + "\n"
    )
    reset()
    prefill_payload = dict(payload)
    prefill_payload["max_tokens"] = args.prefill_tokens
    prefill = post(args.server + "/v1/chat/completions", prefill_payload)
    prefill_content = ui.parse_assistant_responses([prefill])
    dflash_request = {
        "max_tokens": args.max_tokens - args.prefill_tokens,
        "temperature": 0.0,
        "seed": 260713,
        "spec_top_k": 8,
        "reasoning_active": bool(prefill_content.thinking and not prefill_content.final),
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }
    dflash = post(args.server + "/v1/ds4/deepspec_generate_dflash", dflash_request)
    (args.output_dir / "ON_RAW_RESPONSE.json").write_text(
        json.dumps({"request": payload, "prefill": prefill, "dflash_request": dflash_request,
                    "dflash": dflash}, ensure_ascii=False, indent=2) + "\n"
    )

    direct_ids = runtime_tokens(direct)
    off_ids = runtime_tokens(off)
    on_ids = runtime_tokens(prefill) + runtime_tokens(dflash)
    direct_content = ui.parse_assistant_responses([direct])
    off_content = ui.parse_assistant_responses([off])
    on_content = ui.parse_assistant_responses([prefill, dflash])
    proposed = int(dflash.get("total_draft_token_count", 0))
    accepted = int(dflash.get("accepted_draft_tokens", 0))
    direct_finish = ui.response_finish(direct)
    off_finish = ui.response_finish(off)
    on_finish = ui.response_finish(dflash)
    marker_free = all(
        marker not in value
        for marker in ("<think>", "</think>")
        for content in (direct_content, off_content, on_content)
        for value in (content.thinking, content.final)
    )
    exact = direct_ids == off_ids == on_ids
    routed = (
        direct_content == off_content == on_content
        and not on_content.thinking
        and bool(on_content.final)
        and marker_free
    )
    off_result = {
        "pass": exact and routed and int((off.get("ds4_runtime") or {}).get("proposed_draft_tokens", 0)) == 0,
        "exact": direct_ids == off_ids,
        "token_ids": off_ids,
        "proposed": int((off.get("ds4_runtime") or {}).get("proposed_draft_tokens", 0)),
        "normalized": off_content.__dict__,
        "finish": off_finish,
        "limit_notice": ui.ui_text("it", "token_limit_notice") if ui.token_limit_reached(*off_finish) else "",
    }
    on_result = {
        "pass": exact and routed and proposed > 0 and accepted == proposed,
        "exact": direct_ids == on_ids,
        "token_ids": on_ids,
        "proposed": proposed,
        "accepted": accepted,
        "normalized": on_content.__dict__,
        "finish": on_finish,
        "limit_notice": ui.ui_text("it", "token_limit_notice") if ui.token_limit_reached(*on_finish) else "",
    }
    (args.output_dir / "DFLASH_OFF_RESULTS.json").write_text(
        json.dumps(off_result, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "DFLASH_ON_RESULTS.json").write_text(
        json.dumps(on_result, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "pass": off_result["pass"] and on_result["pass"],
        "direct_token_ids": direct_ids,
        "direct_normalized": direct_content.__dict__,
        "direct_finish": direct_finish,
        "off": off_result,
        "on": on_result,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
