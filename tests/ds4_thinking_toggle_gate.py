#!/usr/bin/env python3
"""Bounded native-token gate for DS4 thinking across direct/OFF/ON routes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


UI_PATH = Path(__file__).resolve().parents[1] / "scripts/ds4_ui.py"
SPEC = importlib.util.spec_from_file_location("ds4_ui_thinking_toggle_gate", UI_PATH)
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui
SPEC.loader.exec_module(ui)


def token_ids(response: dict[str, Any]) -> list[int]:
    runtime = response.get("ds4_runtime")
    values = runtime.get("completion_token_ids") if isinstance(runtime, dict) else None
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise RuntimeError("native completion token IDs missing")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--proxy", default="http://127.0.0.1:8081")
    parser.add_argument("--thinking", choices=("on", "off"), required=True)
    parser.add_argument("--language", choices=("en", "it"), default="en")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prefill-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260713)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prompt", default="Think briefly, then answer exactly: OK.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.max_tokens <= 256:
        parser.error("max tokens must be in 2..256")
    if not 1 <= args.prefill_tokens < args.max_tokens:
        parser.error("prefill tokens must be positive and smaller than max tokens")

    enabled = args.thinking == "on"
    history = [
        {"role": "system", "content": "Be clear, useful, and direct."},
        {"role": "user", "content": args.prompt},
    ]
    payload = ui.apply_thinking_control({
        "model": "ds4",
        "messages": ui.build_request_messages(history, args.language, enabled),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": False,
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }, enabled)

    def post(url: str, body: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return ui.http_json("POST", url, body, timeout or args.timeout)[0]

    def reset() -> None:
        response = post(args.server + "/v1/ds4/deepspec_reset_sample", {}, 30)
        if response.get("ok") is not True:
            raise RuntimeError(f"reset failed: {response}")

    reset()
    direct = post(args.server + "/v1/chat/completions", payload)
    reset()
    off = post(args.proxy + "/v1/chat/completions", payload)
    reset()
    prefill_payload = dict(payload)
    prefill_payload["max_tokens"] = args.prefill_tokens
    prefill = post(args.server + "/v1/chat/completions", prefill_payload)
    continuation_request = ui.apply_thinking_control({
        "max_tokens": args.max_tokens - len(token_ids(prefill)),
        "temperature": 0.0,
        "seed": args.seed,
        "spec_top_k": 8,
        "reasoning_active": enabled,
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }, enabled)
    dflash = post(
        args.server + "/v1/ds4/deepspec_generate_dflash", continuation_request
    )

    direct_ids = token_ids(direct)
    off_ids = token_ids(off)
    on_ids = token_ids(prefill) + token_ids(dflash)
    normalized = {
        "direct": ui.parse_assistant_responses([direct]),
        "off": ui.parse_assistant_responses([off]),
        "on": ui.parse_assistant_responses([prefill, dflash]),
    }
    exact = direct_ids == off_ids == on_ids
    normalized_exact = normalized["direct"] == normalized["off"] == normalized["on"]
    marker_free = all(
        marker not in value
        for marker in ("<think>", "</think>")
        for content in normalized.values()
        for value in (content.thinking, content.final)
    )
    proposed = int(dflash.get("total_draft_token_count", 0))
    accepted = int(dflash.get("accepted_draft_tokens", 0))
    off_proposals = int((off.get("ds4_runtime") or {}).get("proposed_draft_tokens", 0))
    reasoning_ok = bool(normalized["on"].thinking) if enabled else all(
        not content.thinking for content in normalized.values()
    )
    passed = bool(
        exact and normalized_exact and marker_free and reasoning_ok
        and off_proposals == 0 and proposed > 0 and accepted > 0
    )
    result = {
        "schema": "ds4_thinking_toggle_gate_v1",
        "pass": passed,
        "thinking": enabled,
        "language": args.language,
        "request": payload,
        "direct": {
            "token_ids": direct_ids,
            "normalized": normalized["direct"].__dict__,
            "finish": ui.response_finish(direct),
            "raw": direct,
        },
        "dflash_off": {
            "token_ids": off_ids,
            "normalized": normalized["off"].__dict__,
            "finish": ui.response_finish(off),
            "proposed": off_proposals,
            "raw": off,
        },
        "dflash_on": {
            "token_ids": on_ids,
            "normalized": normalized["on"].__dict__,
            "finish": ui.response_finish(dflash),
            "proposed": proposed,
            "accepted": accepted,
            "prefill_raw": prefill,
            "continuation_request": continuation_request,
            "raw": dflash,
        },
        "assertions": {
            "native_token_exact": exact,
            "normalized_exact": normalized_exact,
            "marker_free": marker_free,
            "reasoning_state_correct": reasoning_ok,
            "off_zero_proposals": off_proposals == 0,
            "on_native_proposals": proposed > 0,
            "on_native_acceptance": accepted > 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pass": passed, "thinking": enabled, "token_ids": direct_ids,
        "proposed": proposed, "accepted": accepted,
        "normalized": normalized["on"].__dict__,
        "finish": ui.response_finish(dflash),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
