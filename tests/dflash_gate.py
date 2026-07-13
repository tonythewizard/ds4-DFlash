#!/usr/bin/env python3
"""Bounded token-exactness and native DFlash acceptance gate."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    "Ciao! Presentati in modo conciso.",
    "Spiega in italiano tecnico la differenza tra mutex e semaforo.",
    "Explain why deterministic tests matter in distributed systems.",
    "Scrivi una funzione Python che calcoli il massimo comun divisore.",
    "Continua la sequenza e spiega il criterio: 2, 3, 5, 7, 11; simboli: !?.,",
    "Riassumi in tre punti perché una cache deve avere una politica di invalidazione.",
    "Translate into English: La correttezza viene prima della velocità.",
    "Leggi il seguente contesto e produci una sintesi tecnica: "
    + "Un sistema di inferenza deterministico deve mantenere invariati prompt, template, "
      "tokenizzazione, seed e parametri di campionamento. " * 18,
]


def post(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    if not isinstance(body, dict):
        raise RuntimeError("response is not a JSON object")
    return body, time.perf_counter() - started


def runtime_tokens(body: dict[str, Any]) -> list[int]:
    runtime = body.get("ds4_runtime") or {}
    tokens = runtime.get("completion_token_ids")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"native completion_token_ids missing: {body}")
    return tokens


def chat_payload(prompt: str, max_tokens: int, temperature: float, seed: int) -> dict[str, Any]:
    return {
        "model": "ds4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False,
        "thinking": False,
        "ds4_return_runtime_metrics": True,
        "ds4_return_token_ids": True,
    }


def reset(server: str, timeout: float) -> None:
    body, _ = post(server + "/v1/ds4/deepspec_reset_sample", {}, min(timeout, 30))
    if body.get("ok") is not True:
        raise RuntimeError(f"reset failed: {body}")


def extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message.get("content"), str):
            return message["content"]
    for key in ("text", "content"):
        if isinstance(body.get(key), str):
            return body[key]
    return ""


def finish_reason(body: dict[str, Any]) -> Any:
    choices = body.get("choices") or []
    return choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else body.get("finish_reason")


def same_usage(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return all((left.get("usage") or {}).get(key) == (right.get("usage") or {}).get(key) for key in keys)


def cycle_records(body: dict[str, Any]) -> list[dict[str, Any]]:
    draft_counts = body.get("cycle_draft_token_counts") or []
    commit_counts = body.get("cycle_committed_tokens") or []
    accepted = body.get("cycle_accepted_draft_tokens") or []
    positions = body.get("cycle_positions") or []
    committed = body.get("committed_token_ids") or []
    timing = body.get("timing_ms") or {}
    sidecar_times = timing.get("cycle_sidecar") or []
    verifier_times = timing.get("cycle_verifier") or []
    total_times = timing.get("cycle_total") or []
    proposal = (body.get("proposal_debug") or {}).get("token_ids") or []
    result: list[dict[str, Any]] = []
    proposal_offset = 0
    commit_offset = 0
    for index, count in enumerate(draft_counts):
        commit_count = commit_counts[index]
        result.append({
            "cycle": index,
            "position": positions[index],
            "proposal_token_ids": proposal[proposal_offset:proposal_offset + count],
            "proposed": count,
            "accepted": accepted[index],
            "committed_token_ids": committed[commit_offset:commit_offset + commit_count],
            "fallback_reason": None if accepted[index] == count else "native_verifier_rejection",
            "sidecar_ms": sidecar_times[index] if index < len(sidecar_times) else None,
            "verifier_ms": verifier_times[index] if index < len(verifier_times) else None,
            "total_ms": total_times[index] if index < len(total_times) else None,
            "proposal_mode": body.get("proposal_mode", "target_assisted_correctness_first"),
            "checkpoint_sha256": body.get("checkpoint_sha256"),
            "vocab_sha256": body.get("vocab_sha256"),
        })
        proposal_offset += count
        commit_offset += commit_count
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--proxy", default="http://127.0.0.1:8081")
    parser.add_argument("--mode", choices=("off", "on", "both"), default="both")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--skip-seeded-non-greedy", action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(1, len(PROMPTS) + 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "ds4_dflash_gate_v1",
        "config": {"server": args.server, "proxy": args.proxy, "mode": args.mode,
                   "max_tokens": args.max_tokens, "prefill_tokens": args.prefill_tokens,
                   "timeout": args.timeout},
        "cases": [],
    }
    off_exact = True
    on_exact = True
    proposed = 0
    accepted = 0
    prompt_cases = list(enumerate(PROMPTS))
    if args.case_index is not None:
        prompt_cases = [prompt_cases[args.case_index - 1]]
    for index, prompt in prompt_cases:
        seed = 260713 + index
        payload = chat_payload(prompt, args.max_tokens, 0.0, seed)
        reset(args.server, args.timeout)
        direct, direct_wall = post(args.server + "/v1/chat/completions", payload, args.timeout)
        direct_ids = runtime_tokens(direct)
        case: dict[str, Any] = {
            "name": f"greedy_{index + 1}", "prompt": prompt,
            "direct": {"token_ids": direct_ids, "text": extract_text(direct),
                       "usage": direct.get("usage"), "runtime": direct.get("ds4_runtime"),
                       "finish_reason": finish_reason(direct), "wall_seconds": direct_wall},
        }
        if args.mode in {"off", "both"}:
            reset(args.server, args.timeout)
            off, off_wall = post(args.proxy + "/v1/chat/completions", payload, args.timeout)
            off_ids = runtime_tokens(off)
            exact = (off_ids == direct_ids and extract_text(off) == extract_text(direct)
                     and same_usage(direct, off) and finish_reason(direct) == finish_reason(off))
            off_exact = off_exact and exact
            case["off"] = {"token_ids": off_ids, "text": extract_text(off),
                           "usage": off.get("usage"), "runtime": off.get("ds4_runtime"),
                           "finish_reason": finish_reason(off), "wall_seconds": off_wall, "exact": exact}
        if args.mode in {"on", "both"}:
            reset(args.server, args.timeout)
            prefill_payload = chat_payload(prompt, args.prefill_tokens, 0.0, seed)
            prefill, prefill_wall = post(args.server + "/v1/chat/completions", prefill_payload, args.timeout)
            prefill_ids = runtime_tokens(prefill)
            remaining = max(1, args.max_tokens - len(prefill_ids))
            dflash, on_wall = post(args.server + "/v1/ds4/deepspec_generate_dflash", {
                "max_tokens": remaining, "temperature": 0.0, "seed": seed,
                "spec_top_k": 8, "ds4_return_runtime_metrics": True,
                "ds4_return_token_ids": True,
            }, args.timeout)
            dflash_ids = runtime_tokens(dflash)
            on_ids = prefill_ids + dflash_ids
            exact = on_ids == direct_ids
            on_exact = on_exact and exact
            proposed += int(dflash.get("total_draft_token_count", 0))
            accepted += int(dflash.get("accepted_draft_tokens", 0))
            case["on"] = {"token_ids": on_ids, "text": extract_text(prefill) + extract_text(dflash),
                          "prefill_token_ids": prefill_ids, "dflash": dflash,
                          "cycles_detail": cycle_records(dflash), "prefill_wall_seconds": prefill_wall,
                          "generation_wall_seconds": on_wall, "exact": exact}
        result["cases"].append(case)

    seeded_cases = 0
    if args.mode in {"off", "both"} and not args.skip_seeded_non_greedy and args.case_index is None:
        for index, prompt in enumerate(PROMPTS[:2]):
            seed = 7260713 + index
            payload = chat_payload(prompt, args.max_tokens, 0.7, seed)
            reset(args.server, args.timeout)
            direct, direct_wall = post(args.server + "/v1/chat/completions", payload, args.timeout)
            reset(args.server, args.timeout)
            off, off_wall = post(args.proxy + "/v1/chat/completions", payload, args.timeout)
            direct_ids, off_ids = runtime_tokens(direct), runtime_tokens(off)
            exact = (off_ids == direct_ids and extract_text(off) == extract_text(direct)
                     and same_usage(direct, off) and finish_reason(direct) == finish_reason(off))
            off_exact = off_exact and exact
            seeded_cases += 1
            result["cases"].append({
                "name": f"seeded_non_greedy_{index + 1}", "prompt": prompt,
                "temperature": 0.7, "seed": seed,
                "direct": {"token_ids": direct_ids, "text": extract_text(direct),
                           "usage": direct.get("usage"), "finish_reason": finish_reason(direct),
                           "wall_seconds": direct_wall},
                "off": {"token_ids": off_ids, "text": extract_text(off),
                        "usage": off.get("usage"), "finish_reason": finish_reason(off),
                        "wall_seconds": off_wall, "exact": exact},
            })

    rate = accepted / proposed if proposed else 0.0
    result["summary"] = {
        "greedy_cases": len(prompt_cases), "seeded_non_greedy_cases": seeded_cases, "off_exact": off_exact,
        "on_exact": on_exact, "proposed": proposed, "accepted": accepted,
        "acceptance_rate": rate,
        "off_gate_pass": off_exact if args.mode in {"off", "both"} else None,
        "on_gate_pass": (on_exact and proposed >= 128 and accepted >= 64 and rate >= 0.5)
        if args.mode in {"on", "both"} else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passed = (args.mode == "off" and off_exact) or (args.mode == "on" and result["summary"]["on_gate_pass"]) or (
        args.mode == "both" and off_exact and result["summary"]["on_gate_pass"])
    print(json.dumps(result["summary"], indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
