#!/usr/bin/env python3
"""One-shot bounded target-only smoke for native token provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def post(url: str, value: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(value).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readiness-timeout", type=int, default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stdout = (args.output_dir / "server.stdout.log").open("w")
    stderr = (args.output_dir / "server.stderr.log").open("w")
    command = [str(args.server), "--cuda", "--ssd-streaming", "--ssd-streaming-cold",
               "--ctx", "2048", "-m", str(args.model), "--host", "127.0.0.1",
               "--port", "8080"]
    env = dict(os.environ)
    env.update({"DS4_CUDA_DIRECT_MODEL": "1", "DS4_CUDA_NO_FD_CACHE": "1",
                "DS4_CUDA_NO_Q8_F16_CACHE": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "2",
                "DS4_CUDA_STREAM_EXPERT_LAYER_BUDGET": "8"})
    process = subprocess.Popen(command, cwd=args.server.parent, env=env,
                               stdout=stdout, stderr=stderr, start_new_session=True)
    result = {"schema": "ds4_target_token_provenance_smoke_v1", "pid": process.pid,
              "pgid": os.getpgid(process.pid), "command": command, "retry_count": 0,
              "started_at": time.time(), "status": "inconclusive"}
    try:
        deadline = time.monotonic() + args.readiness_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                result["reason"] = f"server exited during readiness: {process.returncode}"
                break
            try:
                status, _ = post("http://127.0.0.1:8080/v1/ds4/deepspec_reset_sample", {}, 3)
                if status == 200:
                    payload = {"model": "ds4", "messages": [{"role": "user", "content":
                               "Rispondi con una sola parola: prova."}], "max_tokens": 4,
                               "temperature": 0.0001, "seed": 26001, "stream": False,
                               "ds4_return_runtime_metrics": True, "ds4_return_token_ids": True}
                    http_status, body = post("http://127.0.0.1:8080/v1/chat/completions", payload, 180)
                    runtime = body.get("ds4_runtime") or {}
                    ids = runtime.get("completion_token_ids")
                    completion = (body.get("usage") or {}).get("completion_tokens")
                    passed = (http_status == 200 and isinstance(ids, list) and
                              len(ids) == completion and runtime.get("token_id_provenance") ==
                              "native_commit_path" and all(float(runtime.get(key, -1)) >= 0 for key in
                              ("ttft_ms", "decode_ms", "total_generation_ms",
                               "generation_tokens_per_second")))
                    result.update({"status": "pass" if passed else "fail", "http_status": http_status,
                                   "response": body, "completion_token_count": completion,
                                   "token_ids_count": len(ids) if isinstance(ids, list) else None})
                    break
            except Exception as error:
                result["last_readiness_error"] = str(error)
                time.sleep(1)
        else:
            result["reason"] = "bounded readiness timeout"
    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL); process.wait(timeout=10)
        stdout.close(); stderr.close()
        result.update({"exit_code": process.returncode, "cleanup_complete": process.poll() is not None,
                       "finished_at": time.time(), "promotion_executed": False})
        for name in ("server.stdout.log", "server.stderr.log"):
            path = args.output_dir / name
            result[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
