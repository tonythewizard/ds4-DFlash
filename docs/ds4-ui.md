# DS4 terminal UI

`scripts/ds4_ui.py` is the dependency-free terminal client for the continual
runtime. Install it on the command path with a symlink, or run it directly:

```sh
ln -s "$(pwd)/scripts/ds4_ui.py" ~/.local/bin/ds4-ui
ds4-ui
```

The client starts in English with DFlash OFF. Its important commands are:

```text
/dflash on | /dflash off | /dflash auto
/language | /language en | /language it
/reset | /stats | /status | /help
```

`/language` changes both the interface and one canonical language system
instruction in subsequent request payloads. It does not clear history. English
aliases are `en` and `english`; Italian aliases are `it`, `italian`, and
`italiano` (case-insensitive).

Every completed assistant turn has a separate THINKING/RAGIONAMENTO box. The
parser first uses structured `reasoning`, `reasoning_content`, or `thinking`
fields, then supports legacy `<think>...</think>` text with a carry buffer for
tags split across streaming chunks. Raw tags never enter the displayed final
answer. An empty box explicitly says that no thinking stream was received.

The prefill display is derived only from backend prompt metrics:

```text
prompt_tokens / (prompt_processing_ms / 1000)
```

If prompt processing time is missing or unusable it displays `N/A`; request wall
time is never substituted. Normal, DFlash, AUTO, and fallback paths use the same
normalizer and formatter. Generation throughput remains wall-time based as in
the previous client.

Useful environment overrides for a non-default continual installation are
`DS4_CONTINUAL_ROOT` and `DS4_CONTINUAL_LAUNCHER`. Set `NO_COLOR=1` to disable
the cyan thinking panel. `--stream` enables OpenAI-compatible SSE for normal and
fallback replies; DFlash's current synchronous endpoint is still parsed through
the same state machine when its completed response arrives.

Run the UI tests and bounded live-stack smoke with:

```sh
python3 -m unittest -v tests.test_ds4_ui
python3 tests/ds4_ui_runtime_smoke.py \
  --output /tmp/ds4-ui-smoke.json \
  --transcript /tmp/ds4-ui-transcript.txt
```
