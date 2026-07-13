# ds4-DFlash

**A correctness-first DS4 fork with native dFlash speculative decoding, CUDA low-memory inference, SSD-backed model streaming, DeepSpec draft integration, continual-learning tooling, and a bilingual terminal UI.**

> **Validation status**
>
> The complete end-to-end stack is currently validated only on **CUDA `sm_86`**, using an NVIDIA GeForce RTX 3070 Ti.
>
> The repository still contains the upstream Metal, ROCm, and CPU backends, but the ds4-DFlash speculative-decoding and continual-runtime gates have **not yet been validated on those backends**. Apple Silicon, including M3 systems, and ROCm targets must therefore be treated as experimental until backend-specific exactness, acceptance, build, and runtime tests are completed.

---

## Overview

ds4-DFlash extends DS4 with a practical speculative-decoding and continual-learning workflow designed for local inference on constrained hardware.

The project focuses on four goals:

1. preserve exact target-model behavior when dFlash is disabled;
2. add native dFlash proposal, verification, acceptance, fallback, and commit paths;
3. run the target model efficiently on consumer CUDA hardware with SSD streaming and bounded GPU-memory use;
4. support real-use capture, short supervised draft training, atomic checkpoint activation, rollback, and a usable terminal interface.

The current implementation is correctness-first. dFlash can reach high acceptance while still being slower than stock generation because verifier, sidecar, probability-transfer, and residual-sampling overheads remain performance bottlenecks.

---

## Backend status

| Backend | Core code present | ds4-DFlash end-to-end validation |
|---|---:|---:|
| CUDA `sm_86` | Yes | **Validated** |
| CUDA other architectures | Yes | Not yet validated |
| Metal / Apple Silicon | Yes | Not yet validated |
| Apple M3 | Expected to build through Metal, but not yet validated | Not yet validated |
| ROCm | Yes | Not yet validated |
| CPU | Yes | Not yet validated as a full dFlash stack |

Validated CUDA platform:

```text
GPU: NVIDIA GeForce RTX 3070 Ti
CUDA architecture: sm_86
Target backend: CUDA
Draft sidecar: CPU / FP32
```

The phrase “supported” in this README refers only to code availability unless explicitly marked as validated.

---

## Main features

### Native dFlash speculative decoding

- DeepSpec/dSpark draft sidecar proposes draft tokens.
- DS4 computes target probabilities.
- Native dFlash verification decides accepted and rejected draft tokens.
- Accepted tokens are committed through the normal target session path.
- Rejected proposals fall back safely to target generation.
- No oracle acceptance is used in the validated runtime path.
- Accepted-token counters originate from native verifier and commit behavior.

Supported runtime modes:

```text
/dflash off
/dflash on
/dflash auto
```

### Exactness guarantees

With dFlash disabled, the speculative path is bypassed.

Required OFF-mode invariants:

```text
same serialized request
same prompt and system messages
same seed
same sampling parameters
same reset/session state
same committed token IDs
same decoded output
zero speculative proposals
```

Required ON-mode invariant for deterministic test requests:

```text
final committed token IDs == stock target token IDs
```

Current validated regression results:

| Gate | Result |
|---|---:|
| dFlash OFF greedy exactness | 8/8 token-exact |
| dFlash OFF speculative proposals | 0 |
| dFlash ON final exactness | 8/8 token-exact |
| dFlash ON native acceptance baseline | 223/223 accepted |
| Recent bounded ON smoke | 3/3 accepted and final-exact |

These values describe the included validation workloads and must not be interpreted as a universal acceptance rate or speed claim.

---

## CUDA and low-memory inference

The validated ds4-DFlash runtime includes:

- CUDA target inference;
- explicit `sm_86` build validation;
- SSD-backed model streaming;
- direct-model CUDA access;
- low-memory expert loading;
- VRAM-aware routed-expert cache limits;
- configurable GPU-memory reserve;
- cache-disable modes for correctness diagnostics;
- protection against invalid selected-expert IDs;
- safer selected-expert asynchronous paths;
- startup logic designed to avoid duplicate target or sidecar loads.

Typical build:

```bash
export CUDA_HOME=/opt/cuda

make -B cuda \
  CUDA_HOME="$CUDA_HOME" \
  CUDA_ARCH=sm_86
```

Expected binary:

```text
./ds4-server
```

Example direct launch:

```bash
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}

./ds4-server \
  --cuda \
  --ctx 2048 \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Example SSD-streaming launch:

```bash
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}

./ds4-server \
  --cuda \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ctx 2048 \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Correctness-first diagnostic settings used during development may include:

```bash
export DS4_CUDA_DIRECT_MODEL=1
export DS4_CUDA_NO_FD_CACHE=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export NO_FD_CACHE=1
export NO_Q8_F16_CACHE=1
```

These are not guaranteed to be optimal on every machine. Optional caches should be enabled only after token-exact regression tests on the intended model and hardware.

---

## DeepSpec integration

The draft system uses selected target hidden states and a small DeepSpec/dSpark model.

Current validated draft contract:

```text
vocab_size:         129280
hidden_size:        4096
target_layer_ids:   [0, 8, 16, 24, 32, 40]
num_target_layers:  43
draft layers:       1
block_size:         7
mask_token_id:      128799
tied output:        false
```

Captured training/runtime artifacts can include:

- input token IDs;
- target hidden states;
- final target hidden states;
- masks;
- sequence metadata;
- runtime provenance.

### Corrected vocabulary extraction

The draft model uses separate target embedding and output-head matrices.

A critical extraction rule is:

> If GGUF dequantization already returns the physical `[vocab, hidden]` tensor layout, do not reshape it again merely because a logical tensor shape is reported in reversed order.

A destructive reshape can preserve aggregate statistics while silently scrambling token-row identity.

The corrected extraction path:

- preserves physical row order;
- keeps `token_embd.weight` and `output.weight` separate;
- supports untied output weights;
- verifies target-row alignment against captured final hidden states and target-server argmax behavior before training.

---

## Practical continual-learning stack

The development stack uses four long-running components:

```text
Terminal UI / API client
          |
          v
Capture-aware proxy :8081
          |
          v
DS4 target server :8080 <----> DeepSpec draft sidecar :8091
          |
          v
Collector / continual buffer
```

Components:

- **DS4 target server**
  - CUDA target inference;
  - target probabilities;
  - native dFlash verification;
  - native commit and fallback paths.

- **DeepSpec draft sidecar**
  - loads the active trainable-only draft checkpoint;
  - reads captured frontier data;
  - produces draft token IDs and draft probabilities.

- **Capture-aware proxy**
  - fronts the public API;
  - can capture real-use training records;
  - keeps normal generation available when dFlash is disabled.

- **Collector**
  - imports and deduplicates captures;
  - prepares bounded training inputs.

- **Launcher**
  - starts, stops, restarts, and reports all components;
  - avoids duplicate server/sidecar processes.

Operational workflow:

```text
real-use capture
-> short bounded daytime training
-> short runtime smoke test
-> atomic checkpoint activation
-> rollback on failure
```

The project intentionally favors practical continual learning over certification-style freeze, sealing, attestation, or master-validation pipelines.

---

## Atomic checkpoint activation

A runtime checkpoint directory should contain at least:

```text
manifest.json
trainable_model.safetensors
config.json
```

Recommended activation sequence:

1. stop the runtime stack;
2. validate checkpoint files and manifest;
3. record the current active checkpoint as rollback;
4. atomically replace the active checkpoint symlink;
5. restart the stack;
6. run a bounded exactness and dFlash smoke test;
7. roll back immediately on failure.

Never overwrite a mapped checkpoint file in place and never switch the active checkpoint during an in-flight request.

---

## Terminal UI

The ds4-DFlash terminal UI includes:

- English as the default language;
- Italian language support;
- runtime language switching;
- dFlash mode controls;
- localized help and status messages;
- prompt, generation, and elapsed-time metrics;
- native accepted/proposed token counters;
- active-checkpoint and runtime status;
- separate reasoning and final-answer rendering.

Commands:

```text
/help
/reset
/dflash off
/dflash on
/dflash auto
/language
/language en
/language it
```

### Language handling

Fresh TUI processes default to English.

English aliases:

```text
en
english
```

Italian aliases:

```text
it
italian
italiano
```

The selected language controls:

- UI messages;
- command confirmations;
- help;
- thinking-panel labels;
- the language instruction inserted into future model requests.

### Thinking panel

Reasoning is always displayed separately from the final answer.

English:

```text
THINKING
```

Italian:

```text
RAGIONAMENTO
```

The parser:

- prefers structured reasoning fields;
- supports `<think>...</think>`;
- handles tags split across chunks;
- handles incomplete tags;
- handles stray closing tags;
- prevents raw tags from leaking into the final answer;
- keeps final content separate from reasoning;
- disables color cleanly when output is not a TTY or `NO_COLOR` is set.

### Prefill metrics

The TUI computes prefill throughput only when valid positive timing data is available:

```text
prefill_tok_s = prompt_tokens / (prompt_processing_ms / 1000.0)
```

When prompt-processing time is missing or invalid, the UI shows:

```text
N/A
```

It never fabricates `0.00 tok/s` from absent timing data.

Current UI/parser regression result:

```text
20/20 tests PASS
```

---

## API endpoints

The server exposes several compatible API styles.

Main endpoints:

```text
POST /v1/chat/completions
POST /v1/responses
POST /v1/completions
POST /v1/messages
GET  /v1/models
```

Experimental ds4-DFlash and DeepSpec endpoints:

```text
POST /v1/ds4/deepspec_reset_sample
POST /v1/ds4/deepspec_dump_sequence
POST /v1/ds4/deepspec_generate_dflash
POST /v1/ds4/spec_verify_dflash
POST /v1/ds4/spec_target_logprobs
```

Experimental schemas may change during development.

Basic API smoke:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "ds4",
    "messages": [
      {
        "role": "user",
        "content": "Explain speculative decoding in one sentence."
      }
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

---

## Runtime ports

Default development ports:

| Component | Address |
|---|---|
| DS4 target server | `127.0.0.1:8080` |
| Capture-aware proxy | `127.0.0.1:8081` |
| DeepSpec draft sidecar | `127.0.0.1:8091` |

The TUI normally connects to:

```text
http://127.0.0.1:8081/v1
```

---

## Validation policy

Before publishing a build, run:

- full CUDA build;
- CUDA runtime smoke;
- DFlash OFF direct-vs-proxy token exactness;
- zero-proposal assertion while OFF;
- DFlash ON final-token exactness;
- native accepted/proposed counter checks;
- TUI language tests;
- thinking-parser tests;
- prefill-metric tests;
- process and endpoint health checks;
- OOM/Xid inspection.

The currently validated platform is only CUDA `sm_86`.

### Metal and Apple Silicon

The repository retains the native Metal backend and may build on Apple Silicon, including M3 systems. However, ds4-DFlash must not be described as validated on M3 until the following tests pass on real hardware:

- Metal build;
- target-runtime smoke;
- DFlash OFF token exactness;
- DFlash ON final exactness;
- native acceptance;
- hidden-state capture;
- verifier and commit behavior;
- sidecar compatibility on arm64;
- CPU or MPS sidecar runtime;
- unified-memory pressure tests;
- macOS launcher and TUI integration.

### ROCm

The repository retains ROCm-related code, but no ds4-DFlash end-to-end validation has been completed on AMD hardware. ROCm support remains experimental until equivalent build, exactness, acceptance, and runtime tests pass.

### CPU

The CPU backend remains useful for reference and fallback work, but the complete dFlash/continual stack has not been validated as a release target on CPU-only systems.

---

## Known limitations

- Only CUDA `sm_86` is currently validated end-to-end.
- Metal, Apple M3, ROCm, other CUDA architectures, and CPU-only deployment remain unverified.
- The current dFlash path may be slower than normal decoding.
- The validated draft sidecar runs on CPU and adds latency.
- Draft-probability transport and residual verification remain performance bottlenecks.
- Acceptance depends on prompt distribution and training data.
- The continual-learning tools are still developer-oriented.
- Some local launchers may contain machine-specific paths and should be parameterized before general distribution.
- Model weights, GGUFs, draft checkpoints, captured hidden states, and training data are not distributed.

---

## Troubleshooting

### Server or sidecar fails to start

```bash
ss -ltnp | grep -E ':(8080|8081|8091)\b'
nvidia-smi
journalctl -k --since '10 minutes ago' \
  | grep -Ei 'oom|out of memory|killed process|nvrm|xid'
```

Do not start a second full server or sidecar while the first stack is loaded.

### DFlash accepts zero tokens

Check:

- active checkpoint;
- corrected vocabulary-weight file;
- vocabulary size;
- target-layer IDs;
- frontier anchor;
- target versus draft probability;
- draft rank of the real target token;
- request distribution versus training captures.

### Output differs with DFlash OFF

Treat this as a correctness failure.

Compare:

- serialized messages;
- language system instruction;
- reset state;
- seed;
- temperature and sampling parameters;
- direct target route;
- proxy route;
- token IDs, not only decoded text.

### Prefill shows `N/A`

The backend did not provide a valid positive prompt-processing duration. `N/A` is the correct display and is preferable to a false `0.00 tok/s`.

### Raw thinking tags appear

Run the thinking-parser tests and verify that the installed launcher resolves to the updated canonical TUI source.

### OOM

- stop duplicate processes;
- lower context length;
- reduce or disable expert caches;
- preserve a GPU-memory reserve;
- keep the sidecar on CPU when the target consumes most VRAM;
- avoid loading multiple copies of the full vocabulary matrices.

---

## Repository hygiene

Before pushing:

```bash
git status --short
git diff --check
git ls-files | grep -E '\.(gguf|safetensors)$' && exit 1 || true
```

Do not commit:

```text
*.gguf
*.safetensors
raw hidden-state captures
training caches
runtime logs
credentials
tokens
machine-local checkpoints
personal absolute-path configuration
```

---

## Roadmap

### Portability

- validate CUDA architectures beyond `sm_86`;
- validate Metal on Apple Silicon, including M3;
- add a macOS launcher;
- validate sidecar execution on CPU and MPS;
- validate ROCm on real AMD hardware;
- add backend-specific exactness suites.

### Performance

- remove temporary-file draft-probability transport;
- evaluate shared-memory or binary-socket transport;
- reduce native verifier overhead;
- profile residual-sampling cost;
- evaluate GPU-resident draft inference where memory permits;
- improve routed-expert locality and cache behavior.

### Packaging

- replace personal paths with configuration;
- add example configs;
- add reproducible build scripts;
- add CI for parser, metric, and CPU-safe unit tests;
- package exactness and acceptance tests behind one command;
- attach validation artifacts to releases.

---

## Upstream attribution and license

ds4-DFlash is a development fork of DS4/DwarfStar.

Preserve:

- original Git history;
- copyright notices;
- attribution;
- license files;
- third-party notices.

This README does not redefine the upstream license. Review the repository `LICENSE` and all upstream notices before redistribution.

DeepSpec/dSpark components may have separate licensing requirements. Preserve and verify those notices before vendoring or distribution.

---

## Development baseline

Known-good development commits at the time this README was prepared:

```text
DFlash/CUDA baseline: 7fc9def
UI/language/thinking:  de589f317f6ecf84a2f27fad4547440f98e0a1c7
```

Current development branch:

```text
codex/ds4-ui-language-thinking-20260713
```

Final fork name:

```text
ds4-DFlash
```
