# DS4 runtime token provenance

Non-streaming OpenAI chat/completion requests and the internal DFlash endpoint
support two request-level opt-in flags:

```json
{
  "ds4_return_runtime_metrics": true,
  "ds4_return_token_ids": true
}
```

Without either flag, the response is unchanged. With either flag, the response
contains `ds4_runtime` using schema `ds4_runtime_generation_v1`. Token IDs are
recorded at the native commit point; they are never reconstructed from text.
`completion_token_count` is therefore equal to the length of
`completion_token_ids`.

`ttft_ms` is measured from the start of valid request processing to the first
committed completion token. `decode_ms` spans the first through last committed
token. `generation_tokens_per_second` is `(completion_token_count - 1)` divided
by `decode_ms` in seconds.

Target-only reports zero proposed/accepted draft tokens and zero cycles.
Speculative generation uses the same schema; its `completion_token_ids` is
serialized from the same native array as the legacy `committed_token_ids`.

## DFlash correctness modes

`temperature: 0` selects the deterministic verifier contract. Each proposed
token is compared with the target model's native argmax. A match is counted and
committed as an accepted draft token; a mismatch commits the target argmax.
Consequently, rejection cannot change the target-only greedy sequence.

The external sidecar may run in `draft_model` mode or in the explicitly named
`target_assisted_correctness_first` mode. The latter projects the captured
`target_last_hidden` through the fixed vocabulary head and proposes one token
per cycle. It is a slow correctness mode, not an acceleration: every proposal
still passes through DS4's native verifier and native commit path. Set
`DS4_DFLASH_PROPOSAL_MODE=draft` to return to the learned block draft path.
There is no verifier override or oracle-accept environment switch.

The local continual launcher defaults to DFlash OFF at the client/UI layer.
Its proxy capture path is also opt-in (`DS4_CAPTURE_ENABLED=1`), because an
asynchronous capture/reset on the shared target session would violate the OFF
bypass contract. OFF forwards exactly one request and never calls the proposal
endpoint.
When DFlash is enabled, its server response reports `proposal_mode`, checkpoint
and vocabulary hashes, per-cycle counts and timings. The verifier never reads
or changes the active checkpoint during a request.

On memory-constrained CUDA/SSD-streaming deployments, the correctness profile
sets `DS4_CUDA_STREAMING_EXPERT_CACHE_N=0`. A dynamically resized expert cache
can otherwise make a long generation split across many endpoint calls follow a
different numerical routing frontier than one uninterrupted direct request.
This setting intentionally trades throughput for repeatable token identity.

For deployments that install the continual wrappers on `PATH`:

```sh
ds4-continual start
ds4-ui
```

The UI starts in OFF mode. Use `/dflash on` for the explicit correctness-first
path, `/dflash auto` for policy-controlled use, and `/dflash off` to return to
the direct target path. Check process and endpoint state with
`ds4-continual status`.

## Reproducible CUDA build

For an RTX 3070 Ti (compute capability 8.6):

```sh
CUDA_ARCH=sm_86 JOBS=2 scripts/build_cuda.sh build-cuda
```

The script detects `CUDA_HOME` from `nvcc`, builds in a separate directory, and
checks that the resulting server links CUDA runtime and cuBLAS. Override
`CUDA_ARCH`, `CUDA_HOME`, `JOBS`, or `BUILD_TIMEOUT` for another machine.

## Product gates

With one stack already running, execute:

```sh
python tests/dflash_gate.py --mode off --output off.json
python tests/dflash_gate.py --mode on --output on.json
```

The OFF gate compares native token IDs, decoded text, usage counts and stop
reason through the direct and proxy paths for eight greedy prompts plus two
seeded non-greedy checks. The ON gate compares the concatenated
prefill/native-DFlash commit sequence against direct target generation and
requires at least 128 proposals, 64 native acceptances, and 50% acceptance.

Streaming token provenance is not exposed in this version: benchmark clients
must use `stream: false`. This avoids duplicating the complete ID list in SSE
chunks. A future final-event schema may close this explicitly tracked gap.
