# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`audio.cpp`: a ggml-based C++ inference runtime for 110+ local audio model families (TTS, ASR,
voice conversion, diarization, VAD, codecs, source separation, music generation). One runtime
behind a CLI, an HTTP server, an embedded WebUI, and a C ABI.

## Build

Linux/macOS helper script (preferred; writes to `build/linux-<backend>-<type>/`):

```bash
scripts/build_linux.sh --backend cpu|cuda|vulkan|hip --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend cuda --cuda-arch native --ccache --target audiocpp_cli
scripts/build_metal.sh                # macOS
```

`--ccache` is opt-in and matters: nothing caches compiler output by default, and a rebuild is
~14x faster with it. CUDA/Vulkan/HIP are mutually exclusive in this script.

Direct CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_cli
```

Binaries land in `<build-dir>/bin/`.

Windows: `make -f Makefile.windows cuda|cpu|vulkan [TARGET=audiocpp_cli] [JOBS=8]`, which wraps
`scripts/build_windows.ps1 -Preset windows-cuda-release`. Presets: `cmake --list-presets`.

Flags that matter most:

| Flag | Effect |
|---|---|
| `AUDIOCPP_MODEL_SET=full\|core\|custom` + `AUDIOCPP_MODELS=a,b,c` | Compile only the model families you need. **Use this for any single-model work** — a full build compiles ~110 model trees. |
| `ENGINE_ENABLE_CUDA` / `_VULKAN` / `_HIP` / `_METAL` | ggml backends. CPU is always built. |
| `ENGINE_ENABLE_NATIVE_CPU` (default ON) | Host-ISA CPU kernels. Set OFF if the assembler rejects a generated instruction (e.g. `vpdpbusd`). |
| `ENGINE_BUILD_TESTS` / `_EXTENDED_TESTS` / `_MODEL_TESTS` | Test gates, see below. |
| `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` | Builds `audiocpp_model_manager` + server download endpoints (pulls BoringSSL at configure). Required for WebUI downloads. |
| `AUDIOCPP_DEPLOYMENT_BUILD=ON` | Embeds `model_specs/*.json` into the binary so it runs outside a checkout. Implied by `--hip`. |
| `AUDIOCPP_BUILD_C_API=ON` | Builds the `audiocpp` shared library (C ABI). |

Run CI jobs locally in the same container images (needs docker): `ci/run-local.sh linux-cpu`,
`linux-vulkan`, `nix-cpu`, `nix-vulkan`, `all`.

## Test

```bash
cmake -S . -B build -DENGINE_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --parallel 4
ctest --test-dir build -R '^audio_dsp_test$' --output-on-failure     # single test
```

Three compile-time test gates, all OFF by default: `ENGINE_BUILD_TESTS` (framework unit tests +
server/CLI tests — the only ones CI runs), `ENGINE_BUILD_EXTENDED_TESTS` (parity probes, warm
benchmarks), `ENGINE_BUILD_MODEL_TESTS` (per-family tests needing real weights). A fourth,
`ENGINE_BUILD_WARMBENCH`, builds the warmbench helper binaries and is not part of `ctest`.

Model tests are registered only when the family is in `AUDIOCPP_LINKED_MODELS`, so the fast
loop for one model is:

```bash
cmake -S . -B build -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=<family>
```

No ctest labels and no test env vars: gating is compile-time plus
`set_tests_properties(... SKIP_RETURN_CODE 77)`, so a test that needs absent weights or a
missing backend reports `Not Run` rather than failing. Tests are plain C++ executables with
hand-written asserts (`tests/unittests/test_assert.h`), registered by `add_engine_unittest` /
`add_engine_warmbench` in the root `CMakeLists.txt` (there is no `tests/CMakeLists.txt`).
The sanitizer CI jobs compile `AUDIOCPP_MODELS=builtin_audio_utils`, so a model test (or a test
touching model config) must gate on `make_default_registry().supports_family(family)` at runtime,
never on link-time presence of the family's symbols.

There is **no** clang-format, clang-tidy, pre-commit, or format target in this repo. Match the
style of the file you are editing.

Before opening a PR, run the loader/catalog gate (first step of every CI job):

```bash
python3 tools/check_loader_catalog_sync.py --self-test && python3 tools/check_loader_catalog_sync.py
```

## Architecture

`external/ggml` is vendored as tracked files (not a submodule; override the location with
`AUDIOCPP_GGML_SOURCE_DIR`). The only submodule is
`external/audio.cpp-server-frontends` (optional in-process server transport modules; `git
submodule update --init external/audio.cpp-server-frontends` only if you use it). Other vendored
deps: `cJSON`, `cpp-httplib` (native-manager builds only), `libyaml`, `llama_tokenizer`,
`sentencepiece`.

Layering, bottom to top:

- `src/framework/**` + `include/engine/framework/**` — shared, model-agnostic runtime, compiled
  into the `engine_core` object library.
  - `core/` — ggml backend/device resolution (`backend.h`: `init_backend`, `compute_backend_graph`,
    `BackendType`), module base + `TensorShape`, execution context, shared weight cache.
  - `runtime/` — the registry, session interfaces, graph executor/optimizer, KV caches, task vocabulary.
  - `modules/` — reusable nn blocks (attention, conv, transformer, conformer, zipformer, ebranchformer,
    vocoders: vocos/hifigan/bigvgan/hift; speech encoders: whisper/wav2vec2/wavlm/s3-tokenizer;
    text encoders, speaker encoders, pitch extractors) + `modules/optimizations/` for fused/fast paths.
  - `codecs/` — audio tokenizers / VAE decoders reused across TTS and music models (fish_dac, snac,
    mimi, fsq, moss_audio, nemo_nano, oobleck, redae, Wan video VAE, mel-latent VAE).
  - `conditioners/` — cross-modal conditioners (OpenCLIP, CLAP, Synchformer, CoCa-MAE-ST, MusicGen style).
  - `decoders/` — TDT/CTC decoders and the MOSS delay decoder.
  - `audio/` (FFT/STFT/ISTFT, resample, Kaldi fbank, mel, mixing, WAV IO, and the model-based
    denoise/enhance utilities: RNNoise, DeepFilterNet2, ZipEnhancer, FlashSR, GTCRN),
    `tokenizers/`, `sampling/`, `text/`, `io/` (json/yaml/safetensors/binary/config),
    `assets/` (tensor sources, Torch `.bin`, LoRA, weight-norm folding), `model_spec/` (typed spec
    schema), `midi/`, `debug/` (trace/profiler), `package_manager/` (native downloads).
- `src/models/<family>/` + `include/engine/models/<family>/` — core release-surface models.
- `src/community_models/<family>/` + `include/engine/community_models/<family>/` — community
  ports, lighter review bar; they graduate into `src/models/` once validated and maintained.
- `app/**` — binaries. `app/cli`, `app/server`, `app/gguf`, `app/model_manager` are the
  executables; `app/common`, `app/streaming`, `app/workflow` are shared sources compiled into
  them (`app/workflow` is the `IAppPipeline` / `AppPipelineRegistry` fan-out layer behind the
  JSON pipeline feature; there is no separate pipeline binary).
- `include/audiocpp.h` + `src/capi/audiocpp.cpp` — opt-in C ABI facade over `engine::runtime`
  (opaque handles, `audiocpp_status` returns, no exceptions cross the boundary). Adding an entry
  point means updating the header **and** both export allowlists (`src/capi/audiocpp.map` for
  Linux, `src/capi/audiocpp.symbols` for Apple); `tests/capi/export_surface.py` fails otherwise.

Namespaces: framework and models use `engine::*` (`engine::models::<family>`,
`engine::community_models::<family>`); the app layer keeps the legacy `minitts::server` /
`minitts::app`.

Executables: `audiocpp_cli` (inference + pipelines), `audiocpp_server` (HTTP/OpenAI-compatible
API + embedded WebUI), `audiocpp_gguf` (safetensors/bin → GGUF converter, `--inspect`),
`audiocpp_model_manager` (only with the native-manager flag), `model_perf`.

WebUI: a SvelteKit/TS SPA in `webui/native/`. `npm run build` emits the single file
`webui/native/dist/index.html`, which is **committed** and embedded as a byte array
(`app/server/audiocpp_ui_asset.h.in`) into `audiocpp_server` at build time, along with the
demo voices. The server needs no Node at runtime. Rebuild with
`cd webui/native && npm ci && npm run build`, then rebuild the server; `npm run dev` proxies
`/health` and `/v1` to a server on :8080.

Server: `app/server/runtime.cpp` is the legacy, contribution-priority runtime (one loaded model +
one task session per model id, reused across requests). `parallel_runtime.cpp` /
`parallel_http.cpp` is the experimental multi-slot scheduler, opted into at startup with
`--parallel-jobs`, and explicitly **not** a contribution priority. `model_memory.cpp` estimates
resident bytes for the load guard; `busy_guard.h` stops requests piling behind a stuck model.

## Adding or changing a model — the sync rule

The most important repo invariant. A family lives in sync across four surfaces; CI enforces the
first three:

1. `CMakeLists.txt`: `audiocpp_add_model(<family> SOURCES ... INCLUDES <session.h> LOADERS
   engine::[community_models::]<family>::make_<family>_loader)`. This is the **only** registration
   point — CMake generates `model_registry_{includes,loaders}.inc`, which
   `src/framework/runtime/registry.cpp` includes. No self-registration, no dlopen.
2. `model_specs/<family>.json` — typed spec (`schema_version: 1`). Its `family` must equal the
   filename stem **and** `loader->family()`.
3. `README.md` supported-model table row.
4. `docs/model_licenses.md` row, plus `docs/models/<family>.md` or
   `docs/community_models/<family>.md`.

`tools/check_loader_catalog_sync.py` fails if 1/2/3 drift, if a spec `family` has no registered
loader, if a spec `packages[]` id is missing from `tools/model_manager_v2.py list --json`, or if a
spec `options[].depends_on` names an unknown option.

The factory name is a registration symbol, not the family id — the runtime contract is
`loader->family()` matching the spec `family`. Aliases are not implied.

Interfaces to implement (copy an existing family; `qwen3_asr`, `pocket_tts`, `silero_vad` are
compact references):

- `engine::runtime::IVoiceModelLoader` — `family()`, `can_load()`, `inspect()`, `load()`
- `engine::runtime::ILoadedVoiceModel` — `metadata()`, `capabilities()`, `create_task_session()`
- a session implementing `IOfflineVoiceTaskSession::run()` and/or `IStreamingVoiceTaskSession`
  (`start_stream`/`next_stream_event`/`finish_stream`, `process_audio_chunk`/`finalize`)

Model resolution: CLI `--family` → `registry.find_by_family()` → `loader->load()`. Always pass
`--family`; auto-discovery is slow.

## Run

```bash
build/bin/audiocpp_cli --task tts --family pocket_tts --model models/pocket-tts \
  --backend cuda --text "hello" --voice-ref assets/resources/sample.wav --out out.wav
build/bin/audiocpp_cli --task asr --family qwen3_asr --model <path> --audio in.wav
build/bin/audiocpp_cli --list-loaders --json          # machine-readable loader contract
build/bin/audiocpp_cli --inspect --family <f> --model <path>
build/bin/audiocpp_server --ui --ui-management --backend cuda    # http://127.0.0.1:8080
```

`--task` values: `gen tts clon vc svc asr align vad diar sep vdes midi spk`. Model options go
through `--load-option` / `--session-option` / `--request-option key=value`.

Model weights go in `models/` at the repo root (gitignored). Download with
`python3 tools/model_manager_v2.py install <package-id>` (reads `model_specs/*.json`), the native
`audiocpp_model_manager`, or the WebUI Models page. `silero_vad` and `marblenet_vad` ship bundled
under `assets/framework/models/`.

Useful CLI flags beyond the table above: `--out-format pcm16|pcm24|float32` for the WAV sample
format of `--out`/`--out-dir` (`float32` keeps inter-sample peaks unclipped), and
`--text-chunk-size <chars>` to split long TTS input — see the RK3588 section for why it is not
optional on `piper_tts`.

## Measured on RK3588 (Orange Pi 5 Plus: 8 big.LITTLE cores + Mali-G610)

Native aarch64 build (`scripts/build_linux.sh --backend cpu`, and `--backend vulkan` for the
Mali). Facts that only show up on that hardware:

- **There is no ggml NPU backend, and none is needed for the decoder.** ggml has no RKNN backend, so
  a full "audio.cpp on the NPU" is a ggml-backend port. What works per-stage is a model-local
  `librknnrt` module: see the RKNN speech-decoder sidecar bullet below.
- **Vulkan on the Mali-G610 builds, runs, and is 2.1x slower than CPU for TTS.** Measured on
  `kokoro_tts`. Do not reach for `--backend vulkan` to speed up TTS on an ARM SoC; on this class of
  GPU it is a correctness target, not a performance one.
- **`piper_tts` graph arena is 16 MiB and long text overflows it.** Without
  `--text-chunk-size`, 443 chars dies with `ggml_new_object: not enough space in the context's
  memory pool (needed 17287712, available 16777216)`. `model_specs/piper_tts.json` has no
  graph-size knob, so chunking is the only fix from the CLI. On Vulkan the same arena is the
  binding constraint sooner, because ggml allocates graph objects in it (on CPU the tensor payloads
  live outside the context): `kDurationGraphArenaBytes`/`kDecoderGraphArenaBytes` in
  `src/community_models/piper_tts/vits_runtime.cpp` need 32/64 MiB there. Upstream has not touched
  that file since 2026-09-29.
- CPU reference numbers on that board, for regression checks: `piper_tts` RTF 0.227 (443 chars,
  `--text-chunk-size 60`), `kokoro_tts` (iSTFTNet) RTF ≈1.17 at 91 chars and ≈1.26 at 443, with a
  fixed 4.88 s of weight-load/graph-build overhead per process (so a one-word phrase measures
  ~4.9 s of pure overhead).
- **`qwen3_tts` 0.6B q8_0 is vocoder-bound: RTF 3.61, and 60% of it is the speech decoder.**
  Per-token from `--log-file` TIMING (greedy decode, `--threads 4`, 300 Russian chars): speech
  tokenizer decoder **174.6 ms**, code predictor 62.9 ms, talker decode 36.5 ms; 13.4 CPU-sec per
  audio-sec, peak RSS 5.1 GB. Compare `kokoro_tts` (decoder 83% of runtime) and `piper_tts`
  (decoder 75%): on this board every heavy TTS family is decoder-bound, so decoder-side work is
  where all the headroom is.
- **`--session-option qwen3_tts.conv_weight_type=f16` is a free 16%: RTF 3.61 → 3.04 and
  −950 MB RSS.** The default is `f32`, i.e. the speech-tokenizer and speaker-encoder convolution
  weights are dequantized to fp32 in the graph (`conv_modules.cpp:346,438,477` cast weights to
  F32 on the GEMM fast paths), which costs both bandwidth and the fp16 matmul path.
  `speech_decoder_weight_type=f16` on top of it changes nothing — those matmul weights are
  already q8_0 in the shipped GGUF.
- **`--threads` above 4 is a cliff, not a tuning knob.** Code predictor per token: 62.9 ms at
  `--threads 4` (default), 80.4 at 2, **137.3 at 6, 140.4 at 8** — the CP step is ~75 tiny
  S=1 GEMV ops, and ggml's thread barrier waits for the 4×A55@1.8 GHz cores the extra threads
  land on. `taskset -c 4-7` (big cluster only) at `--threads 4` changes nothing (RTF 3.63), so
  the fix is the thread count, not affinity.
- **`qwen3_tts.perf_mode=flash_attention` is unreachable on the shipped packages.** The gate
  (`src/models/qwen3_tts/session.cpp:293-296`) requires a Q8_0 tensor in *both* weight sources,
  and `qwen3-tts-12hz-0.6b-base-q8_0.gguf` quantizes only the talker matmuls — the speech
  tokenizer is fp32 — so it fails with `supported only with Q8_0 GGUF weights` on a file named
  q8_0.
- **The NPU seam is per-stage, not per-backend.** Stages are already graph-isolated and each
  computes through `core::compute_graph(execution, graph, plan, label)` with its own
  `ExecutionContext` (`src/framework/core/execution_context.cpp:9`), so routing one stage to
  `librknnrt` needs a model-local module, not a ggml backend. Ready-made `.rknn` decoders for
  three families sit on that board: `/root/tts-rknn/vocoder/vocoder_stateful_{16,64b}_fp16.rknn`
  (this repo's `qwen3_tts` speech decoder, same 12 Hz/1920-samples-per-token contract, stateful
  conv+KV cache in graph I/O), `/root/paroli-models/decoder_ru.rknn` (VITS decoder, fixed
  `z [1,192,55]` window — paroli's chunking/crossfade has to come with it), and
  `/root/kokoro-server/artifacts/kokoro_decoder.rknn` (iSTFTNet generator → mel; its boundary is
  one stage earlier than this repo's `decode()`, which returns PCM, so the iSTFT head stays on
  CPU). Measured precedent for the same swap in those engines: decoder 5.1× faster on NPU
  (Kokoro), 12.5× RTF gain (Piper). Projected here: RTF 3.61 → 1.56 offline, 1.24 with
  streaming overlap.
- **The `qwen3_tts` RKNN decoder sidecar is implemented and measured on the board**
  (`src/models/qwen3_tts/rknn_speech_decoder.cpp`, selected through
  `--session-option qwen3_tts.exec_plan={"speech_decoder":{"impl":"rknn","artifacts":{"main":<stateful-64>,"first":<stateful-16>}}}`
  — the earlier `qwen3_tts.rknn_decoder*` options are retired; `piper_tts.exec_plan` carries the
  `decoder` component for the piper vocoder sidecar the same way; build with
  `-DENGINE_ENABLE_RKNN=ON
  -DENGINE_RKNN_INCLUDE_DIR=/usr/local/include -DENGINE_RKNN_LIBRARY=/usr/lib/librknnrt.so`).
  Offline (300 Russian chars, greedy, `conv_weight_type=f16`): RTF 3.01 → 2.17, peak RSS
  4200 → 2560 MB, decoder 130.1 → 64.1 ms/token (2.0×). Streaming (`--mode streaming`, same text):
  wall 151.2 → 57.7 s, decoder 113.5 → 19.9 s — the CPU streaming decoder re-voices the 25-token
  left context in every window (29 windows), the stateful NPU model voices each token once. With
  `rknn_decoder_first` the first chunk costs 880 ms (16 tokens + 25 priming on the B=16 model)
  instead of 3.4-3.8 s (a full 64-token group), i.e. TTFB 3.4 s → 0.9 s.
  Quality: offline CPU-vs-NPU SNR 27 dB with the dips only in near-silence (fp16 vocoder noise
  floor, `max_d` 3085/32767), NPU-B64-vs-NPU-B16 57 dB, no seam clicks (max step at the
  64-token boundaries 372 samples vs 4886 global). Streaming CPU-vs-NPU is 16 dB and no lag
  aligns it: the two streaming decoders have different conditioning semantics, not a shift.
  RKNN C-API facts this path depends on: `rknn_query(RKNN_QUERY_INPUT_ATTR).size` reports the
  *native* storage width (the fp16 model reports `cache_valid` size=2, `freqs_in` size=4096 for
  2048 fp32 elements), so with `pass_through=0` the `rknn_input.size` must be
  `n_elems * sizeof(declared_type)`, not `attr.size` — otherwise `rknn_inputs_set` fails with -5
  and `param input size(4096) < model input size(8192)`. `audio_codes` is INT64 with
  `pass_through=1` and does use `attr.size`.
- **The whole RKNN toolchain is pinned to SDK 2.3.2, converter and runtime alike** (measured
  2026-10-19 on 2.5.0 master): the 2.5.0 runtime runs every 2.3.2-compiled engine bit-identically
  but costs +37% `decode_ms` on the qwen3 stateful vocoder (all of it inside `rknn_run`, A/B/A
  reproducible; piper/inflect engines ±2-4%, noise), and the 2.5.0 converter floods
  `REGTASK: bit width exceeds limit` on fp16 builds (+8% engine size, byte-non-deterministic;
  2.3.2 is byte-deterministic). Engines are compiled per SoC and not portable between chips.
  Upstream informed: airockchip/rknn-toolkit2#582, #583. Tripwire for a future re-check: qwen3
  `decode_ms` per SDK release.
- **Engine producers live in `tools/rknn/`** — one `prepare_<family>_rknn_*.py` per family
  (qwen3 vocoder, piper vocoder, inflect_v2 decoder, kokoro generator, cosyvoice3 flow ladder),
  sharing `rknn_build.build_engine` (fixed-shape `load_onnx`+`build`) and the `rknn_build_log`
  gate (build-log REGTASK saturation above 100k lines fails the build). They need the PC-side
  `rknn-toolkit2` wheel in a dedicated venv (pins: `numpy<=1.26.4`, `onnx==1.16.1`); on Python
  3.12 that venv additionally needs `setuptools<81` and an `onnx/mapping.py` shim — see
  `docs/build/rockchip-npu.md`.
- **What will not speed this up on RK3588.** Mali/Vulkan (measured 2.1× slower), the code
  predictor on the NPU (measured 5.4× slower — NPU GEMV tops out near 8 GB/s of weight reads,
  the CPU gets 16-19 GB/s), and a full ggml-rknn backend (RKNN has no runtime graph builder, so
  every (batch, seq) tuple is a separate compiled engine and the split-boundary copies eat the
  win). The board's DRAM read ceiling is 26.4-29.4 GB/s (cache-bypassing STREAM triad, 4 A76
  threads, 804 MB working set; `sysbench memory` with a 64 MB buffer agrees, its default 1 MB
  buffer reports a fake 61 GB/s because it never leaves L3). The NPU sidecars stream weights at
  0.08-0.10 GB/s and the CP at 0.72 GB/s, i.e. 0.3-3% of that ceiling, so they are
  compute/scheduler/latency-bound and shrinking their weights buys little. The only component that
  reads its whole weight set per token is the CPU talker: 471.3 MB of Q8_0 matmuls + `codec_head`
  per token = 11.5-12.9 GB/s = 44-49% of the ceiling, i.e. below it, and bytes convert to time
  sublinearly (re-quantizing the 196 talker matmuls alone to Q4_K: −47% bytes, −23% talker ms,
  −10.7% RTF 2.820 → 2.519, −210 MB; Q6_K is a no-op, the dequant eats the −23% bytes). Stage
  overlap is the other lever.

## Contribution rules that bite

From `CONTRIBUTING.md` — enforce these when editing:

- New standalone model ports start under `community_models/`, not `src/models/`. Extend an
  existing family with a new variant rather than opening a duplicate port.
- Framework modules are high risk. Prefer additive work. A variant of an existing module goes in
  as a separate `xxxExp` module — never rewrite a module existing models depend on.
- Additive framework support in a model PR must be clearly scoped and opt-in; model-specific logic
  stays model-local. Non-additive framework changes are a separate PR first.
- Any PR touching runtime behavior, model implementations, backend logic, or shared framework
  code needs **actual runtime validation** in the description: exact build and run commands,
  model paths, generated artifacts, path/parity test results, backend tested, RTF/RSS/VRAM notes,
  known limitations. Code inspection is not accepted.
- Keep ≤3 PRs open at a time.

## Tooling

- `tools/check_loader_catalog_sync.py` — CI gate (stdlib only).
- `tools/check_moss_codec_contexts.py` — CI gate for MOSS codec attention contexts.
- `tools/model_manager_v2.py` — `list` / `info` / `sizes` / `install` / `installed` / `uninstall`
  (stdlib only; `huggingface_hub` optional for LFS sizes).
- `tools/model_manager_deprecated.py` — legacy checkpoint-conversion packages only.
- `tools/asr_wer.py` — WER of an ASR family against the four LibriSpeech clips in
  `assets/asr_validation/librispeech/`. Sanity check, not a benchmark.
- `tools/prepare_kokoro_gguf.py`, `tools/prepare_vibevoice_asr_streaming_gguf.py` — one-off GGUF prep.
- `tests/perf/model_perf.py` + `model_perf_cases.json` — perf harness (needs the `model_perf`
  binary and weights).
- `tests/warmbench.py` — warm-cache benchmark (needs `ENGINE_BUILD_WARMBENCH=ON`).
- `tests/audio/stft_torch_parity.py` — the only script needing torch.
- `tools/audiocpp_cli/` — CLI path-coverage test runner (`run_audiocpp_cli_path_tests.py` over the
  `*_cases.json` files) plus result-comparison scripts.
- `tools/community_models/convert_*.py` — per-family checkpoint-conversion scripts.
- `tools/rknn/prepare_*_rknn_*.py` — PC-side RKNN engine compilers per family (shared
  `rknn_build.py` / `rknn_build_log.py`), the only producers of the sidecar `.rknn` artefacts.
- `tools/streaming/measure_{tts,asr}_streaming_client.py` — streaming latency measurement clients.
- `tools/omtd/` — standalone C++ output-multimodal-decoder boundary (own `CMakeLists.txt`, not part
  of the main build).

## Docs

`docs/usage.md` (CLI), `app/server/README.md` (server), `docs/maintainers/loader_and_catalog.md`
(the full sync rule), `docs/maintainers/model_specs.md` (spec schema),
`docs/maintainers/parallel_{sessions,model_validation,server_behavior_contract}.md` (the
experimental parallel runtime), `docs/gguf.md`, `docs/models/*.md`,
`docs/community_models/*.md`, `docs/build/*.md` (per-backend and per-arch, incl. aarch64/Jetson),
`docs/reports/*.md` (published graph-memory and optimization reports per backend).
