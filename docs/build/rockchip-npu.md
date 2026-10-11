# Rockchip NPU (RKNN) Decoder Path

Opt-in, per-stage acceleration of one model stage on Rockchip RK3588/RK3576/RK356x NPUs through
`librknnrt`. This is not a ggml backend: ggml has no RKNN backend, and a full backend port is a
different project. Here a single model stage that already computes through its own graph and
`ExecutionContext` is routed to a pre-compiled RKNN network instead of ggml.

Currently implemented for one family: the `qwen3_tts` speech tokenizer decoder (the vocoder,
codes to PCM). Everything below is measured on an Orange Pi 5 Plus (RK3588, 4x A76 + 4x A55).

## Requirements

Board side:

- `librknnrt.so` (RKNPU2 runtime, 2.x) on the loader path, and `rknn_api.h` for the headers.
  The Rockchip SDK's `install.sh` puts the library in `/usr/lib` and the headers in
  `/usr/local/include`.
- The RKNPU kernel driver. Check with `cat /sys/kernel/debug/rknpu/version` or `dmesg | grep -i rknpu`.

Host side (x86_64, for building the `.rknn`):

- `rknn-toolkit2` in a dedicated venv. Its pins are strict (`numpy<=1.26.4`, `onnx==1.16.1`,
  upper-bounded `torch`, Python 3.6-3.12); do not resolve them against a system Python.
- On Python 3.12 the 2.3.2 converter needs two patches the SDK does not ship: `setuptools<81`
  (`rknn.api.base_utils` imports `pkg_resources`, removed in setuptools 81), and an
  `onnx/mapping.py` shim inside the venv's `onnx` package (the module is gone in onnx 1.16, and
  onnx 1.15 has no cp312 wheel). RKNN does `import onnx` then `onnx.mapping.X`, so the shim must
  also be bound by appending `from onnx import mapping` to `onnx/__init__.py`.

## Build

Native on the board (the same flags work in a cross build):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_RKNN=ON \
  -DENGINE_RKNN_INCLUDE_DIR=/usr/local/include \
  -DENGINE_RKNN_LIBRARY=/usr/lib/librknnrt.so \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_tts
cmake --build build --parallel --target audiocpp_cli
```

`ENGINE_ENABLE_RKNN` is `OFF` by default and nothing in the default build links `librknnrt`. With
the flag off, passing `qwen3_tts.rknn_decoder` fails with
`Qwen3 speech decoder RKNN: built without ENGINE_ENABLE_RKNN` rather than falling back silently.

## Model preparation

The `.rknn` is not downloadable and nothing in the runtime produces it. Build it on the host from
the `Qwen/Qwen3-TTS-12Hz-0.6B-Base` checkpoint (the `speech_tokenizer` subdirectory, the same
weights the GGUF package was converted from):

```bash
python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir <snapshot-dir> --codes-length 64
python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir <snapshot-dir> --codes-length 16
```

Each run re-traces the decoder as one fixed-shape streaming step, checks the stepwise decode
against a full decode (SNR must stay above 40 dB), exports ONNX, and compiles it for
`--target` (default `rk3588`). The graph is stateful: the causal-conv tails and the sliding-window
KV history travel through the graph I/O as one flat fp32 cache, so a step decodes from continuous
state and there are no chunk seams. `--codes-length` is compiled in, so the two-tier first chunk
needs one file per batch size; the cache layout does not depend on it, so a small-batch step's
cache feeds the large-batch model directly.

The two stages want different toolchains (the export needs the model package, the conversion needs
the toolkit's pinned `torch`/`onnx`), so run them in the two venvs and hand the ONNX over:

```bash
<export-venv>/bin/python tools/prepare_qwen3_tts_rknn_vocoder.py \
  --model-dir <snapshot-dir> --codes-length 64 --skip-rknn --output vocoder_stateful_64.onnx
<converter-venv>/bin/python tools/prepare_qwen3_tts_rknn_vocoder.py \
  --onnx vocoder_stateful_64.onnx --target rk3588
```

`--onnx` reads the step contract (batch size, cache size, head dim) back out of the graph, so the
conversion stage restates nothing.

The export stage is version-sensitive: a recent `torch`/`onnx` decomposes LayerNorm and Softmax into
patterns (`Pad`, `Exp`, `Shape`, `ConstantOfShape`) that `rknn.build` aborts on inside
`fold_constant`, with a core dump and no Python traceback. The script counts those ops after
exporting and says so there, where it is cheap to act on. Measured with this graph: the
toolkit-era pair exports 1064 nodes and compiles; a current pair exports 2653 and aborts.

A `.rknn` is compiled per target SoC and is not portable between chips. The graph is built fp16
(w16a16) and deliberately not quantized: the Snake activation makes Q8/INT8 audibly noisy
(9.5-13.4 dB SNR) and fp16 measures lossless (56.8 dB).

`--target` accepts the RKNPU2 platforms the toolkit knows, but only `rk3588` has been built and
measured here; the others are untested, not unsupported.

## Verified

- Export stage: stepwise-vs-full-decode self-check 111.4 dB at B=64 and 103.7 dB at B=16,
  `cache_numel` 1303424, `head_dim` 64, `kv_len` 71.
- Conversion stage: converting a reference ONNX reproduces the reference `.rknn` byte for byte
  (identical MD5, 259688011 bytes).


## Run

```bash
./build/bin/audiocpp_cli --task tts --family qwen3_tts --mode streaming \
  --model models/Qwen3-TTS-12Hz-0.6B-Base-GGUF --text "..." --voice-ref ref.wav \
  --out out.wav \
  --session-option qwen3_tts.conv_weight_type=f16 \
  --session-option qwen3_tts.rknn_decoder=/root/tts-rknn/vocoder/vocoder_stateful_64b_fp16.rknn \
  --session-option qwen3_tts.rknn_decoder_first=/root/tts-rknn/vocoder/vocoder_stateful_16_fp16.rknn
```

`rknn_decoder_first` is optional and only affects streaming: the first token group runs on the
small-batch model, which is what cuts time-to-first-audio. When the RKNN decoder is active the
ggml decoder weights are not loaded at all, which is where most of the memory saving comes from.

## Measured on RK3588

300 Russian characters, greedy decoding, `conv_weight_type=f16`, all other stages on CPU:

| mode | RTF / wall | peak RSS | decoder ms per token |
|---|---|---|---|
| offline, ggml decoder | 3.01 | 4200 MB | 130.1 |
| offline, RKNN B=64 | 2.17 | 2560 MB | 64.1 |
| streaming, ggml decoder | 151.2 s | - | 365 |
| streaming, RKNN 16+64 | 57.7 s, first audio 0.88 s | - | 62.4 |

Streaming gains more than offline because the ggml streaming decoder re-voices the left-context
tokens in every window, while the stateful model voices each token once.

Quality: offline ggml-vs-RKNN output SNR 27 dB, with the per-second dips confined to
near-silence (the fp16 noise floor), and no discontinuity at the group boundaries. The two
batch sizes agree to 57 dB.

## What this path is not for

The talker and the code predictor stay on the CPU. Moving the code predictor to the same NPU was
built and measured at 5.4x slower than the CPU path: the step is roughly 75 tiny S=1 GEMV
operations, and the NPU matrix units only pay off on real GEMMs like the vocoder's 64-token
chunk. The Mali GPU is 2.1x slower than the CPU for this workload.
