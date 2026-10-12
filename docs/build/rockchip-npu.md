# Rockchip NPU (RKNN) Decoder Path

Opt-in, per-stage acceleration of one model stage on Rockchip RK3588/RK3576/RK356x NPUs through
`librknnrt`. This is not a ggml backend: ggml has no RKNN backend, and a full backend port is a
different project. Here a single model stage that already computes through its own graph and
`ExecutionContext` is routed to a pre-compiled RKNN network instead of ggml.

Currently implemented for three families: the `qwen3_tts` speech tokenizer decoder (the vocoder,
codes to PCM), the `piper_tts` VITS decoder (latent to PCM) and the `kokoro_tts` iSTFTNet
generator (latent to spectrogram). Everything below is measured on an
Orange Pi 5 Plus (RK3588, 4x A76 + 4x A55).

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

### Piper (VITS) decoder

`tools/prepare_piper_tts_rknn_vocoder.py` slices a Piper voice ONNX at the seam the runtime already
has — the inverse-flow input `z [1,192,N]` and its frame mask `y_mask [1,1,N]` (plus `g` for a
multi-speaker voice) — checks the fixed-window decode against a full-length decode, and compiles
the slice:

```bash
<export-venv>/bin/python tools/prepare_piper_tts_rknn_vocoder.py \
  --voice-onnx en_US-lessac-medium.onnx --skip-rknn --output decoder_z.onnx
<converter-venv>/bin/python tools/prepare_piper_tts_rknn_vocoder.py \
  --onnx decoder_z.onnx --target rk3588
```

The seam is the flow input and not the input of `dec.conv_pre`: the flow belongs to the decoder
half, and the runtime's latent is the pre-flow one. Slicing at `conv_pre` compiles a decoder that
expects post-flow samples and feeds it pre-flow ones, which is quiet mush (measured: output RMS
4x low, no intelligible speech).

RKNN has no dynamic shapes, so the engine is compiled for one window of 93 latent slices and the
runtime walks the latent over it: 45 slices emitted per step, 24 slices of real context on each
side, a short window zero-padded and mask-zeroed at the end that has no data, and the window that
reaches the end of the latent extends its left context to fill the window. The 24 is measured, not
inherited: at the 45+5+5 window paroli ships, interior windows score 8-60 dB against a
full-length decode (93: 98+ dB, 77: 51+ dB). The mask is 1 over the real slices and 0 over the
padding — the flow multiplies its activations by the mask after every convolution, so a padded
frame marked 1 propagates its conv biases into the real frames and costs 10-20 dB there.

The last slices of the latent are the one thing a fixed-shape window cannot reproduce: a
full-length decode has the end of the tensor there, where a window has padding, and a conv layer
two levels deep reads the pad's own conv output (nonzero, from the biases) where the full decode
reads the implicit zeros past the tensor end. No mask value in the padding fixes it. Measured: the
last ~7 slices of a short latent land 20-40 dB off, the interior windows are exact. paroli ships
the same trade.

### Kokoro (iSTFTNet) generator

`tools/prepare_kokoro_tts_rknn_generator.py` re-traces the generator the GGUF weights describe
(`decoder.module.generator.*` in `kokoro-v1_0.pth`) at one fixed window and compiles it. The seam
is the generator alone — `x [1,512,W]`, the harmonic conditioning `har [1,22,60W+1]`, the style
vector, and the frame mask at the two rates the graph normalizes at, in and out as
`spec_phase [1,22,60W+1]`. Source-signal synthesis, the forward STFT and the iSTFT head stay on
CPU:

```bash
<export-venv>/bin/python tools/prepare_kokoro_tts_rknn_generator.py \
  --checkpoint kokoro-v1_0.pth --config config.json --dump-dir <runtime-dump> \
  --output-dir artifacts --frames 128 --skip-rknn
<converter-venv>/bin/python tools/prepare_kokoro_tts_rknn_generator.py \
  --onnx artifacts/generator_w128.onnx --target rk3588
```

The window is capped by the NPU, not by the model: an op-encoding field on the time axis is 13
bits, so the longest axis in the graph, the fine rate `60W + 1`, has to stay under 8192. At W=512
the compiler emits ~800k `REGTASK: The bit width of field value exceeds the limit, ... limit:
0x1fff, value: 0x7800` lines and the engine voices clipped noise (rms 32764 of 32767, SNR -27 dB
against the CPU decode). 128 is the widest power of two inside the limit, and it emits
`128 - 2*28 = 72` latent frames per step, where 28 is the measured receptive field.

Three rewrites are compiled in, all of them the reference Kokoro RKNN port's:

- **Weight-norm folded at load.** Left in the graph, `vector_norm` traces to 51 `ReduceL2` nodes
  and `rknn.build` aborts.
- **Masked instance norm through `GlobalAveragePool`.** A `ReduceSum` over the time axis lowers to
  a register field capped at 8191 (the same limit as above); GAP is the pooling engine's own
  reduction. The mask has to zero the padding at *every* stage boundary — after the upsampling
  reflect-pad, on the resblock sum, on the noise branch output — not only inside AdaIN's
  statistics, because a conv after a masked AdaIN writes real-frame sums into the pad and the next
  `conv_transpose` reads them where a full decode reads implicit zeros. With that, a padded window
  reproduces the same frames decoded unpadded to 135 dB.
- **Snake's `sin` to a clipped minimax polynomial, and its `1/alpha` folded into the following
  convolution.** `librknnrt` has no NPU `Sin`: the 48 sine nodes of a 128-frame window ran on CPU
  and were 41% of the window's time (350 ms of 846 ms, `RKNN_LOG_LEVEL=5` operator ranking). The
  polynomial is Clip + 3 Mul + 2 Add at degree 5, and degree 5, 7 and 9 measure the same 30.4 dB
  against the exact sine, so the clip saturation past pi dominates and the cheap fit wins. The
  alpha-fold is exact algebra and removes one multiply and one divide per Snake; elementwise ops
  are 84% of this graph's NPU time and the convolutions 15%, so removing full-tensor passes is
  where the wall clock is.

The exported graph is checked against the torch graph it was traced from, on the last window of
the runtime's walk, with the torch reference evaluated in the same sine form (124.6 dB), so the
gate measures the export and not the approximation.

## Verified

- Export stage: stepwise-vs-full-decode self-check 111.4 dB at B=64 and 103.7 dB at B=16,
  `cache_numel` 1303424, `head_dim` 64, `kv_len` 71.
- Conversion stage: converting a reference ONNX reproduces the reference `.rknn` byte for byte
  (identical MD5, 259688011 bytes).
- Kokoro export stage: the re-traced generator reproduces the runtime's own generator at 60.5 dB
  interior SNR against a dumped decode, a padded 128-frame window reproduces the same frames
  decoded unpadded at 135.1 dB, and the exported ONNX reproduces the torch graph it was traced
  from at 124.6 dB.


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

Piper takes one option and is offline only:

```bash
./build/bin/audiocpp_cli --task tts --family piper_tts \
  --model models/Piper-TTS-GGUF/piper-en-us-lessac-medium-orig.gguf \
  --text "..." --text-chunk-size 60 --out out.wav \
  --session-option piper_tts.rknn_decoder=/root/tts-rknn/piper-lessac-w93.rknn
```

The engine is per voice: a `.rknn` compiled from lessac's latent is validated against the loaded
voice's channel count at init and fails with a message naming the producer if it does not match.

Kokoro takes one option and is offline only:

```bash
./build/bin/audiocpp_cli --task tts --family kokoro_tts \
  --model models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf \
  --text "..." --seed 1234 --out out.wav \
  --session-option kokoro_tts.rknn_generator=/root/tts-rknn/gen_w128.rknn
```

The engine is per model, not per voice: the generator weights are in the GGUF, and the engine is
validated against the loaded model's latent channel count, its spectrogram row count and its audio
samples per latent frame at init.

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

### Piper, measured on the same board

`piper-en-us-lessac-medium-orig.gguf`, 443 characters at `--text-chunk-size 60` (10 chunks,
25.18 s of audio), seed 1234:

| path | RTF | peak RSS | decoder ms |
|---|---|---|---|
| ggml decoder | 0.198 | 299 MB | 2070 |
| RKNN decoder (93-slice window) | 0.0696 | 175 MB | 1030 |

2.85x wall, 2.0x on the decoder itself (0.95 → 0.46 ms per emitted slice), -42% RSS. The
decoder-side headroom is smaller than the projection from paroli's numbers because the window
carries 24+24 slices of context for 45 emitted slices, and this board's ggml VITS decoder is
already fast (RTF 0.20).

Quality: ggml-vs-RKNN output SNR 24.1 dB global, 40 dB median per 256-sample slice, no alignment
drift (best lag 0 samples in every window), and the max sample-to-sample step is unchanged
(13384 vs 13412), so no seam clicks. The dips are near-silence (LSB noise) and one loud plosive
onset, i.e. the fp16 noise floor, not the stitching.

### Kokoro, measured on the same board

`kokoro-82m-q8_0.gguf`, 185 Russian characters, seed 1234, `--threads 4` (14.85 s of audio,
1188 latent frames, 17 windows of 72 emitted frames):

| path | RTF | peak RSS | generator ms |
|---|---|---|---|
| ggml generator | 1.415 | 2777 MB | 17748 |
| RKNN generator (128-frame window) | 0.867 | 2334 MB | 9609 |

1.63x wall, 1.85x on the generator itself (14.9 → 8.1 ms per latent frame processed), -16% RSS.
The generator-side gain is smaller than the 5.1x measured in the reference Kokoro engine because
this window carries 28+28 frames of context for 72 emitted frames (1.78x of compute spent on
context, and the 28 is the exact receptive field, so it cannot shrink), and because the window
width is capped at 136 frames by the register-field limit, which is what forces the 17 windows.

Quality: ggml-vs-RKNN output SNR 0.49 dB. That number is *not* a defect measurement and has to be
read against what it compares. The generator normalizes per window (AdaIN), so a multi-window
decode is a different valid synthesis and not an approximation of the full-length one: dropping 2
of 486 latent frames from the statistics moves the generator output by 25% RMS. What is measured
is exactness per window — a padded window reproduces the same frames decoded unpadded to 135.1 dB
— and the module reproduces the runtime's own generator to 60.5 dB interior (the floor is the
q8_0 style affine). The reference Kokoro engine, with the same per-window normalization at
`T_fix=50`, measures 17.11 dB against its own CPU decode on a 7.1 s utterance (2 windows); the
gap between its 17 dB and our 0.5 dB is window count, not stitching. Utterances of 72 latent
frames or fewer (about 0.9 s) fit one window and are exact.

## NPU core use

`rknn.config` in 2.3.2 has no `core_num` (it has `single_core_mode`), so an engine is built
multi-core and the runtime core mask decides. `RKNN_NPU_CORE_AUTO` lets the driver pick, and it
picks one core for every one of these engines: `/sys/kernel/debug/rknpu/load` stays at
`Core0: 55%, Core1: 0%, Core2: 0%` for the whole `qwen3_tts` vocoder run and at `Core0: 100%` for
the smaller ones. All three sidecars therefore ask for `RKNN_NPU_CORE_0_1_2` explicitly and fall
back to `AUTO` if the driver rejects it (single-core SoCs do). What the mask buys is per engine:
`piper_tts` and `kokoro_tts` get real spread (kokoro Core0 95%, Core1 18%, Core2 19%), and the
`qwen3_tts` vocoder gets spread too (41/15/16) for 1.8% on the stage (21766 → 21383 ms for 300
Russian characters, same binary with both masks, greedy) — that engine reads its weights per step,
so it is DRAM-bound, not scheduler-bound, and the extra cores have no bandwidth to use.

One context per sidecar, and one job per context: the mask spreads a *single* job over the cores
(kokoro measures Core0 95%, Core1 18%, Core2 19% during a run). Running the windows concurrently
on three `rknn_dup_context` contexts, which is what the reference Kokoro engine does, was built
and measured here at 1.15x on the window loop (9650 → 8375 ms) with corrupted output (-7 dB SNR,
rms 2x the reference): duplicated contexts share the runtime's tensor staging, and this engine
streams ~75 MB of weights per window, so the windows are DRAM-bound, not scheduler-bound.

## What this path is not for

The talker and the code predictor stay on the CPU. Moving the code predictor to the same NPU was
built and measured at 5.4x slower than the CPU path: the step is roughly 75 tiny S=1 GEMV
operations, and the NPU matrix units only pay off on real GEMMs like the vocoder's 64-token
chunk. The Mali GPU is 2.1x slower than the CPU for this workload.
