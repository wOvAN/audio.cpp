#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::kokoro_tts {

// Rockchip NPU (RKNN) backend for the iSTFTNet generator (latent to spectrogram).
//
// The graph is the generator the GGUF weights describe, re-traced at one fixed window
// (see tools/prepare_kokoro_tts_rknn_generator.py): `x [1,512,W]`, the harmonic
// conditioning `har [1,22,300W+1]`, the style vector, and the frame mask at the two
// rates the graph normalizes at. One call voices a window and returns its
// spectrogram columns; the engine has no dynamic shapes, so a longer utterance walks
// the window over the latent: `W - 2*padding` frames emitted per step, `padding` real
// frames of context on each side, zero-padded and mask-zeroed where the latent has run
// out, and the window that reaches the end right-aligned in the graph input.
//
// The seams are stitched in the spectrogram domain: each window's emitted columns are
// written into one full-length spectrogram and the iSTFT head runs once over it, so
// there is no waveform crossfade and no depop search.
//
// The generator normalizes per window (AdaIN), so a multi-window decode is a different
// synthesis than a full-length one, not an approximation of it: pick a window wide
// enough that a whole text chunk fits one window, which is exact.
//
// An engine compiled with `--global-stats` (7 inputs / 2 outputs: `stats_in`,
// `use_global`, `stats_out`) takes the per-channel AdaIN (mean, biased variance) of
// every normalization site, interleaved per site in forward order, as an input: the
// engine then normalizes every window with the statistics of a full-length decode. The
// caller computes them on CPU from the full latent. In fp32 that is exact (measured:
// 9.7 dB -> 135.4 dB on the 486-frame dump). On the RK3588 it is a trade, not a win:
// the statistics scale the window activations up, the engine's fp16 range cannot carry
// them at the utterance edges (the first and last windows are taken from the CPU pass),
// and the render costs 1.8x wall clock and +1.5 GB RSS for a per-window loudness spread
// that measured worse than the shipping engine's (15.9 dB against 7.2 dB). Default off.
class KokoroTtsRknnGenerator {
public:
    // `latent_channels`, `feature_rows` and `samples_per_frame` (audio samples per latent
    // frame, the upsampling product times the iSTFT hop) come from the loaded weights;
    // the engine's own shapes are checked against them at load.
    KokoroTtsRknnGenerator(
        const std::string & model_path,
        int64_t latent_channels,
        int64_t feature_rows,
        int64_t samples_per_frame);
    ~KokoroTtsRknnGenerator();

    KokoroTtsRknnGenerator(const KokoroTtsRknnGenerator &) = delete;
    KokoroTtsRknnGenerator & operator=(const KokoroTtsRknnGenerator &) = delete;

    // Latent frames per NPU step (the compiled window) and per emitted step.
    int64_t window() const;
    int64_t emitted() const;
    // Spectrogram columns the generator emits per latent frame (the upsampling product).
    int64_t columns_per_frame() const;
    // True when the engine takes the full-decode AdaIN statistics as an input.
    bool uses_global_stats() const;
    // Floats `run`'s `stats` must carry for such an engine (0 for a shipping engine).
    int64_t stats_elements() const;

    // `x` is `latent_channels` rows of `x_stride` floats, `har` is `feature_rows` rows
    // of `har_stride` floats; only the first `frames` columns of `x` and the first
    // `frames * columns_per_frame() + 1` columns of `har` are real. The result is
    // `feature_rows` rows of `frames * columns_per_frame() + 1` contiguous floats.
    // `stats` (only for a `--global-stats` engine) is `stats_elements()` floats of
    // full-decode AdaIN statistics; null falls back to per-window statistics.
    std::vector<float> run(
        const float * x,
        int64_t x_stride,
        const float * har,
        int64_t har_stride,
        const float * style,
        int64_t frames,
        const float * stats = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kokoro_tts
