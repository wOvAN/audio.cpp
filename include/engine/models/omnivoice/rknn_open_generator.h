#pragma once

// Open-library (rknpu2_matmul) backend for the OmniVoice diffusion-LM generator:
// the same 28-layer trunk as OmniVoiceRknnGenerator's compiled front-shard ladder,
// run through the open multicore fp16 matmul/attention library with a real KV
// cache (prefill once, target-rows-only steps). Private board builds only: the
// library is AGPL and is never linked in upstream builds.
//
// Contract = the front-shard contract: run() writes raw post-trunk hidden rows in
// the packed host layout (conditional target block at [total-target, total), the
// unconditional block at [row_capacity, row_capacity + target)), and the CFG tail
// (final norm, audio heads, guided combine) stays in the host fp32 tail graph.
// Measured on the board: step 3.9-4.0 s vs the compiled engine's 23.2 s at
// seq1536, SNR ladder 65-67 dB (bench fullstep3.cpp / ovs engine gate).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/models/omnivoice/rknn_generator.h"

namespace engine::models::omnivoice {

// One layer's host weight views, straight out of the runtime's weight store
// (generator.cpp Qwen3DiffusionLayerWeights). `qkv`/`gate_up` carry the packed
// row order [q|k|v] / [gate|up] when `packed` is set; otherwise q/k/v and
// gate/up point at the separate tensors. `*_is_f16` says whether the buffer
// holds fp16 bits (GGML_TYPE_F16) or fp32 floats to convert at pack time.
struct OmniVoiceOpenLayerView {
    const void * qkv, * q, * k, * v;
    const void * o;
    const void * gate_up;  // packed row order [gate | up]
    const void * gate, * up, * down;
    const void * n_in, * n_post, * n_qn, * n_kn;  // always fp32 (load_f32_tensor)
    bool packed = false;
    bool qkv_is_f16 = true, o_is_f16 = true, gu_is_f16 = true, down_is_f16 = true;
    bool norms_is_f16 = true;
};

struct OmniVoiceOpenWeightViews {
    std::vector<OmniVoiceOpenLayerView> layers;
    const void * text_embedding = nullptr;   // [text_embedding_rows, hidden]
    int64_t text_embedding_rows = 0;
    const void * audio_embedding = nullptr;  // [audio_embedding_rows, hidden]
    int64_t audio_embedding_rows = 0;
    bool text_embedding_is_f16 = true, audio_embedding_is_f16 = true;
};

struct OmniVoiceOpenModelDims {
    int64_t codebooks = 8;
    int64_t audio_vocab = 1025;
    int64_t heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t hidden = 1024;
    int64_t intermediate = 3072;
    double rope_theta = 1000000.0;
};

class OmniVoiceRknnOpenGenerator : public OmniVoiceRknnGenerator {
public:
    // Views must stay valid for the generator's lifetime (they do: the weight
    // store outlives the sidecar in OmniVoiceQwen3DiffusionRuntime::Impl).
    OmniVoiceRknnOpenGenerator(
        const OmniVoiceOpenWeightViews & weights,
        const OmniVoiceOpenModelDims & dims,
        const std::string & weight_cache_dir,
        std::string fingerprint);
    ~OmniVoiceRknnOpenGenerator() override;

    int64_t max_tokens() const override;
    int64_t engine_count() const override;
    bool front_shard() const override;
    int64_t bucket_for(int64_t total_tokens) const override;
    void run(int64_t bucket, const OmniVoiceRknnStepInputs & inputs, float * out, int64_t row_capacity) override;
    // New request: the next run() re-prefills (caches are per-request state).
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::omnivoice
