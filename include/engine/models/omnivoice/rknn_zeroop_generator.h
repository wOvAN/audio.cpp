#pragma once

// Vendor zero-op (librknnrt rknn_matmul_api) backend for the OmniVoice
// diffusion-LM generator: the same 28-layer trunk and front-shard contract as
// OmniVoiceRknnOpenGenerator, run on the vendor single-core matmul API with
// tile-assembled attention and a host fp32 softmax. AGPL-clean: links only
// librknnrt, so this is the PR-able variant of the open-engine line.
//
// Contract = the front-shard contract (see rknn_open_generator.h): run() writes
// raw post-trunk hidden rows in the packed host layout, the CFG tail stays in
// the host fp32 tail graph. Board gate (2026-11-15, standalone bench): 28-layer
// step/prefill referee ladder 80.6-132.9 dB, step ~14.5 s at seq1536 vs the
// compiled ladder's 23.2 s.

#include "engine/models/omnivoice/rknn_open_generator.h"

namespace engine::models::omnivoice {

// Views struct is engine-agnostic (OmniVoiceOpenWeightViews /
// OmniVoiceOpenModelDims, reused from the open generator header).
class OmniVoiceRknnZeroopGenerator : public OmniVoiceRknnGenerator {
public:
    // Views must stay valid for the generator's lifetime (they do: the weight
    // store outlives the sidecar in OmniVoiceQwen3DiffusionRuntime::Impl).
    OmniVoiceRknnZeroopGenerator(
        const OmniVoiceOpenWeightViews & weights,
        const OmniVoiceOpenModelDims & dims,
        const std::string & weight_cache_dir,
        std::string fingerprint);
    ~OmniVoiceRknnZeroopGenerator() override;

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
