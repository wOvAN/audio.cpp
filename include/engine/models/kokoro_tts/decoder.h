#pragma once

#include "engine/models/kokoro_tts/predictor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace kokoro_ggml {

struct KokoroIstftNetDecoderCapacityContract {
    int64_t decoder_frames = 0;
    int64_t conditioning_frames = 0;
};

class KokoroIstftNetDecoderRuntime {
public:
    // `rknn_generator_path` (optional) routes the generator stage to a pre-compiled
    // RKNN engine; see engine/models/kokoro_tts/rknn_generator.h.
    KokoroIstftNetDecoderRuntime(
        std::shared_ptr<const KokoroWeights> weights,
        ggml_backend_t backend,
        int n_threads,
        bool use_device_backend,
        uint64_t rng_seed,
        KokoroIstftNetDecoderCapacityContract contract,
        const std::string & rknn_generator_path = {});
    ~KokoroIstftNetDecoderRuntime();

    KokoroIstftNetDecoderRuntime(const KokoroIstftNetDecoderRuntime &) = delete;
    KokoroIstftNetDecoderRuntime & operator=(const KokoroIstftNetDecoderRuntime &) = delete;

    void prepare(KokoroIstftNetDecoderCapacityContract contract);

    std::vector<float> decode(
        const PredictorOutputs & predictor,
        const std::vector<float> & ref_s);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kokoro_ggml
