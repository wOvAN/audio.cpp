#if defined(ENGINE_HAVE_RKNN_OPEN)  // open rknpu2_matmul library is AGPL: private builds only
// ovs_engine.cpp - see ovs_engine.h. Direct port of the gate-passing bench
// fullstep3.cpp v6, with prefill upgraded to a full-canvas attention forward
// (v6 prefill ran only the QKV projections, silently feeding layer-l caches the
// layer-0 hidden states; its SNR gate measured layer 0 only, so the engine must
// not inherit that shortcut).
#include "ovs_engine.h"
#if defined(__aarch64__)

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <time.h>

#include "rk_npu_attention_f16.h"
#include "rk_npu_common.h"
#include "rk_npu_matmul_f16.h"

namespace ovs {
namespace {

constexpr int kNqkv = (kNH + 2 * kNKV) * kHD;  // 4096
constexpr int kNo = kHid, kNgu = 2 * kInter, kNd = kHid;
constexpr int kKqkv = kHid, kKo = kNH * kHD, kKgu = kHid, kKd = kInter;
constexpr int kSn[4] = {kNqkv, kNo, kNgu, kNd}, kSk[4] = {kKqkv, kKo, kKgu, kKd};
constexpr int kUsk[4] = {0, 1, 0, 1};  // splitk x3 only for deep-K (O, down)
constexpr int kSplit = 3;
constexpr float kEps = 1e-6f;
constexpr int kCapMax = 2048 + 256;  // attention cache growth limit

inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
inline uint16_t f2h(float x) { __fp16 v = (__fp16)x; uint16_t r; memcpy(&r, &v, 2); return r; }
inline float h2f(uint16_t x) { __fp16 v; memcpy(&v, &x, 2); return (float)v; }

void rmsnorm_neon(const float* x, const uint16_t* w, uint16_t* o, long rows, int cols) {
    for (long r = 0; r < rows; ++r) {
        const float* p = x + (size_t)r * cols;
        float32x4_t acc = vdupq_n_f32(0);
        for (int c = 0; c < cols; c += 4) acc = vmlaq_f32(acc, vld1q_f32(p + c), vld1q_f32(p + c));
        float k = 1.0f / sqrtf((acc[0] + acc[1] + acc[2] + acc[3]) / cols + kEps);
        float32x4_t kq = vdupq_n_f32(k);
        for (int c = 0; c < cols; c += 4) {
            float32x4_t v = vmulq_f32(vld1q_f32(p + c), vmulq_f32(kq, vcvt_f32_f16(vld1_f16((const __fp16*)w + c))));
            vst1_f16((__fp16*)o + (size_t)r * cols + c, vcvt_f16_f32(v));
        }
    }
}
void silu_neon(const float* g, const float* u, uint16_t* o, long n) {
    long i = 0;
    for (; i + 4 <= n; i += 4) {
        const float* g4 = g + i;
        float r[4] = {g4[0] / (1 + expf(-g4[0])), g4[1] / (1 + expf(-g4[1])), g4[2] / (1 + expf(-g4[2])), g4[3] / (1 + expf(-g4[3]))};
        float32x4_t mv = vmulq_f32(vld1q_f32(r), vld1q_f32(u + i));
        vst1_f16((__fp16*)o + i, vcvt_f16_f32(mv));
    }
    for (; i < n; ++i) o[i] = f2h((g[i] / (1.0f + expf(-g[i]))) * u[i]);
}

}  // namespace

struct Engine::Impl {
    rk_npu_ctx* ctx = nullptr;
    rk_npu_iommu_domain* dom = nullptr;
    std::vector<LayerW> w;
    std::string cache_dir, fp, err;
    bool packed = false;

    struct Shape {  // A/D operands, sized at max M ever requested
        rk_npu_mem in{}, out{};
        int m_max = 0;
    } sh[4];
    rk_npu_mem wt[kLayers][4]{};  // packed B, M-independent
    struct PlanPair {
        rk_npu_matmul_f16_plan* p = nullptr;
        rk_npu_matmul_f16_splitk_plan* sp = nullptr;
    };
    std::map<std::pair<int, int>, PlanPair> plans;  // (M, layer*4+shape)

    rk_npu_attention_f16_workspace* ws = nullptr;
    rk_npu_attention_f16_cache* c[2][kLayers]{};
    int cap = 0;

    // geometry state (max ever requested)
    int seq_pad = 0, tgt_pad = 0;
    std::vector<uint16_t> x16, q16, k16, v16, qch, ctxc, ctxh, mid16;
    std::vector<float> qkv32, o32, gu32, d32, h, ht;
    std::vector<uint8_t> maskb;  // [2, seq_pad, seq_pad] boolean, rebuilt per call
    std::vector<uint16_t> kp[2], vp[2];  // canvas KV rows [NKV,seq_pad,HD], V stored = v/p
    float p_scale[2][kLayers]{};
    void (*hook)(int, void*) = nullptr;
    void* hook_ud = nullptr;

    void mcfg(int m, int s, rk_npu_matmul_f16_config& c) {
        rk_npu_matmul_f16_config_init(&c, m, kSn[s], kSk[s], RK_NPU_FUSE_NONE);
        c.core_mask = 7;
    }

    int gemm(int l, int s, int m, const uint16_t* A, float* D, Timings* t) {
        rk_npu_matmul_f16_config c1;
        mcfg(m, s, c1);
        double t0 = now_us();
        if (kUsk[s]) rk_npu_matmul_f16_splitk_pack_a(kSplit, &c1, A, &sh[s].in);
        else rk_npu_matmul_f16_pack_a(&c1, A, &sh[s].in);
        rk_npu_mem_sync(ctx, &sh[s].in, RK_NPU_SYNC_TO_DEVICE);
        if (t) t->pack += now_us() - t0;
        PlanPair& pp = plans[{m, l * 4 + s}];
        int rc;
        t0 = now_us();
        if (kUsk[s]) rc = rk_npu_matmul_f16_splitk_run(ctx, pp.sp, &sh[s].in, &wt[l][s], &sh[s].out);
        else rc = rk_npu_matmul_f16_run(ctx, pp.p, &sh[s].in, &wt[l][s], nullptr, &sh[s].out);
        if (t) t->gemm += now_us() - t0;
        if (rc) { err = "gemm run l" + std::to_string(l) + " s" + std::to_string(s); return 1; }
        t0 = now_us();
        rk_npu_mem_sync(ctx, &sh[s].out, RK_NPU_SYNC_FROM_DEVICE);
        if (kUsk[s]) rk_npu_matmul_f16_splitk_unpack_d_f32(kSplit, &c1, &sh[s].out, D);
        else rk_npu_matmul_f16_unpack_d_f32(&c1, &sh[s].out, D);
        if (t) t->unpack += now_us() - t0;
        return 0;
    }

    // head-major rope of one canvas row from its qkv32 row (q/k norm+rope, v
    // raw); dst head stride hstride (seq_pad*HD in prefill, tgt_pad*HD in step)
    void rope_row(int l, const float* row, const float* cs, const float* sn,
                  uint16_t* qdst, uint16_t* kdst, uint16_t* vdst, long hstride) {
        const uint16_t* nqn = w[l].n_qn;
        const uint16_t* nkn = w[l].n_kn;
        for (int hh = 0; hh < kNH; ++hh) {
            const float* p = row + (size_t)hh * kHD;
            float ss = 0;
            for (int c = 0; c < kHD; ++c) ss += p[c] * p[c];
            float k2 = 1.0f / sqrtf(ss / kHD + kEps);
            uint16_t* dst = qdst ? qdst + (size_t)hh * hstride : nullptr;
            for (int j = 0; j < kHD / 2; ++j) {
                float a = p[j] * k2 * h2f(nqn[j]), z = p[j + kHD / 2] * k2 * h2f(nqn[j + kHD / 2]);
                if (dst) {
                    dst[j] = f2h(a * cs[j] - z * sn[j]);
                    dst[j + kHD / 2] = f2h(z * cs[j + kHD / 2] + a * sn[j + kHD / 2]);
                }
            }
        }
        for (int hh = 0; hh < kNKV; ++hh) {
            const float* p = row + (size_t)kNH * kHD + (size_t)hh * kHD;
            float ss = 0;
            for (int c = 0; c < kHD; ++c) ss += p[c] * p[c];
            float k2 = 1.0f / sqrtf(ss / kHD + kEps);
            uint16_t* dst = kdst + (size_t)hh * hstride;
            for (int j = 0; j < kHD / 2; ++j) {
                float a = p[j] * k2 * h2f(nkn[j]), z = p[j + kHD / 2] * k2 * h2f(nkn[j + kHD / 2]);
                dst[j] = f2h(a * cs[j] - z * sn[j]);
                dst[j + kHD / 2] = f2h(z * cs[j + kHD / 2] + a * sn[j + kHD / 2]);
            }
            const float* pv = row + (size_t)(kNH + kNKV) * kHD + (size_t)hh * kHD;
            uint16_t* dv = vdst + (size_t)hh * hstride;
            for (int c = 0; c < kHD; ++c) dv[c] = f2h(pv[c]);
        }
    }

    // boolean canvas mask [2,sp,sp] from additive bias; pad rows self-diagonal
    // (compiled convention: a pad token attends to itself only, never a full row)
    void build_mask(const float* mask_bias, int seq, int sp) {
        maskb.assign((size_t)2 * sp * sp, 0);
        for (int b = 0; b < 2; ++b) {
            for (int r = 0; r < seq; ++r)
                for (int cl = 0; cl < seq; ++cl)
                    maskb[((size_t)b * sp + r) * sp + cl] = mask_bias[((size_t)b * seq + r) * seq + cl] > -1000.0f;
            for (int r = seq; r < sp; ++r) maskb[((size_t)b * sp + r) * sp + r] = 1;
        }
    }

    // attention for canvas rows [r0, r0+nrows) of batch b over span sp;
    // qsrc base is q16, qstride = rows per head (seq_pad in prefill, tgt_pad in step)
    int attn_rows(int l, int b, int r0, int nrows, int sp, long qstride, Timings* t) {
        for (int ch = 0; ch < nrows / kQrows; ++ch) {
            for (int hh = 0; hh < kNH; ++hh)
                memcpy(&qch[(size_t)hh * kQrows * kHD],
                       &q16[((size_t)b * kNH + hh) * qstride + (size_t)(r0 + ch * kQrows) * kHD],
                       (size_t)kQrows * kHD * 2);
            rk_npu_attention_f16_timings at{};
            rk_npu_attention_f16_boolean_mask m;
            rk_npu_attention_f16_boolean_mask_init(
                &m, maskb.data() + ((size_t)b * sp + r0 + ch * kQrows) * sp,
                (uint64_t)2 * sp * sp, 1, kQrows, sp);
            // version pinned at 1 is inert only because STABLE_SOFTMAX never
            // builds the device mask (update_mask runs on the non-stable path
            // only); switch acfg.flags to FIXED_SHIFT and this must become a
            // per-rewrite counter
            m.head_stride = 0; m.query_stride = sp; m.key_stride = 1; m.version = 1;
            double t0 = now_us();
            const int arc = rk_npu_attention_f16_run_half(ctx, ws, c[b][l], qch.data(), kQrows,
                                                          r0 + ch * kQrows, ctxc.data(), &at, &m);
            if (arc) {
                err = "attn run l" + std::to_string(l) + " b" + std::to_string(b) +
                      " r" + std::to_string(r0 + ch * kQrows) + " rc" + std::to_string(arc);
                return 1;
            }
            if (t) { t->attn += at.hardware_us; t->cpu += (now_us() - t0) - at.hardware_us; }
            // scatter head-major chunk into canvas-row-major ctxh
            for (int s = 0; s < kQrows; ++s)
                for (int hh = 0; hh < kNH; ++hh)
                    memcpy(&ctxh[(((size_t)b * nrows + (size_t)ch * kQrows + s)) * kNH * kHD + (size_t)hh * kHD],
                           &ctxc[((size_t)hh * kQrows + s) * kHD], kHD * 2);
        }
        return 0;
    }
};

Engine::Engine() : im_(new Impl()) {}
Engine::~Engine() { close(); }

void Engine::close() {
    Impl& I = *im_;
    if (I.ws) { rk_npu_attention_f16_workspace_free(I.ws); I.ws = nullptr; }
    for (int b = 0; b < 2; ++b)
        for (int l = 0; l < kLayers; ++l)
            if (I.c[b][l]) { rk_npu_attention_f16_cache_free(I.c[b][l]); I.c[b][l] = nullptr; }
    for (int s = 0; s < 4; ++s) {
        if (I.sh[s].in.handle) rk_npu_mem_free(I.ctx, &I.sh[s].in);
        if (I.sh[s].out.handle) rk_npu_mem_free(I.ctx, &I.sh[s].out);
        I.sh[s] = {};
    }
    for (int l = 0; l < kLayers; ++l)
        for (int s = 0; s < 4; ++s)
            if (I.wt[l][s].handle) rk_npu_mem_free(I.ctx, &I.wt[l][s]);
    for (auto& kv : I.plans) {
        if (kv.second.p) rk_npu_matmul_f16_plan_free(kv.second.p);
        if (kv.second.sp) rk_npu_matmul_f16_splitk_plan_free(kv.second.sp);
    }
    I.plans.clear();
    if (I.dom) { rk_npu_iommu_domain_free(I.dom); I.dom = nullptr; }
    if (I.ctx) { rk_npu_close(I.ctx); I.ctx = nullptr; }
    I.packed = false;
}

int Engine::open(const char* npu_dev, const char* weight_cache_dir, const std::string& fingerprint) {
    Impl& I = *im_;
    I.ctx = rk_npu_open(npu_dev);
    if (!I.ctx) { I.err = "npu open"; return 1; }
    I.dom = rk_npu_iommu_domain_create(I.ctx, 0);
    if (!I.dom) { I.err = "domain"; close(); return 1; }
    I.cache_dir = weight_cache_dir ? weight_cache_dir : "";
    I.fp = fingerprint;
    rk_npu_attention_f16_config acfg{};
    rk_npu_attention_f16_config_init(&acfg);
    acfg.query_heads = kNH; acfg.kv_heads = kNKV; acfg.head_dim = kHD;
    acfg.max_query_rows = kQrows; acfg.kv_tile = 1024;
    acfg.initial_capacity = kQrows; acfg.max_capacity = kCapMax;
    acfg.core_mask = 7;
    acfg.flags = (uint32_t)RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    acfg.mask_mode = RK_NPU_ATTENTION_F16_BOOLEAN_MASK;
    if (rk_npu_attention_f16_prepare(I.dom, &acfg, &I.ws)) { I.err = "attn ws"; close(); return 1; }
    for (int b = 0; b < 2; ++b)
        for (int l = 0; l < kLayers; ++l)
            if (rk_npu_attention_f16_cache_create(I.dom, &acfg, &I.c[b][l])) { I.err = "attn cache"; close(); return 1; }
    return 0;
}

const std::string& Engine::last_error() const { return im_->err; }
void Engine::set_layer_hook(void (*fn)(int, void*), void* ud) { im_->hook = fn; im_->hook_ud = ud; }
const uint16_t* Engine::dbg_q16() const { return im_->q16.data(); }
const uint16_t* Engine::dbg_k16() const { return im_->k16.data(); }
const uint16_t* Engine::dbg_ctxh() const { return im_->ctxh.data(); }
const float* Engine::dbg_h() const { return im_->h.data(); }
float Engine::dbg_p(int b, int l) const { return im_->p_scale[b][l]; }

void Engine::set_weights(const std::vector<LayerW>& w) {
    im_->w = w;
    im_->packed = false;
}

int Engine::pack_weights() {
    Impl& I = *im_;
    if (I.w.size() != kLayers) { I.err = "weights: need 28 layers"; return 1; }
    const int Mw = 2 * kQrows;  // weight_bytes is M-independent; any M queries it
    for (int s = 0; s < 4; ++s) {
        rk_npu_matmul_f16_config c1;
        I.mcfg(Mw, s, c1);
        rk_npu_matmul_sizes z{};
        if (kUsk[s] ? rk_npu_matmul_f16_splitk_query(kSplit, &c1, &z) : rk_npu_matmul_f16_query(&c1, &z)) { I.err = "query wt"; return 1; }
        for (int l = 0; l < kLayers; ++l)
            if (!I.wt[l][s].handle && rk_npu_mem_alloc(I.dom, z.weight_bytes, RK_NPU_MEM_DATA_DEFAULT, &I.wt[l][s])) { I.err = "alloc wt"; return 1; }
    }
    std::vector<uint64_t> wsz(4);
    for (int s = 0; s < 4; ++s) {
        rk_npu_matmul_f16_config c1;
        I.mcfg(Mw, s, c1);
        rk_npu_matmul_sizes z{};
        // exact per-shape packed size (splitk packs SPLIT aligned K-slices,
        // not 3x the plain-packed buffer - never derive it from the plain query)
        if (kUsk[s] ? rk_npu_matmul_f16_splitk_query(kSplit, &c1, &z) : rk_npu_matmul_f16_query(&c1, &z)) { I.err = "query wsz"; return 1; }
        wsz[s] = z.weight_bytes;
    }
    bool warm = !I.cache_dir.empty() && !I.fp.empty();
    char fn[512];
    for (int l = 0; warm && l < kLayers; ++l)
        for (int s = 0; s < 4; ++s) {
            snprintf(fn, sizeof(fn), "%s/%s.%d.%d.bin", I.cache_dir.c_str(), I.fp.c_str(), l, s);
            FILE* f = fopen(fn, "rb");
            if (!f) { warm = false; break; }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fclose(f);
            if (sz != (long)wsz[s]) { warm = false; break; }
        }
    std::vector<uint16_t> B;
    for (int l = 0; !warm && l < kLayers; ++l) {
        for (int s = 0; s < 4; ++s) {
            const int N = kSn[s], K = kSk[s];
            B.assign((size_t)K * N, 0);
            const LayerW& w = I.w[l];
            auto transpose = [&](const uint16_t* W, int rows, int col0) {
                for (int n = 0; n < rows; ++n)
                    for (int k = 0; k < K; ++k) B[(size_t)k * N + col0 + n] = W[(size_t)n * K + k];
            };
            switch (s) {
                // w.k == nullptr: q is the fused [q|k|v] block (audio.cpp packed store layout)
                case 0: if (w.k) { transpose(w.q, kNH * kHD, 0); transpose(w.k, kNKV * kHD, kNH * kHD); transpose(w.v, kNKV * kHD, (kNH + kNKV) * kHD); } else transpose(w.q, kSn[0], 0); break;
                case 1: transpose(w.o, kHid, 0); break;
                case 2: if (w.up) { transpose(w.gate, kInter, 0); transpose(w.up, kInter, kInter); } else transpose(w.gate, kSn[2], 0); break;
                case 3: transpose(w.down, kHid, 0); break;
            }
            rk_npu_matmul_f16_config c1;
            I.mcfg(Mw, s, c1);
            int rc = kUsk[s] ? rk_npu_matmul_f16_splitk_pack_b(kSplit, &c1, B.data(), &I.wt[l][s])
                             : rk_npu_matmul_f16_pack_b(&c1, B.data(), &I.wt[l][s]);
            if (rc) { I.err = "pack_b l" + std::to_string(l) + " s" + std::to_string(s); return 1; }
            rk_npu_mem_sync(I.ctx, &I.wt[l][s], RK_NPU_SYNC_TO_DEVICE);
            if (!I.cache_dir.empty() && !I.fp.empty()) {
                snprintf(fn, sizeof(fn), "%s/%s.%d.%d.bin", I.cache_dir.c_str(), I.fp.c_str(), l, s);
                FILE* f = fopen(fn, "wb");
                if (f) { fwrite(I.wt[l][s].vaddr, 1, (size_t)wsz[s], f); fclose(f); }
            }
        }
    }
    if (warm)
        for (int l = 0; l < kLayers; ++l)
            for (int s = 0; s < 4; ++s) {
                snprintf(fn, sizeof(fn), "%s/%s.%d.%d.bin", I.cache_dir.c_str(), I.fp.c_str(), l, s);
                FILE* f = fopen(fn, "rb");
                bool bad = !f || fread(I.wt[l][s].vaddr, 1, (size_t)wsz[s], f) != (size_t)wsz[s];
                if (f) fclose(f);
                if (bad) { I.err = std::string("cache read ") + fn; return 1; }
                rk_npu_mem_sync(I.ctx, &I.wt[l][s], RK_NPU_SYNC_TO_DEVICE);
            }
    I.packed = true;
    return 0;
}

int Engine::ensure_geometry(int seq_pad, int tgt_pad) {
    Impl& I = *im_;
    if (!I.packed) { I.err = "pack_weights first"; return 1; }
    if (seq_pad % kQrows || tgt_pad % kQrows || tgt_pad > seq_pad) { I.err = "geometry: alignment"; return 1; }
    int m1 = 2 * seq_pad;
    for (int s = 0; s < 4; ++s) {
        Impl::Shape& sb = I.sh[s];
        if (m1 > sb.m_max) {
            if (sb.in.handle) { rk_npu_mem_free(I.ctx, &sb.in); rk_npu_mem_free(I.ctx, &sb.out); sb = {}; }
            rk_npu_matmul_f16_config c1;
            I.mcfg(m1, s, c1);
            rk_npu_matmul_sizes z{};
            if (kUsk[s] ? rk_npu_matmul_f16_splitk_query(kSplit, &c1, &z) : rk_npu_matmul_f16_query(&c1, &z)) { I.err = "query geom"; return 1; }
            if (rk_npu_mem_alloc(I.dom, z.input_bytes, RK_NPU_MEM_DATA_DEFAULT, &sb.in) ||
                rk_npu_mem_alloc(I.dom, z.output_bytes, RK_NPU_MEM_DATA_DEFAULT, &sb.out)) { I.err = "alloc geom"; return 1; }
            sb.m_max = m1;
        }
    }
    for (int m : {m1, 2 * tgt_pad})
        for (int l = 0; l < kLayers; ++l)
            for (int s = 0; s < 4; ++s) {
                auto key = std::make_pair(m, l * 4 + s);
                if (I.plans.count(key)) continue;
                rk_npu_matmul_f16_config c1;
                I.mcfg(m, s, c1);
                Impl::PlanPair pp{};
                if (kUsk[s]) pp.sp = rk_npu_matmul_f16_splitk_prepare(I.dom, kSplit, &c1);
                else pp.p = rk_npu_matmul_f16_prepare(I.dom, &c1);
                if (kUsk[s] ? !pp.sp : !pp.p) { I.err = "prepare m" + std::to_string(m); return 1; }
                I.plans[key] = pp;
            }
    if (I.seq_pad < seq_pad || I.tgt_pad < tgt_pad) {
        I.seq_pad = std::max(I.seq_pad, seq_pad);
        I.tgt_pad = std::max(I.tgt_pad, tgt_pad);
        int M = 2 * I.seq_pad;
        I.x16.assign((size_t)M * kHid, 0);
        I.q16.assign((size_t)2 * kNH * I.seq_pad * kHD, 0);
        I.k16.assign((size_t)2 * kNKV * I.seq_pad * kHD, 0);
        I.v16.assign((size_t)2 * kNKV * I.seq_pad * kHD, 0);
        I.qch.assign((size_t)kNH * kQrows * kHD, 0);
        I.ctxc.assign((size_t)kNH * kQrows * kHD, 0);
        I.ctxh.assign((size_t)M * kNH * kHD, 0);
        I.mid16.assign((size_t)M * kInter, 0);
        I.qkv32.assign((size_t)M * kNqkv, 0);
        I.o32.assign((size_t)M * kHid, 0);
        I.gu32.assign((size_t)M * kNgu, 0);
        I.d32.assign((size_t)M * kHid, 0);
        I.h.assign((size_t)M * kHid, 0);
        I.ht.assign((size_t)2 * I.tgt_pad * kHid, 0);
        for (int b = 0; b < 2; ++b) {
            I.kp[b].assign((size_t)kNKV * I.seq_pad * kHD, 0);
            I.vp[b].assign((size_t)kNKV * I.seq_pad * kHD, 0);
        }
    }
    if (seq_pad > I.cap) {
        for (int b = 0; b < 2; ++b)
            for (int l = 0; l < kLayers; ++l)
                if (rk_npu_attention_f16_cache_reserve(I.ctx, I.c[b][l], seq_pad)) { I.err = "cache reserve"; return 1; }
        I.cap = seq_pad;
    }
    return 0;
}

int Engine::prefill(const float* h_canvas, const float* cos2, const float* sin2,
                    const float* mask_bias, int seq, int seq_pad, int tgt,
                    float* h_out, Timings* t) {
    Impl& I = *im_;
    (void)tgt;
    if (!I.packed || seq_pad > I.seq_pad) { I.err = "call ensure_geometry"; return 1; }
    const int sp = seq_pad, m1 = 2 * sp, prefix = seq - tgt;
    I.build_mask(mask_bias, seq, sp);
    memcpy(I.h.data(), h_canvas, (size_t)m1 * kHid * 4);
    for (int l = 0; l < kLayers; ++l) {
        rmsnorm_neon(I.h.data(), I.w[l].n_in, I.x16.data(), m1, kHid);
        if (I.gemm(l, 0, m1, I.x16.data(), I.qkv32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < sp; ++s) {
                int cix = std::min(s, seq - 1);  // pad rows clamp to last real position
                I.rope_row(l, &I.qkv32[(size_t)(b * sp + s) * kNqkv], &cos2[(size_t)(b * seq + cix) * kHD],
                           &sin2[(size_t)(b * seq + cix) * kHD],
                           &I.q16[((size_t)b * kNH) * (long)sp * kHD + (size_t)s * kHD],
                           &I.k16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)s * kHD],
                           &I.v16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)s * kHD], (long)sp * kHD);
            }
        // store canvas KV (V prescaled down by pow2 p), full canvas in caches
        for (int b = 0; b < 2; ++b) {
            uint16_t* kb = &I.k16[(size_t)b * kNKV * sp * kHD];
            uint16_t* vb = &I.v16[(size_t)b * kNKV * sp * kHD];
            float vmax = 0;
            // abs scan: the library's LUT-exp gate is max|V| (sign bit masked),
            // a signed max leaves neg-skewed layers p=1 and silently drops them
            // off the NPU LUT path into CPU exp
            for (long i = 0; i < (long)kNKV * sp * kHD; ++i) vmax = std::max(vmax, std::fabs(h2f(vb[i])));
            float p = 1.0f;
            while (vmax / p > 31.0f) p *= 2.0f;
            I.p_scale[b][l] = p;
            for (long i = 0; i < (long)kNKV * sp * kHD; ++i) {
                I.kp[b][i] = kb[i];
                I.vp[b][i] = f2h(h2f(vb[i]) / p);
            }
            if (rk_npu_attention_f16_cache_load(I.ctx, I.c[b][l], I.kp[b].data(), I.vp[b].data(), sp)) { I.err = "kv prefill load"; return 1; }
        }
        if (I.attn_rows(l, 0, 0, sp, sp, (long)sp * kHD, t) || I.attn_rows(l, 1, 0, sp, sp, (long)sp * kHD, t)) return 1;
        if (I.gemm(l, 1, m1, I.ctxh.data(), I.o32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b) {
            float p = I.p_scale[b][l];  // ctx was computed from V/p -> residual scales by p
            for (int s = 0; s < sp; ++s) {
                float* dst = &I.h[(size_t)(b * sp + s) * kHid];
                const float* src = &I.o32[(size_t)(b * sp + s) * kHid];
                for (int c = 0; c < kHid; c += 4) vst1q_f32(dst + c, vmlaq_f32(vld1q_f32(dst + c), vld1q_f32(src + c), vdupq_n_f32(p)));
            }
        }
        rmsnorm_neon(I.h.data(), I.w[l].n_post, I.x16.data(), m1, kHid);
        if (I.gemm(l, 2, m1, I.x16.data(), I.gu32.data(), t)) return 1;
        for (long r = 0; r < m1; ++r) silu_neon(&I.gu32[(size_t)r * kNgu], &I.gu32[(size_t)r * kNgu + kInter], &I.mid16[(size_t)r * kInter], kInter);
        if (I.gemm(l, 3, m1, I.mid16.data(), I.d32.data(), t)) return 1;
        for (long r = 0; r < m1; ++r)
            for (int c = 0; c < kHid; c += 4) vst1q_f32(&I.h[(size_t)r * kHid] + c, vaddq_f32(vld1q_f32(&I.h[(size_t)r * kHid] + c), vld1q_f32(&I.d32[(size_t)r * kHid] + c)));
        // caches keep only the cond prompt prefix between steps; the uncond row
        // has no prompt (its compiled keys are its own target block, rebuilt per step)
        for (int b = 0; b < 2; ++b) {
            const int pb = b == 0 ? prefix : 0;
            if (rk_npu_attention_f16_cache_truncate(I.c[b][l], pb)) { I.err = "kv prefill truncate"; return 1; }
            // repack stored prefix KV from canvas head-stride sp to prefix head-stride pb:
            // a later prescale-growth reload passes this buffer with length pb
            for (int hh = 0; hh < kNKV; ++hh)
                for (int s = 0; s < pb; ++s) {
                    // memmove: within a head the dest/src row ranges overlap (hh==0 is identical)
                    memmove(&I.kp[b][((size_t)hh * pb + s) * kHD], &I.kp[b][((size_t)hh * sp + s) * kHD], kHD * 2);
                    memmove(&I.vp[b][((size_t)hh * pb + s) * kHD], &I.vp[b][((size_t)hh * sp + s) * kHD], kHD * 2);
                }
        }
        if (I.hook) I.hook(l, I.hook_ud);
    }
    memcpy(h_out, I.h.data(), (size_t)m1 * kHid * 4);
    return 0;
}

int Engine::step(const float* h_tgt, const float* cos2, const float* sin2,
                 const float* mask_bias, int seq, int tgt, int tgt_pad,
                 float* h_out, Timings* t) {
    Impl& I = *im_;
    if (!I.packed || tgt_pad > I.tgt_pad) { I.err = "call ensure_geometry"; return 1; }
    const int sp = I.seq_pad, tp = tgt_pad, m2 = 2 * tp, prefix = seq - tgt;
    if (prefix < 0 || prefix + tp > sp) { I.err = "step: target block overruns canvas"; return 1; }
    float* hc = I.h.data();  // canvas [2,sp,kHid] seeded by prefill(); we own target rows
    I.build_mask(mask_bias, seq, sp);
    // b1's block-local pad rows [tgt,tp) are all-hidden in mask_bias (canvas space);
    // give them the pad self-diagonal, the same convention build_mask applies to
    // canvas pad rows, so they never run a fully-masked softmax
    for (int r = tgt; r < tp; ++r) I.maskb[(size_t)(sp + r) * sp + r] = 1;
    for (int b = 0; b < 2; ++b)
        for (int s = 0; s < tp; ++s)  // rows [tgt,tp) are pad (zero) -> canvas pad rows re-zeroed
            memcpy(&hc[(size_t)(b * sp + seq - tgt + s) * kHid], &h_tgt[(size_t)(b * tp + s) * kHid], kHid * 4);
    // dLLM canvas masks are self-only for every step query row (the target block is
    // unmasked one token per step, so each row attends to exactly itself). Then the
    // fused attention run is an identity on V: ctx[row] = V_scaled[row]. Detect it
    // once per call; any non-self-only mask (causal, block) keeps the NPU path.
    int selfonly = 1;
    for (int b = 0; b < 2 && selfonly; ++b) {
        const int rb0 = b == 0 ? prefix : 0;
        for (int s = 0; s < tp && selfonly; ++s) {
            const uint8_t* mrow = &I.maskb[((size_t)b * sp + rb0 + s) * sp];
            const uint8_t* v0 = (const uint8_t*)memchr(mrow, 1, (size_t)sp);
            if (!v0 || v0 != mrow + rb0 + s || (v0 + 1 < mrow + sp && memchr(v0 + 1, 1, (size_t)(sp - (rb0 + s) - 1)))) selfonly = 0;
        }
    }
    for (int l = 0; l < kLayers; ++l) {
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s)
                memcpy(&I.ht[(size_t)(b * tp + s) * kHid], &hc[(size_t)(b * sp + seq - tgt + s) * kHid], kHid * 4);
        for (int b = 0; b < 2; ++b)
            rmsnorm_neon(&I.ht[(size_t)b * tp * kHid], I.w[l].n_in, &I.x16[(size_t)b * tp * kHid], tp, kHid);
        if (I.gemm(l, 0, m2, I.x16.data(), I.qkv32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s) {
                // rope'd target rows land on their canvas rows in canvas head-stride sp,
                // exactly where attn_rows reads them: cond at [seq-tgt, seq-tgt+tp),
                // uncond block-local at [0, tp)
                int row = (b == 0 ? prefix : 0) + s;
                int cix = std::min(row, seq - 1);
                I.rope_row(l, &I.qkv32[(size_t)(b * tp + s) * kNqkv], &cos2[(size_t)(b * seq + cix) * kHD],
                           &sin2[(size_t)(b * seq + cix) * kHD],
                           &I.q16[((size_t)b * kNH) * (long)sp * kHD + (size_t)row * kHD],
                           &I.k16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)row * kHD],
                           &I.v16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)row * kHD], (long)sp * kHD);
            }
        double kv0 = now_us();
        float p_now[2];
        for (int b = 0; b < 2; ++b) {
            const int pb = b == 0 ? prefix : 0;  // uncond cache holds only the appended block
            // append scratch (contiguous [kNKV,tp,HD]) = b1's prefix-KV buffers: b1 has
            // pb=0 so kp[1]/vp[1] are dead after prefill, each is kNKV*sp*kHD >= the
            // kNKV*tp*kHD needed (tp <= sp), and cache_append is a synchronous host
            // copy, so both batches can share it (x16 would overflow when 2*tp > sp).
            // ponytail: give the scratch its own buffer if b1 ever grows a prefix.
            uint16_t* kt = I.kp[1].data();
            uint16_t* vt = I.vp[1].data();
            if (b == 1 && pb) { I.err = "step: b1 prefix shares append scratch"; return 1; }
            float vmax = 0;  // abs, see prefill: the LUT gate is max|V|
            for (long i = 0; i < (long)kNKV * pb * kHD; ++i) vmax = std::max(vmax, std::fabs(h2f(I.vp[b][i])));
            for (int hh = 0; hh < kNKV; ++hh)
                for (int s = 0; s < tp; ++s)
                    vmax = std::max(vmax, std::fabs(h2f(I.v16[((size_t)b * kNKV + hh) * sp * kHD + (size_t)(pb + s) * kHD])));
            float p = I.p_scale[b][l];
            while (vmax / p > 31.0f) p *= 2.0f;
            p_now[b] = p;
            if (p != I.p_scale[b][l]) {  // prescale grew: rescale stored V, reload prefix
                for (long i = 0; i < (long)kNKV * pb * kHD; ++i) I.vp[b][i] = f2h(h2f(I.vp[b][i]) * (I.p_scale[b][l] / p));
                I.p_scale[b][l] = p;
                if (pb && rk_npu_attention_f16_cache_load(I.ctx, I.c[b][l], I.kp[b].data(), I.vp[b].data(), pb)) { I.err = "kv reload"; return 1; }
            }
            for (int hh = 0; hh < kNKV; ++hh)
                for (int s = 0; s < tp; ++s) {
                    const uint16_t* sk = &I.k16[((size_t)b * kNKV + hh) * sp * kHD + (size_t)(pb + s) * kHD];
                    const uint16_t* sv = &I.v16[((size_t)b * kNKV + hh) * sp * kHD + (size_t)(pb + s) * kHD];
                    uint16_t* dk = &kt[((size_t)hh * tp + s) * kHD];
                    uint16_t* dv = &vt[((size_t)hh * tp + s) * kHD];
                    for (int c = 0; c < kHD; ++c) dv[c] = f2h(h2f(sv[c]) / p);
                    memcpy(dk, sk, kHD * 2);
                }
            if (rk_npu_attention_f16_cache_append(I.ctx, I.c[b][l], kt, vt, tp)) { I.err = "kv append"; return 1; }
        }
        if (t) t->kv += now_us() - kv0;
        // cond step rows read their canvas mask rows; uncond rows are block-local
        // (maskb b1 rows [0,tgt) = visible [0,tgt) = the appended target block)
        if (selfonly) {
            double a0 = now_us();
            for (int b = 0; b < 2; ++b) {
                const float inv = 1.0f / p_now[b];
                const int rb0 = b == 0 ? prefix : 0;
                for (int hh = 0; hh < kNH; ++hh)
                    for (int s = 0; s < tp; ++s) {
                        const uint16_t* sv = &I.v16[((size_t)b * kNKV + hh / 2) * (long)sp * kHD + (size_t)(rb0 + s) * kHD];
                        uint16_t* dc = &I.ctxh[(((size_t)b * tp + s) * kNH + hh) * kHD];
                        for (int c = 0; c < kHD; ++c) dc[c] = f2h(h2f(sv[c]) * inv);
                    }
            }
            if (t) t->attn += now_us() - a0;
        } else if (I.attn_rows(l, 0, seq - tgt, tp, sp, (long)sp * kHD, t) ||
                   I.attn_rows(l, 1, 0, tp, sp, (long)sp * kHD, t)) return 1;
        if (I.gemm(l, 1, m2, I.ctxh.data(), I.o32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s) {
                float* dst = &hc[(size_t)(b * sp + seq - tgt + s) * kHid];
                const float* src = &I.o32[(size_t)(b * tp + s) * kHid];
                for (int c = 0; c < kHid; c += 4) vst1q_f32(dst + c, vmlaq_f32(vld1q_f32(dst + c), vld1q_f32(src + c), vdupq_n_f32(p_now[b])));
            }
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s)
                memcpy(&I.ht[(size_t)(b * tp + s) * kHid], &hc[(size_t)(b * sp + seq - tgt + s) * kHid], kHid * 4);
        for (int b = 0; b < 2; ++b)
            rmsnorm_neon(&I.ht[(size_t)b * tp * kHid], I.w[l].n_post, &I.x16[(size_t)b * tp * kHid], tp, kHid);
        if (I.gemm(l, 2, m2, I.x16.data(), I.gu32.data(), t)) return 1;
        for (long r = 0; r < m2; ++r) silu_neon(&I.gu32[(size_t)r * kNgu], &I.gu32[(size_t)r * kNgu + kInter], &I.mid16[(size_t)r * kInter], kInter);
        if (I.gemm(l, 3, m2, I.mid16.data(), I.d32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s) {
                float* dst = &hc[(size_t)(b * sp + seq - tgt + s) * kHid];
                const float* src = &I.d32[(size_t)(b * tp + s) * kHid];
                for (int c = 0; c < kHid; c += 4) vst1q_f32(dst + c, vaddq_f32(vld1q_f32(dst + c), vld1q_f32(src + c)));
            }
        for (int b = 0; b < 2; ++b)
            if (rk_npu_attention_f16_cache_truncate(I.c[b][l], b == 0 ? prefix : 0)) { I.err = "kv step truncate"; return 1; }
        if (I.hook) I.hook(l, I.hook_ud);
    }
    for (int b = 0; b < 2; ++b)
        for (int s = 0; s < tp; ++s)
            memcpy(&h_out[(size_t)(b * tp + s) * kHid], &hc[(size_t)(b * sp + seq - tgt + s) * kHid], kHid * 4);
    return 0;
}

int Engine::caches_truncate(int len) {
    Impl& I = *im_;
    for (int b = 0; b < 2; ++b)
        for (int l = 0; l < kLayers; ++l)
            if (rk_npu_attention_f16_cache_truncate(I.c[b][l], len)) { I.err = "truncate"; return 1; }
    return 0;
}

}  // namespace ovs

#else  // !__aarch64__: the engine is ARMv8.2-A fp16 NEON and links the
       // aarch64-only rknpu2_matmul library. ENGINE_ENABLE_RKNN_OPEN is
       // arch-neutral, so non-aarch64 builds compile this stub; open() fails
       // and the generator surfaces last_error().
namespace ovs {

Engine::Engine() : im_(nullptr) {}
Engine::~Engine() {}
int Engine::open(const char*, const char*, const std::string&) { return -1; }
void Engine::close() {}
void Engine::set_weights(const std::vector<LayerW>&) {}
int Engine::pack_weights() { return -1; }
int Engine::ensure_geometry(int, int) { return -1; }
int Engine::prefill(const float*, const float*, const float*, const float*, int, int, int, float*, Timings*) { return -1; }
int Engine::step(const float*, const float*, const float*, const float*, int, int, int, float*, Timings*) { return -1; }
int Engine::caches_truncate(int) { return 0; }
const std::string& Engine::last_error() const {
    static const std::string e = "omnivoice rknn_open engine requires aarch64";
    return e;
}
void Engine::set_layer_hook(void (*)(int, void*), void*) {}
const uint16_t* Engine::dbg_q16() const { return nullptr; }
const uint16_t* Engine::dbg_k16() const { return nullptr; }
const uint16_t* Engine::dbg_ctxh() const { return nullptr; }
const float* Engine::dbg_h() const { return nullptr; }
float Engine::dbg_p(int, int) const { return 0.0f; }

}  // namespace ovs

#endif  // __aarch64__

#endif  // ENGINE_HAVE_RKNN_OPEN
