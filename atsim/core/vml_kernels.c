/*
 * Virtual Media Lab (VML) — High-Performance Native C11 + OpenMP + AVX2/SIMD Kernels
 *
 * Designed for:
 *   1. Linux / macOS shared libraries (.so / .dylib) with OpenMP + AVX2/FMA (-O3 -ffast-math -fopenmp -mavx2 -mfma)
 *   2. Windows 64-bit standalone DLL (vml_kernels_win_amd64.dll) compiled with MS x64 ABI (-D_WIN64_STANDALONE -mabi=ms)
 *      or MSVC / MinGW-w64 with zero external runtime dependencies.
 *
 * All kernels preserve the exact mathematical behavior and floating-point precision of the physical simulation stages
 * while eliminating multi-pass full-frame temporary array allocations and memory bandwidth bottlenecks.
 */

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN64_STANDALONE)
  #define VML_API __attribute__((ms_abi))
#elif defined(_WIN32) || defined(_WIN64)
  #if defined(_MSC_VER)
    #define VML_API __declspec(dllexport)
  #else
    #define VML_API __attribute__((dllexport))
  #endif
#else
  #define VML_API __attribute__((visibility("default")))
#endif

#if defined(_OPENMP) && !defined(_WIN64_STANDALONE)
  #include <omp.h>
  #define VML_HAS_OMP 1
#else
  #define VML_HAS_OMP 0
#endif

/* Inline hardware-accelerated math primitives (compile directly to SSE4.1/AVX2 instructions, zero libm calls) */
static inline float vml_minf(float a, float b) { return a < b ? a : b; }
static inline float vml_maxf(float a, float b) { return a > b ? a : b; }
static inline float vml_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline float vml_fabsf(float x) {
    return __builtin_fabsf(x);
}
static inline float vml_floorf(float x) {
#if defined(__SSE4_1__) || defined(__AVX__)
    return __builtin_floorf(x);
#else
    int32_t i = (int32_t)x;
    return (float)(i - (x < (float)i ? 1 : 0));
#endif
}

/* Matches OpenCV BORDER_REFLECT / scipy mode='reflect' ('fedcba|abcdefgh|hgfedcb') */
static inline int vml_bound_sym(int idx, int N) {
    if (idx < 0) idx = -idx - 1;
    if (idx >= N) idx = 2 * N - idx - 1;
    if (idx < 0) idx = 0;
    if (idx >= N) idx = N - 1;
    return idx;
}

#if defined(_WIN64_STANDALONE)
VML_API int DllMain(void* hinst, unsigned int reason, void* reserved) {
    (void)hinst;
    (void)reason;
    (void)reserved;
    return 1;
}
#endif

VML_API int vml_has_openmp(void) {
#if VML_HAS_OMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

/* ============================================================================
 * 1. ValueNoise3D.sample — Fused Temporal + 2D Separable Bilinear Interpolation
 * ============================================================================ */
VML_API void vml_value_noise_sample_f32(
    const float* __restrict lat0,
    const float* __restrict lat1,
    float ft,
    const int32_t* __restrict y0,
    const float* __restrict wy,
    const int32_t* __restrict x0,
    const float* __restrict wx,
    float* __restrict out,
    int H, int W, int gx
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int y = 0; y < H; ++y) {
        int iy0 = y0[y];
        int iy1 = iy0 + 1;
        float f_wy = wy[y];
        const float* l0_r0 = lat0 + (size_t)iy0 * gx;
        const float* l0_r1 = lat0 + (size_t)iy1 * gx;
        const float* l1_r0 = lat1 + (size_t)iy0 * gx;
        const float* l1_r1 = lat1 + (size_t)iy1 * gx;
        float* out_row = out + (size_t)y * W;

        float row_c[512];
        if (gx <= 512) {
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < gx; ++k) {
                float v0 = l0_r0[k] + (l1_r0[k] - l0_r0[k]) * ft;
                float v1 = l0_r1[k] + (l1_r1[k] - l0_r1[k]) * ft;
                row_c[k] = v0 + (v1 - v0) * f_wy;
            }
            for (int x = 0; x < W; ++x) {
                int ix0 = x0[x];
                float c0 = row_c[ix0];
                float c1 = row_c[ix0 + 1];
                out_row[x] = c0 + (c1 - c0) * wx[x];
            }
        } else {
            for (int x = 0; x < W; ++x) {
                int ix0 = x0[x];
                int ix1 = ix0 + 1;
                float fx = wx[x];
                float v00 = l0_r0[ix0] + (l1_r0[ix0] - l0_r0[ix0]) * ft;
                float v10 = l0_r1[ix0] + (l1_r1[ix0] - l0_r1[ix0]) * ft;
                float c0  = v00 + (v10 - v00) * f_wy;
                float v01 = l0_r0[ix1] + (l1_r0[ix1] - l0_r0[ix1]) * ft;
                float v11 = l0_r1[ix1] + (l1_r1[ix1] - l0_r1[ix1]) * ft;
                float c1  = v01 + (v11 - v01) * f_wy;
                out_row[x] = c0 + (c1 - c0) * fx;
            }
        }
    }
}

/* ============================================================================
 * 2. Colour Space & Subcarrier Modulation/Demodulation Kernels
 * ============================================================================ */
VML_API void vml_rgb_to_yiq_f32(
    const float* __restrict rgb,
    float* __restrict y_out,
    float* __restrict i_out,
    float* __restrict q_out,
    int n_pixels
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(n_pixels >= 65536)
    #endif
    for (int idx = 0; idx < n_pixels; ++idx) {
        size_t base = (size_t)idx * 3;
        float r = rgb[base + 0];
        float g = rgb[base + 1];
        float b = rgb[base + 2];
        y_out[idx] = r * 0.299f  + g * 0.587f  + b * 0.114f;
        i_out[idx] = r * 0.5959f - g * 0.2746f - b * 0.3213f;
        q_out[idx] = r * 0.2115f - g * 0.5227f + b * 0.3112f;
    }
}

VML_API void vml_yiq_to_rgb_f32(
    const float* __restrict y_in,
    const float* __restrict i_in,
    const float* __restrict q_in,
    const float* __restrict cross_luma,
    int is_pal,
    int H, int W,
    int do_clip,
    float* __restrict rgb_out
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int r = 0; r < H; ++r) {
        float q_sign = (is_pal && (r & 1)) ? -1.0f : 1.0f;
        size_t row_off = (size_t)r * W;
        const float* yp = y_in + row_off;
        const float* ip = i_in + row_off;
        const float* qp = q_in + row_off;
        const float* clp = cross_luma ? (cross_luma + row_off) : NULL;
        float* out_p = rgb_out + row_off * 3;

        if (do_clip) {
            if (clp) {
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int c = 0; c < W; ++c) {
                    float yv = yp[c] + clp[c];
                    float iv = ip[c];
                    float qv = qp[c] * q_sign;
                    float rv = yv + 0.9563f * iv + 0.6210f * qv;
                    float gv = yv - 0.2721f * iv - 0.6474f * qv;
                    float bv = yv - 1.1070f * iv + 1.7046f * qv;
                    out_p[c * 3 + 0] = vml_clampf(rv, 0.0f, 1.0f);
                    out_p[c * 3 + 1] = vml_clampf(gv, 0.0f, 1.0f);
                    out_p[c * 3 + 2] = vml_clampf(bv, 0.0f, 1.0f);
                }
            } else {
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int c = 0; c < W; ++c) {
                    float yv = yp[c];
                    float iv = ip[c];
                    float qv = qp[c] * q_sign;
                    float rv = yv + 0.9563f * iv + 0.6210f * qv;
                    float gv = yv - 0.2721f * iv - 0.6474f * qv;
                    float bv = yv - 1.1070f * iv + 1.7046f * qv;
                    out_p[c * 3 + 0] = vml_clampf(rv, 0.0f, 1.0f);
                    out_p[c * 3 + 1] = vml_clampf(gv, 0.0f, 1.0f);
                    out_p[c * 3 + 2] = vml_clampf(bv, 0.0f, 1.0f);
                }
            }
        } else {
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int c = 0; c < W; ++c) {
                float yv = clp ? (yp[c] + clp[c]) : yp[c];
                float iv = ip[c];
                float qv = qp[c] * q_sign;
                out_p[c * 3 + 0] = yv + 0.9563f * iv + 0.6210f * qv;
                out_p[c * 3 + 1] = yv - 0.2721f * iv - 0.6474f * qv;
                out_p[c * 3 + 2] = yv - 1.1070f * iv + 1.7046f * qv;
            }
        }
    }
}

VML_API void vml_composite_encode_f32(
    const float* __restrict y_in,
    const float* __restrict i_in,
    const float* __restrict q_in,
    const float* __restrict cos_ph,
    const float* __restrict sin_ph,
    float chroma_gain,
    float cb,
    float sb,
    int has_burst,
    float* __restrict comp_out,
    int n_pixels
) {
    if (!has_burst) {
        #if VML_HAS_OMP
        #pragma omp parallel for simd schedule(static) if(n_pixels >= 65536)
        #endif
        for (int k = 0; k < n_pixels; ++k) {
            comp_out[k] = y_in[k] + chroma_gain * (i_in[k] * cos_ph[k] + q_in[k] * sin_ph[k]);
        }
    } else {
        #if VML_HAS_OMP
        #pragma omp parallel for simd schedule(static) if(n_pixels >= 65536)
        #endif
        for (int k = 0; k < n_pixels; ++k) {
            float c_ph = cos_ph[k];
            float s_ph = sin_ph[k];
            float cos_tot = c_ph * cb - s_ph * sb;
            float sin_tot = s_ph * cb + c_ph * sb;
            comp_out[k] = y_in[k] + chroma_gain * (i_in[k] * cos_tot + q_in[k] * sin_tot);
        }
    }
}

VML_API void vml_remod_chroma_f32(
    const float* __restrict i_in,
    const float* __restrict q_in,
    const float* __restrict cos_ph,
    const float* __restrict sin_ph,
    float chroma_gain,
    float* __restrict out,
    int n_pixels
) {
    #if VML_HAS_OMP
    #pragma omp parallel for simd schedule(static) if(n_pixels >= 65536)
    #endif
    for (int k = 0; k < n_pixels; ++k) {
        out[k] = chroma_gain * (i_in[k] * cos_ph[k] + q_in[k] * sin_ph[k]);
    }
}

VML_API void vml_demod_scale_f32(
    const float* __restrict chroma,
    const float* __restrict cos_ph,
    const float* __restrict sin_ph,
    float scale,
    float* __restrict i_pre,
    float* __restrict q_pre,
    int n_pixels
) {
    #if VML_HAS_OMP
    #pragma omp parallel for simd schedule(static) if(n_pixels >= 65536)
    #endif
    for (int k = 0; k < n_pixels; ++k) {
        float c = chroma[k] * scale;
        i_pre[k] = c * cos_ph[k];
        q_pre[k] = c * sin_ph[k];
    }
}

VML_API void vml_comb_split_f32(
    const float* __restrict comp,
    const float* __restrict chroma_in,
    float* __restrict luma_out,
    float* __restrict chroma_out,
    int H, int W,
    float comb_depth
) {
    if (comb_depth <= 0.001f) {
        int n = H * W;
        #if VML_HAS_OMP
        #pragma omp parallel for simd schedule(static) if(n >= 65536)
        #endif
        for (int k = 0; k < n; ++k) {
            float c = chroma_in[k];
            chroma_out[k] = c;
            luma_out[k] = comp[k] - c;
        }
        return;
    }
    float gain = 1.0f - comb_depth;
    float inv_gain = (vml_fabsf(gain) > 1e-3f) ? (1.0f / gain) : 1.0f;
    float w_cur = (1.0f - 0.5f * comb_depth) * inv_gain;
    float w_prv = (0.5f * comb_depth) * inv_gain;

    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int r = 0; r < H; ++r) {
        int r_prev = (r > 0) ? (r - 1) : 0;
        const float* cp = comp + (size_t)r * W;
        const float* ch_cur = chroma_in + (size_t)r * W;
        const float* ch_prv = chroma_in + (size_t)r_prev * W;
        float* l_out = luma_out + (size_t)r * W;
        float* c_out = chroma_out + (size_t)r * W;

        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int c = 0; c < W; ++c) {
            float cm = w_cur * ch_cur[c] + w_prv * ch_prv[c];
            c_out[c] = cm;
            l_out[c] = cp[c] - cm;
        }
    }
}

/* ============================================================================
 * 3. Symmetric 1D / 2D Separable FIR & Fused Demodulation / CA Kernels
 * ============================================================================ */
static inline void vml_horiz_row_sym_impl(
    const float* __restrict s_row,
    float* __restrict t_row,
    const float* __restrict k_half,
    int R, int W, int C
) {
    float k0 = k_half[0];
    int r_lim = R < W ? R : W;
    for (int x = 0; x < r_lim; ++x) {
        for (int c = 0; c < C; ++c) {
            float acc = s_row[x * C + c] * k0;
            for (int d = 1; d <= R; ++d) {
                int xl = vml_bound_sym(x - d, W);
                int xr = vml_bound_sym(x + d, W);
                acc += (s_row[xl * C + c] + s_row[xr * C + c]) * k_half[d];
            }
            t_row[x * C + c] = acc;
        }
    }
    if (W > 2 * R) {
        int i_start = R * C;
        int i_end = (W - R) * C;
        switch (R) {
        case 1:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1];
            }
            break;
        case 2:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2];
            }
            break;
        case 3:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3];
            }
            break;
        case 4:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4];
            }
            break;
        case 5:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5];
            }
            break;
        case 6:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6];
            }
            break;
        case 7:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7];
            }
            break;
        case 8:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8];
            }
            break;
        case 9:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8] + (s_row[i - 9*C] + s_row[i + 9*C]) * k_half[9];
            }
            break;
        case 10:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8] + (s_row[i - 9*C] + s_row[i + 9*C]) * k_half[9] + (s_row[i - 10*C] + s_row[i + 10*C]) * k_half[10];
            }
            break;
        case 11:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8] + (s_row[i - 9*C] + s_row[i + 9*C]) * k_half[9] + (s_row[i - 10*C] + s_row[i + 10*C]) * k_half[10] + (s_row[i - 11*C] + s_row[i + 11*C]) * k_half[11];
            }
            break;
        case 12:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8] + (s_row[i - 9*C] + s_row[i + 9*C]) * k_half[9] + (s_row[i - 10*C] + s_row[i + 10*C]) * k_half[10] + (s_row[i - 11*C] + s_row[i + 11*C]) * k_half[11] + (s_row[i - 12*C] + s_row[i + 12*C]) * k_half[12];
            }
            break;
        case 13:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8] + (s_row[i - 9*C] + s_row[i + 9*C]) * k_half[9] + (s_row[i - 10*C] + s_row[i + 10*C]) * k_half[10] + (s_row[i - 11*C] + s_row[i + 11*C]) * k_half[11] + (s_row[i - 12*C] + s_row[i + 12*C]) * k_half[12] + (s_row[i - 13*C] + s_row[i + 13*C]) * k_half[13];
            }
            break;
        case 14:
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int i = i_start; i < i_end; ++i) {
                t_row[i] = s_row[i] * k0 + (s_row[i - 1*C] + s_row[i + 1*C]) * k_half[1] + (s_row[i - 2*C] + s_row[i + 2*C]) * k_half[2] + (s_row[i - 3*C] + s_row[i + 3*C]) * k_half[3] + (s_row[i - 4*C] + s_row[i + 4*C]) * k_half[4] + (s_row[i - 5*C] + s_row[i + 5*C]) * k_half[5] + (s_row[i - 6*C] + s_row[i + 6*C]) * k_half[6] + (s_row[i - 7*C] + s_row[i + 7*C]) * k_half[7] + (s_row[i - 8*C] + s_row[i + 8*C]) * k_half[8] + (s_row[i - 9*C] + s_row[i + 9*C]) * k_half[9] + (s_row[i - 10*C] + s_row[i + 10*C]) * k_half[10] + (s_row[i - 11*C] + s_row[i + 11*C]) * k_half[11] + (s_row[i - 12*C] + s_row[i + 12*C]) * k_half[12] + (s_row[i - 13*C] + s_row[i + 13*C]) * k_half[13] + (s_row[i - 14*C] + s_row[i + 14*C]) * k_half[14];
            }
            break;
        default: {
            float acc[256];
            int i = i_start;
            for (; i + 256 <= i_end; i += 256) {
                const float* sc = s_row + i;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < 256; ++k) acc[k] = sc[k] * k0;
                for (int d = 1; d <= R; ++d) {
                    float kd = k_half[d];
                    int off = d * C;
                    const float* sl = sc - off;
                    const float* sr = sc + off;
                    #if VML_HAS_OMP
                    #pragma omp simd
                    #endif
                    for (int k = 0; k < 256; ++k) acc[k] += (sl[k] + sr[k]) * kd;
                }
                float* tp = t_row + i;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < 256; ++k) tp[k] = acc[k];
            }
            if (i < i_end) {
                int rem = i_end - i;
                const float* sc = s_row + i;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < rem; ++k) acc[k] = sc[k] * k0;
                for (int d = 1; d <= R; ++d) {
                    float kd = k_half[d];
                    int off = d * C;
                    const float* sl = sc - off;
                    const float* sr = sc + off;
                    #if VML_HAS_OMP
                    #pragma omp simd
                    #endif
                    for (int k = 0; k < rem; ++k) acc[k] += (sl[k] + sr[k]) * kd;
                }
                float* tp = t_row + i;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < rem; ++k) tp[k] = acc[k];
            }
            break;
        }
        }
    }
    int x_right_start = (W - R > r_lim) ? (W - R) : r_lim;
    for (int x = x_right_start; x < W; ++x) {
        for (int c = 0; c < C; ++c) {
            float acc = s_row[x * C + c] * k0;
            for (int d = 1; d <= R; ++d) {
                int xl = vml_bound_sym(x - d, W);
                int xr = vml_bound_sym(x + d, W);
                acc += (s_row[xl * C + c] + s_row[xr * C + c]) * k_half[d];
            }
            t_row[x * C + c] = acc;
        }
    }
}

VML_API void vml_sym_fir_horiz_f32(
    const float* __restrict src,
    float* __restrict dst,
    const float* __restrict k_half,
    int R,
    int H, int W, int C
) {
    int stride = W * C;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int y = 0; y < H; ++y) {
        vml_horiz_row_sym_impl(src + (size_t)y * stride, dst + (size_t)y * stride, k_half, R, W, C);
    }
}

static inline void vml_vert_row_sym_impl(
    const float* __restrict src,
    const float* __restrict orig_row,
    float* __restrict d_row,
    const float* __restrict k_half,
    int R, int y, int H, int stride,
    float orig_weight, float blur_weight
) {
    float k0 = k_half[0];
    const float* s_center = src + (size_t)y * stride;
    const float* sl_ptrs[33];
    const float* sr_ptrs[33];
    int r_eff = R <= 32 ? R : 32;
    for (int d = 1; d <= r_eff; ++d) {
        sl_ptrs[d] = src + (size_t)vml_bound_sym(y - d, H) * stride;
        sr_ptrs[d] = src + (size_t)vml_bound_sym(y + d, H) * stride;
    }

    int pure_blur = (orig_weight == 0.0f && blur_weight == 1.0f);
    if (pure_blur) {
        switch (r_eff) {
            case 1: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1;
                }
                return;
            }
            case 2: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2;
                }
                return;
            }
            case 3: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3;
                }
                return;
            }
            case 4: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4;
                }
                return;
            }
            case 5: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5;
                }
                return;
            }
            case 6: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6;
                }
                return;
            }
            case 7: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7;
                }
                return;
            }
            case 8: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8;
                }
                return;
            }
            case 9: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9;
                }
                return;
            }
            case 10: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10;
                }
                return;
            }
            case 11: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11;
                }
                return;
            }
            case 12: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                const float* l12 = sl_ptrs[12]; const float* r12 = sr_ptrs[12]; float k12 = k_half[12];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11 + (l12[i] + r12[i]) * k12;
                }
                return;
            }
            case 13: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                const float* l12 = sl_ptrs[12]; const float* r12 = sr_ptrs[12]; float k12 = k_half[12];
                const float* l13 = sl_ptrs[13]; const float* r13 = sr_ptrs[13]; float k13 = k_half[13];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11 + (l12[i] + r12[i]) * k12 + (l13[i] + r13[i]) * k13;
                }
                return;
            }
            case 14: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                const float* l12 = sl_ptrs[12]; const float* r12 = sr_ptrs[12]; float k12 = k_half[12];
                const float* l13 = sl_ptrs[13]; const float* r13 = sr_ptrs[13]; float k13 = k_half[13];
                const float* l14 = sl_ptrs[14]; const float* r14 = sr_ptrs[14]; float k14 = k_half[14];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    d_row[i] = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11 + (l12[i] + r12[i]) * k12 + (l13[i] + r13[i]) * k13 + (l14[i] + r14[i]) * k14;
                }
                return;
            }
            default: break;
        }
    } else {
        switch (r_eff) {
            case 1: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 2: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 3: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 4: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 5: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 6: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 7: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 8: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 9: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 10: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 11: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 12: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                const float* l12 = sl_ptrs[12]; const float* r12 = sr_ptrs[12]; float k12 = k_half[12];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11 + (l12[i] + r12[i]) * k12;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 13: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                const float* l12 = sl_ptrs[12]; const float* r12 = sr_ptrs[12]; float k12 = k_half[12];
                const float* l13 = sl_ptrs[13]; const float* r13 = sr_ptrs[13]; float k13 = k_half[13];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11 + (l12[i] + r12[i]) * k12 + (l13[i] + r13[i]) * k13;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            case 14: {
                const float* l1 = sl_ptrs[1]; const float* r1 = sr_ptrs[1]; float k1 = k_half[1];
                const float* l2 = sl_ptrs[2]; const float* r2 = sr_ptrs[2]; float k2 = k_half[2];
                const float* l3 = sl_ptrs[3]; const float* r3 = sr_ptrs[3]; float k3 = k_half[3];
                const float* l4 = sl_ptrs[4]; const float* r4 = sr_ptrs[4]; float k4 = k_half[4];
                const float* l5 = sl_ptrs[5]; const float* r5 = sr_ptrs[5]; float k5 = k_half[5];
                const float* l6 = sl_ptrs[6]; const float* r6 = sr_ptrs[6]; float k6 = k_half[6];
                const float* l7 = sl_ptrs[7]; const float* r7 = sr_ptrs[7]; float k7 = k_half[7];
                const float* l8 = sl_ptrs[8]; const float* r8 = sr_ptrs[8]; float k8 = k_half[8];
                const float* l9 = sl_ptrs[9]; const float* r9 = sr_ptrs[9]; float k9 = k_half[9];
                const float* l10 = sl_ptrs[10]; const float* r10 = sr_ptrs[10]; float k10 = k_half[10];
                const float* l11 = sl_ptrs[11]; const float* r11 = sr_ptrs[11]; float k11 = k_half[11];
                const float* l12 = sl_ptrs[12]; const float* r12 = sr_ptrs[12]; float k12 = k_half[12];
                const float* l13 = sl_ptrs[13]; const float* r13 = sr_ptrs[13]; float k13 = k_half[13];
                const float* l14 = sl_ptrs[14]; const float* r14 = sr_ptrs[14]; float k14 = k_half[14];
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int i = 0; i < stride; ++i) {
                    float blr = s_center[i] * k0 + (l1[i] + r1[i]) * k1 + (l2[i] + r2[i]) * k2 + (l3[i] + r3[i]) * k3 + (l4[i] + r4[i]) * k4 + (l5[i] + r5[i]) * k5 + (l6[i] + r6[i]) * k6 + (l7[i] + r7[i]) * k7 + (l8[i] + r8[i]) * k8 + (l9[i] + r9[i]) * k9 + (l10[i] + r10[i]) * k10 + (l11[i] + r11[i]) * k11 + (l12[i] + r12[i]) * k12 + (l13[i] + r13[i]) * k13 + (l14[i] + r14[i]) * k14;
                    d_row[i] = orig_weight * orig_row[i] + blur_weight * blr;
                }
                return;
            }
            default: break;
        }
    }
    int i = 0;
    float acc[256];
    for (; i + 256 <= stride; i += 256) {
        const float* sc = s_center + i;
        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int k = 0; k < 256; ++k) {
            acc[k] = sc[k] * k0;
        }
        for (int d = 1; d <= r_eff; ++d) {
            float kd = k_half[d];
            const float* pl = sl_ptrs[d] + i;
            const float* pr = sr_ptrs[d] + i;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < 256; ++k) {
                acc[k] += (pl[k] + pr[k]) * kd;
            }
        }
        float* dp = d_row + i;
        if (pure_blur) {
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < 256; ++k) {
                dp[k] = acc[k];
            }
        } else {
            const float* op = orig_row + i;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < 256; ++k) {
                dp[k] = orig_weight * op[k] + blur_weight * acc[k];
            }
        }
    }
    if (i < stride) {
        int rem = stride - i;
        const float* sc = s_center + i;
        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int k = 0; k < rem; ++k) {
            acc[k] = sc[k] * k0;
        }
        for (int d = 1; d <= r_eff; ++d) {
            float kd = k_half[d];
            const float* pl = sl_ptrs[d] + i;
            const float* pr = sr_ptrs[d] + i;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < rem; ++k) {
                acc[k] += (pl[k] + pr[k]) * kd;
            }
        }
        float* dp = d_row + i;
        if (pure_blur) {
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < rem; ++k) {
                dp[k] = acc[k];
            }
        } else {
            const float* op = orig_row + i;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < rem; ++k) {
                dp[k] = orig_weight * op[k] + blur_weight * acc[k];
            }
        }
    }
}

VML_API void vml_sym_fir_vert_f32(
    const float* __restrict src,
    float* __restrict dst,
    const float* __restrict k_half,
    int R,
    int H, int W, int C,
    float orig_weight,
    float blur_weight
) {
    int stride = W * C;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int y = 0; y < H; ++y) {
        vml_vert_row_sym_impl(
            src, src + (size_t)y * stride, dst + (size_t)y * stride,
            k_half, R, y, H, stride, orig_weight, blur_weight
        );
    }
}

VML_API void vml_sym_fir_2d_f32(
    const float* __restrict src,
    float* __restrict tmp,
    float* __restrict dst,
    const float* __restrict k_half,
    int R,
    int H, int W, int C,
    float unsharp_amount
) {
    int stride = W * C;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int y = 0; y < H; ++y) {
        vml_horiz_row_sym_impl(src + (size_t)y * stride, tmp + (size_t)y * stride, k_half, R, W, C);
    }

    float orig_w = (unsharp_amount == 0.0f) ? 0.0f : (1.0f + unsharp_amount);
    float blur_w = (unsharp_amount == 0.0f) ? 1.0f : (-unsharp_amount);
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int y = 0; y < H; ++y) {
        vml_vert_row_sym_impl(
            tmp, src + (size_t)y * stride, dst + (size_t)y * stride,
            k_half, R, y, H, stride, orig_w, blur_w
        );
    }
}

VML_API void vml_demod_chroma_fused_f32(
    const float* __restrict chroma,
    const float* __restrict cos_ph,
    const float* __restrict sin_ph,
    float scale,
    const float* __restrict k_half_i, int R_i,
    const float* __restrict k_half_q, int R_q,
    float* __restrict tmp_i,
    float* __restrict tmp_q,
    float* __restrict i_out,
    float* __restrict q_out,
    int H, int W
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int y = 0; y < H; ++y) {
        size_t off = (size_t)y * W;
        const float* ch_r = chroma + off;
        const float* cs_r = cos_ph + off;
        const float* sn_r = sin_ph + off;
        float* ti = tmp_i + off;
        float* tq = tmp_q + off;
        float* oi = i_out + off;
        float* oq = q_out + off;

        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int x = 0; x < W; ++x) {
            float c = ch_r[x] * scale;
            ti[x] = c * cs_r[x];
            tq[x] = c * sn_r[x];
        }
        vml_horiz_row_sym_impl(ti, oi, k_half_i, R_i, W, 1);
        vml_horiz_row_sym_impl(tq, oq, k_half_q, R_q, W, 1);
    }
}

VML_API void vml_ca_vert_merge_f32(
    const float* __restrict tmp_r, const float* __restrict k_half_r, int R_r,
    const float* __restrict tmp_g, const float* __restrict k_half_g, int R_g,
    const float* __restrict tmp_b, const float* __restrict k_half_b, int R_b,
    float* __restrict rgb_out,
    int H, int W
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int y = 0; y < H; ++y) {
        const float* rl[33]; const float* rr[33];
        const float* gl[33]; const float* gr[33];
        const float* bl[33]; const float* br[33];
        int max_r = R_r;
        if (R_g > max_r) max_r = R_g;
        if (R_b > max_r) max_r = R_b;
        if (max_r > 32) max_r = 32;
        for (int d = 1; d <= max_r; ++d) {
            size_t yl = (size_t)vml_bound_sym(y - d, H) * W;
            size_t yr = (size_t)vml_bound_sym(y + d, H) * W;
            rl[d] = tmp_r + yl; rr[d] = tmp_r + yr;
            gl[d] = tmp_g + yl; gr[d] = tmp_g + yr;
            bl[d] = tmp_b + yl; br[d] = tmp_b + yr;
        }
        size_t off = (size_t)y * W;
        const float* r0 = tmp_r + off;
        const float* g0 = tmp_g + off;
        const float* b0 = tmp_b + off;
        float* out_row = rgb_out + off * 3;
        float kr0 = k_half_r[0], kg0 = k_half_g[0], kb0 = k_half_b[0];

        float ar[128], ag[128], ab[128];
        int x = 0;
        for (; x + 128 <= W; x += 128) {
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < 128; ++k) {
                ar[k] = r0[x + k] * kr0;
                ag[k] = g0[x + k] * kg0;
                ab[k] = b0[x + k] * kb0;
            }
            for (int d = 1; d <= R_r; ++d) {
                float kd = k_half_r[d];
                const float* pl = rl[d] + x; const float* pr = rr[d] + x;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < 128; ++k) ar[k] += (pl[k] + pr[k]) * kd;
            }
            for (int d = 1; d <= R_g; ++d) {
                float kd = k_half_g[d];
                const float* pl = gl[d] + x; const float* pr = gr[d] + x;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < 128; ++k) ag[k] += (pl[k] + pr[k]) * kd;
            }
            for (int d = 1; d <= R_b; ++d) {
                float kd = k_half_b[d];
                const float* pl = bl[d] + x; const float* pr = br[d] + x;
                #if VML_HAS_OMP
                #pragma omp simd
                #endif
                for (int k = 0; k < 128; ++k) ab[k] += (pl[k] + pr[k]) * kd;
            }
            float* op = out_row + x * 3;
            for (int k = 0; k < 128; ++k) {
                op[k * 3 + 0] = ar[k];
                op[k * 3 + 1] = ag[k];
                op[k * 3 + 2] = ab[k];
            }
        }
        if (x < W) {
            int rem = W - x;
            for (int k = 0; k < rem; ++k) {
                ar[k] = r0[x + k] * kr0;
                ag[k] = g0[x + k] * kg0;
                ab[k] = b0[x + k] * kb0;
            }
            for (int d = 1; d <= R_r; ++d) {
                float kd = k_half_r[d];
                const float* pl = rl[d] + x; const float* pr = rr[d] + x;
                for (int k = 0; k < rem; ++k) ar[k] += (pl[k] + pr[k]) * kd;
            }
            for (int d = 1; d <= R_g; ++d) {
                float kd = k_half_g[d];
                const float* pl = gl[d] + x; const float* pr = gr[d] + x;
                for (int k = 0; k < rem; ++k) ag[k] += (pl[k] + pr[k]) * kd;
            }
            for (int d = 1; d <= R_b; ++d) {
                float kd = k_half_b[d];
                const float* pl = bl[d] + x; const float* pr = br[d] + x;
                for (int k = 0; k < rem; ++k) ab[k] += (pl[k] + pr[k]) * kd;
            }
            float* op = out_row + x * 3;
            for (int k = 0; k < rem; ++k) {
                op[k * 3 + 0] = ar[k];
                op[k * 3 + 1] = ag[k];
                op[k * 3 + 2] = ab[k];
            }
        }
    }
}

/* ============================================================================
 * 4. Analog Recursive (IIR) Filters — Row-Parallel L1-Cache Resident Kernels
 * ============================================================================ */
VML_API void vml_one_pole_rows_f32(
    const float* __restrict src,
    float* __restrict dst,
    int H, int W,
    float a,
    int sections
) {
    if (W <= 0 || H <= 0) return;
    float b = 1.0f - a;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int r = 0; r < H; ++r) {
        const float* in_row = src + (size_t)r * W;
        float* out_row = dst + (size_t)r * W;
        /* Pass 0: seed from in_row[0] so steady-state DC gain is 1.0 */
        float y = in_row[0];
        out_row[0] = y;
        for (int c = 1; c < W; ++c) {
            y = a * in_row[c] + b * y;
            out_row[c] = y;
        }
        /* Subsequent cascaded sections in-place in L1 cache */
        for (int s = 1; s < sections; ++s) {
            y = out_row[0];
            for (int c = 1; c < W; ++c) {
                y = a * out_row[c] + b * y;
                out_row[c] = y;
            }
        }
    }
}

VML_API void vml_preemphasis_rows_f32(
    const float* __restrict src,
    float* __restrict dst,
    int H, int W,
    float a,
    float amount
) {
    if (W <= 0 || H <= 0) return;
    float b = 1.0f - a;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int r = 0; r < H; ++r) {
        const float* in_row = src + (size_t)r * W;
        float* out_row = dst + (size_t)r * W;
        float lp = in_row[0];
        out_row[0] = in_row[0];
        for (int c = 1; c < W; ++c) {
            float x = in_row[c];
            lp = a * x + b * lp;
            out_row[c] = x + amount * (x - lp);
        }
    }
}

VML_API void vml_biquad_rows_f32(
    const float* __restrict src,
    float* __restrict dst,
    int H, int W,
    float b0, float b1, float b2,
    float a1, float a2,
    float mix_orig, float mix_filt
) {
    if (W <= 0 || H <= 0) return;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int r = 0; r < H; ++r) {
        const float* in_row = src + (size_t)r * W;
        float* out_row = dst + (size_t)r * W;
        float z1 = 0.0f;
        float z2 = 0.0f;
        if (mix_orig == 0.0f && mix_filt == 1.0f) {
            for (int c = 0; c < W; ++c) {
                float x = in_row[c];
                float y = b0 * x + z1;
                z1 = b1 * x - a1 * y + z2;
                z2 = b2 * x - a2 * y;
                out_row[c] = y;
            }
        } else {
            for (int c = 0; c < W; ++c) {
                float x = in_row[c];
                float y = b0 * x + z1;
                z1 = b1 * x - a1 * y + z2;
                z2 = b2 * x - a2 * y;
                out_row[c] = mix_orig * x + mix_filt * y;
            }
        }
    }
}

VML_API void vml_freq_response_lpf_rows_f32(
    const float* __restrict src,
    float* __restrict dst,
    int H, int W,
    float a,
    float soft
) {
    /* Matches lfilter([1-a], [1, -a], out, axis=1) with zero initial state + (1-soft)*out + soft*filt */
    if (W <= 0 || H <= 0) return;
    float b0 = 1.0f - a;
    float w_orig = 1.0f - soft;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int r = 0; r < H; ++r) {
        const float* in_row = src + (size_t)r * W;
        float* out_row = dst + (size_t)r * W;
        float y = 0.0f;
        for (int c = 0; c < W; ++c) {
            float x = in_row[c];
            y = b0 * x + a * y;
            out_row[c] = w_orig * x + soft * y;
        }
    }
}

/* ============================================================================
 * 5. Sub-Pixel Geometric Shifts (Zero Padding Allocation)
 * ============================================================================ */
static inline int vml_bound_idx(int idx, int W, int mode) {
    if (idx >= 0 && idx < W) return idx;
    if (W <= 1) return 0;
    if (mode == 2) { /* wrap */
        int m = idx % W;
        return m < 0 ? m + W : m;
    } else if (mode == 1) { /* edge */
        return idx < 0 ? 0 : (W - 1);
    } else { /* reflect (mode == 0) */
        int period = 2 * (W - 1);
        int m = idx < 0 ? -idx : idx;
        m %= period;
        if (m >= W) m = period - m;
        return m;
    }
}

VML_API void vml_horizontal_shift_2d_f32(
    const float* __restrict src,
    float* __restrict dst,
    const float* __restrict shifts,
    int H, int W,
    int mode
) {
    if (H <= 0 || W <= 0) return;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int r = 0; r < H; ++r) {
        float sh = shifts[r];
        const float* s_row = src + (size_t)r * W;
        float* d_row = dst + (size_t)r * W;

        if (vml_fabsf(sh) <= 1e-6f) {
            for (int c = 0; c < W; ++c) d_row[c] = s_row[c];
            continue;
        }

        int32_t left = (int32_t)vml_floorf(sh);
        int32_t step = (sh > 1e-6f) ? -1 : ((sh < -1e-6f) ? 1 : 0);
        float frac = sh - (float)left;

        int min_off = -left + (step < 0 ? step : 0);
        int max_off = -left + (step > 0 ? step : 0);
        int c_start = (min_off < 0) ? -min_off : 0;
        int c_end   = (W - max_off < W) ? (W - max_off) : W;
        if (c_start > W) c_start = W;
        if (c_end < c_start) c_end = c_start;

        for (int c = 0; c < c_start; ++c) {
            int i0 = vml_bound_idx(c - left, W, mode);
            int i1 = vml_bound_idx(c - left + step, W, mode);
            float a = s_row[i0];
            float b = s_row[i1];
            d_row[c] = a + (b - a) * frac;
        }
        const float* p0 = s_row - left;
        const float* p1 = p0 + step;
        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int c = c_start; c < c_end; ++c) {
            float a = p0[c];
            float b = p1[c];
            d_row[c] = a + (b - a) * frac;
        }
        for (int c = c_end; c < W; ++c) {
            int i0 = vml_bound_idx(c - left, W, mode);
            int i1 = vml_bound_idx(c - left + step, W, mode);
            float a = s_row[i0];
            float b = s_row[i1];
            d_row[c] = a + (b - a) * frac;
        }
    }
}

VML_API void vml_horizontal_shift_3planes_f32(
    const float* __restrict y_in,
    const float* __restrict i_in,
    const float* __restrict q_in,
    float* __restrict y_out,
    float* __restrict i_out,
    float* __restrict q_out,
    const float* __restrict shifts,
    int H, int W,
    int mode
) {
    if (H <= 0 || W <= 0) return;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 16384)
    #endif
    for (int r = 0; r < H; ++r) {
        float sh = shifts[r];
        size_t off = (size_t)r * W;
        const float* sy = y_in + off;
        const float* si = i_in + off;
        const float* sq = q_in + off;
        float* dy = y_out + off;
        float* di = i_out + off;
        float* dq = q_out + off;

        if (vml_fabsf(sh) <= 1e-6f) {
            for (int c = 0; c < W; ++c) {
                dy[c] = sy[c];
                di[c] = si[c];
                dq[c] = sq[c];
            }
            continue;
        }

        int32_t left = (int32_t)vml_floorf(sh);
        int32_t step = (sh > 1e-6f) ? -1 : ((sh < -1e-6f) ? 1 : 0);
        float frac = sh - (float)left;

        int min_off = -left + (step < 0 ? step : 0);
        int max_off = -left + (step > 0 ? step : 0);
        int c_start = (min_off < 0) ? -min_off : 0;
        int c_end   = (W - max_off < W) ? (W - max_off) : W;
        if (c_start > W) c_start = W;
        if (c_end < c_start) c_end = c_start;

        for (int c = 0; c < c_start; ++c) {
            int i0 = vml_bound_idx(c - left, W, mode);
            int i1 = vml_bound_idx(c - left + step, W, mode);
            dy[c] = sy[i0] + (sy[i1] - sy[i0]) * frac;
            di[c] = si[i0] + (si[i1] - si[i0]) * frac;
            dq[c] = sq[i0] + (sq[i1] - sq[i0]) * frac;
        }
        const float* py0 = sy - left;
        const float* py1 = py0 + step;
        const float* pi0 = si - left;
        const float* pi1 = pi0 + step;
        const float* pq0 = sq - left;
        const float* pq1 = pq0 + step;
        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int c = c_start; c < c_end; ++c) {
            dy[c] = py0[c] + (py1[c] - py0[c]) * frac;
            di[c] = pi0[c] + (pi1[c] - pi0[c]) * frac;
            dq[c] = pq0[c] + (pq1[c] - pq0[c]) * frac;
        }
        for (int c = c_end; c < W; ++c) {
            int i0 = vml_bound_idx(c - left, W, mode);
            int i1 = vml_bound_idx(c - left + step, W, mode);
            dy[c] = sy[i0] + (sy[i1] - sy[i0]) * frac;
            di[c] = si[i0] + (si[i1] - si[i0]) * frac;
            dq[c] = sq[i0] + (sq[i1] - sq[i0]) * frac;
        }
    }
}

VML_API void vml_vertical_shift_2d_f32(
    const float* __restrict src,
    float* __restrict dst,
    float shift,
    int H, int W
) {
    if (H <= 0 || W <= 0) return;
    int32_t i0 = (int32_t)vml_floorf(shift);
    float frac = shift - (float)i0;
    int32_t step = (shift > 0.0f) ? 1 : -1;

    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 32768)
    #endif
    for (int r = 0; r < H; ++r) {
        int r0 = r + i0;
        int r1 = r0 + step;
        if (r0 < 0) r0 = 0; else if (r0 >= H) r0 = H - 1;
        if (r1 < 0) r1 = 0; else if (r1 >= H) r1 = H - 1;
        const float* s0 = src + (size_t)r0 * W;
        const float* s1 = src + (size_t)r1 * W;
        float* d_row = dst + (size_t)r * W;
        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int c = 0; c < W; ++c) {
            float a = s0[c];
            d_row[c] = a + (s1[c] - a) * frac;
        }
    }
}

/* ============================================================================
 * 6. Camera & Optical Kernels (Roll-Off, Noise Envelope, Color Science, CA, Smear)
 * ============================================================================ */
VML_API void vml_roll_off_f32(
    const float* __restrict src,
    float* __restrict dst,
    int n_floats,
    float knee,
    float white,
    float inv_span,
    float scale_2_1mk
) {
    #if VML_HAS_OMP
    #pragma omp parallel for simd schedule(static) if(n_floats >= 65536)
    #endif
    for (int i = 0; i < n_floats; ++i) {
        float lin = vml_clampf(src[i], 0.0f, white);
        if (lin <= knee) {
            dst[i] = lin;
        } else {
            float t = (lin - knee) * inv_span;
            dst[i] = knee + scale_2_1mk * (t / (1.0f + t));
        }
    }
}

VML_API void vml_signal_noise_envelope_f32(
    const float* __restrict rgb,
    const float* __restrict luma_lut,
    const float* __restrict chroma_lut,
    const float* __restrict sat_lut,
    float* __restrict env_luma,
    float* __restrict env_chroma,
    int n_pixels,
    int need_chroma
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(n_pixels >= 65536)
    #endif
    for (int i = 0; i < n_pixels; ++i) {
        size_t base = (size_t)i * 3;
        float r = rgb[base + 0];
        float g = rgb[base + 1];
        float b = rgb[base + 2];
        float lum = vml_clampf(r * 0.299f + g * 0.587f + b * 0.114f, 0.0f, 1.5f);
        float peak = vml_maxf(vml_maxf(r, g), b);
        float sat = vml_clampf(vml_maxf(lum, peak * 0.95f), 0.0f, 1.5f);
        int idx_l = (int)(lum * 1024.0f);
        int idx_s = (int)(sat * 1024.0f);
        if (idx_l < 0) idx_l = 0; else if (idx_l > 1536) idx_l = 1536;
        if (idx_s < 0) idx_s = 0; else if (idx_s > 1536) idx_s = 1536;
        float sc = sat_lut[idx_s];
        env_luma[i] = luma_lut[idx_l] * sc;
        if (need_chroma && env_chroma) {
            env_chroma[i] = chroma_lut[idx_l] * sc;
        }
    }
}

VML_API void vml_camera_apply_sensor_fpn_f32(
    const float* __restrict rgb_in,
    float* __restrict rgb_out,
    const float* __restrict noise_2d,
    const float* __restrict fpn_3d,
    const float* __restrict env_sn,
    float wb_r, float wb_g, float wb_b,
    float exp_scale,
    float sn_scale,
    float fp_scale,
    int n_pixels
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(n_pixels >= 65536)
    #endif
    for (int i = 0; i < n_pixels; ++i) {
        size_t base = (size_t)i * 3;
        float r = rgb_in[base + 0] * wb_r * exp_scale;
        float g = rgb_in[base + 1] * wb_g * exp_scale;
        float b = rgb_in[base + 2] * wb_b * exp_scale;
        float env = env_sn ? env_sn[i] : 0.0f;
        float add_sn = (noise_2d && sn_scale != 0.0f) ? (sn_scale * noise_2d[i] * env) : 0.0f;
        if (fpn_3d && fp_scale != 0.0f) {
            float fp_w = fp_scale * (0.25f + 0.75f * env);
            r += add_sn + fpn_3d[base + 0] * fp_w;
            g += add_sn + fpn_3d[base + 1] * fp_w;
            b += add_sn + fpn_3d[base + 2] * fp_w;
        } else {
            r += add_sn;
            g += add_sn;
            b += add_sn;
        }
        rgb_out[base + 0] = r;
        rgb_out[base + 1] = g;
        rgb_out[base + 2] = b;
    }
}

VML_API void vml_color_science_fast_f32(
    const float* __restrict rgb_in,
    float* __restrict rgb_out,
    int n_pixels,
    float gain_r, float gain_g, float gain_b,
    float ht_t,
    int is_cyan_mode,
    int clamp_01
) {
    float ct_r = 1.0f - 0.32f * ht_t;
    float ct_g = 1.0f + 0.08f * ht_t;
    float ct_b = 1.0f + 0.12f * ht_t;
    float td_r = 0.07f * ht_t;
    float td_g = -0.12f * ht_t;
    float td_b = 0.10f * ht_t;
    float inv_024 = 1.0f / 0.24f;

    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(n_pixels >= 65536)
    #endif
    for (int i = 0; i < n_pixels; ++i) {
        size_t base = (size_t)i * 3;
        float r = rgb_in[base + 0] * gain_r;
        float g = rgb_in[base + 1] * gain_g;
        float b = rgb_in[base + 2] * gain_b;
        if (ht_t > 1e-4f) {
            float lum = r * 0.299f + g * 0.587f + b * 0.114f;
            if (lum > 0.72f) {
                float over = vml_clampf((lum - 0.72f) * inv_024, 0.0f, 1.0f);
                if (is_cyan_mode) {
                    float cool = vml_clampf((b - r - 0.05f) * 5.0f, 0.0f, 1.0f);
                    float wc = over * cool;
                    r += wc * (vml_clampf(r * ct_r, 0.0f, 1.0f) - r);
                    g += wc * (vml_clampf(g * ct_g, 0.0f, 1.0f) - g);
                    b += wc * (vml_clampf(b * ct_b, 0.0f, 1.0f) - b);
                } else {
                    r += (r * td_r) * over;
                    g += (g * td_g) * over;
                    b += (b * td_b) * over;
                }
            }
        }
        if (clamp_01) {
            r = vml_clampf(r, 0.0f, 1.0f);
            g = vml_clampf(g, 0.0f, 1.0f);
            b = vml_clampf(b, 0.0f, 1.0f);
        }
        rgb_out[base + 0] = r;
        rgb_out[base + 1] = g;
        rgb_out[base + 2] = b;
    }
}

VML_API void vml_ca_warp_split_f32(
    const float* __restrict rgb,
    float* __restrict r_out,
    float* __restrict g_out,
    float* __restrict b_out,
    const int32_t* __restrict x0_r, const int32_t* __restrict x1_r, const float* __restrict fx_r,
    const int32_t* __restrict y0_r, const int32_t* __restrict y1_r, const float* __restrict fy_r,
    const int32_t* __restrict x0_b, const int32_t* __restrict x1_b, const float* __restrict fx_b,
    const int32_t* __restrict y0_b, const int32_t* __restrict y1_b, const float* __restrict fy_b,
    int H, int W
) {
    size_t stride_row = (size_t)W * 3;
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int y = 0; y < H; ++y) {
        const float* g_src_row = rgb + (size_t)y * stride_row;
        const float* r_row0 = rgb + (size_t)y0_r[y] * stride_row;
        const float* r_row1 = rgb + (size_t)y1_r[y] * stride_row;
        float wr_y = fy_r[y];

        const float* b_row0 = rgb + (size_t)y0_b[y] * stride_row;
        const float* b_row1 = rgb + (size_t)y1_b[y] * stride_row;
        float wb_y = fy_b[y];

        float* rp = r_out + (size_t)y * W;
        float* gp = g_out + (size_t)y * W;
        float* bp = b_out + (size_t)y * W;

        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int x = 0; x < W; ++x) {
            gp[x] = g_src_row[x * 3 + 1];

            int rx0 = x0_r[x] * 3;
            int rx1 = x1_r[x] * 3;
            float wr_x = fx_r[x];
            float r_c0 = r_row0[rx0] + (r_row1[rx0] - r_row0[rx0]) * wr_y;
            float r_c1 = r_row0[rx1] + (r_row1[rx1] - r_row0[rx1]) * wr_y;
            rp[x] = r_c0 + (r_c1 - r_c0) * wr_x;

            int bx0 = x0_b[x] * 3 + 2;
            int bx1 = x1_b[x] * 3 + 2;
            float wb_x = fx_b[x];
            float b_c0 = b_row0[bx0] + (b_row1[bx0] - b_row0[bx0]) * wb_y;
            float b_c1 = b_row0[bx1] + (b_row1[bx1] - b_row0[bx1]) * wb_y;
            bp[x] = b_c0 + (b_c1 - b_c0) * wb_x;
        }
    }
}

VML_API void vml_camera_wb_expose_blur_f32(
    const float* __restrict rgb_in,
    const float* __restrict prev_rgb,
    float* __restrict rgb_out,
    int n_pixels,
    float scale_r, float scale_g, float scale_b,
    float blur_k
) {
    if (prev_rgb != NULL && blur_k > 1e-4f) {
        float w_cur = 1.0f - blur_k;
        float sr = scale_r * w_cur;
        float sg = scale_g * w_cur;
        float sb = scale_b * w_cur;
        #if VML_HAS_OMP
        #pragma omp parallel for schedule(static) if(n_pixels >= 65536)
        #endif
        for (int i = 0; i < n_pixels; ++i) {
            size_t b = (size_t)i * 3;
            rgb_out[b + 0] = rgb_in[b + 0] * sr + prev_rgb[b + 0] * blur_k;
            rgb_out[b + 1] = rgb_in[b + 1] * sg + prev_rgb[b + 1] * blur_k;
            rgb_out[b + 2] = rgb_in[b + 2] * sb + prev_rgb[b + 2] * blur_k;
        }
    } else {
        #if VML_HAS_OMP
        #pragma omp parallel for schedule(static) if(n_pixels >= 65536)
        #endif
        for (int i = 0; i < n_pixels; ++i) {
            size_t b = (size_t)i * 3;
            rgb_out[b + 0] = rgb_in[b + 0] * scale_r;
            rgb_out[b + 1] = rgb_in[b + 1] * scale_g;
            rgb_out[b + 2] = rgb_in[b + 2] * scale_b;
        }
    }
}

VML_API void vml_smear_upsample_blend_f32(
    const float* __restrict rgb_in,
    const float* __restrict streak_ds,
    const float* __restrict trench_ds,
    float* __restrict rgb_out,
    int H, int W,
    int h_ds, int w_ds
) {
    float scale_y = (float)h_ds / (float)H;
    float scale_x = (float)w_ds / (float)W;

    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int y = 0; y < H; ++y) {
        float sy = ((float)y + 0.5f) * scale_y - 0.5f;
        if (sy < 0.0f) sy = 0.0f;
        int y0 = (int)sy;
        int y1 = y0 + 1;
        float fy = sy - (float)y0;
        if (y0 >= h_ds - 1) { y0 = h_ds - 1; y1 = h_ds - 1; fy = 0.0f; }

        float bead = (y & 1) ? 0.955f : 1.045f;
        const float* st_r0 = streak_ds + (size_t)y0 * w_ds * 3;
        const float* st_r1 = streak_ds + (size_t)y1 * w_ds * 3;
        const float* tr_r0 = trench_ds + (size_t)y0 * w_ds;
        const float* tr_r1 = trench_ds + (size_t)y1 * w_ds;
        const float* src_row = rgb_in + (size_t)y * W * 3;
        float* dst_row = rgb_out + (size_t)y * W * 3;

        if (W == 2 * w_ds && w_ds >= 2 && w_ds <= 1024) {
            float st_v[3072];
            float tr_v[1024];
            int n3 = w_ds * 3;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < n3; ++k) {
                st_v[k] = (st_r0[k] + (st_r1[k] - st_r0[k]) * fy) * bead;
            }
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int u = 0; u < w_ds; ++u) {
                tr_v[u] = 1.0f - (tr_r0[u] + (tr_r1[u] - tr_r0[u]) * fy);
            }
            /* x = 0 */
            {
                float k0 = tr_v[0];
                dst_row[0] = vml_clampf(vml_clampf(src_row[0], 0.0f, 2.0f) * k0 + st_v[0], 0.0f, 2.0f);
                dst_row[1] = vml_clampf(vml_clampf(src_row[1], 0.0f, 2.0f) * k0 + st_v[1], 0.0f, 2.0f);
                dst_row[2] = vml_clampf(vml_clampf(src_row[2], 0.0f, 2.0f) * k0 + st_v[2], 0.0f, 2.0f);
            }
            for (int u = 0; u < w_ds - 1; ++u) {
                float k_a = tr_v[u], k_b = tr_v[u + 1];
                float keep1 = 0.75f * k_a + 0.25f * k_b;
                float keep2 = 0.25f * k_a + 0.75f * k_b;
                const float* sa = st_v + u * 3;
                const float* sb = sa + 3;
                int xb1 = (2 * u + 1) * 3;
                int xb2 = xb1 + 3;

                dst_row[xb1 + 0] = vml_clampf(vml_clampf(src_row[xb1 + 0], 0.0f, 2.0f) * keep1 + (0.75f * sa[0] + 0.25f * sb[0]), 0.0f, 2.0f);
                dst_row[xb1 + 1] = vml_clampf(vml_clampf(src_row[xb1 + 1], 0.0f, 2.0f) * keep1 + (0.75f * sa[1] + 0.25f * sb[1]), 0.0f, 2.0f);
                dst_row[xb1 + 2] = vml_clampf(vml_clampf(src_row[xb1 + 2], 0.0f, 2.0f) * keep1 + (0.75f * sa[2] + 0.25f * sb[2]), 0.0f, 2.0f);

                dst_row[xb2 + 0] = vml_clampf(vml_clampf(src_row[xb2 + 0], 0.0f, 2.0f) * keep2 + (0.25f * sa[0] + 0.75f * sb[0]), 0.0f, 2.0f);
                dst_row[xb2 + 1] = vml_clampf(vml_clampf(src_row[xb2 + 1], 0.0f, 2.0f) * keep2 + (0.25f * sa[1] + 0.75f * sb[1]), 0.0f, 2.0f);
                dst_row[xb2 + 2] = vml_clampf(vml_clampf(src_row[xb2 + 2], 0.0f, 2.0f) * keep2 + (0.25f * sa[2] + 0.75f * sb[2]), 0.0f, 2.0f);
            }
            /* x = W - 1 */
            {
                int xb = (W - 1) * 3;
                int sb_i = (w_ds - 1) * 3;
                float kl = tr_v[w_ds - 1];
                dst_row[xb + 0] = vml_clampf(vml_clampf(src_row[xb + 0], 0.0f, 2.0f) * kl + st_v[sb_i + 0], 0.0f, 2.0f);
                dst_row[xb + 1] = vml_clampf(vml_clampf(src_row[xb + 1], 0.0f, 2.0f) * kl + st_v[sb_i + 1], 0.0f, 2.0f);
                dst_row[xb + 2] = vml_clampf(vml_clampf(src_row[xb + 2], 0.0f, 2.0f) * kl + st_v[sb_i + 2], 0.0f, 2.0f);
            }
            continue;
        }

        for (int x = 0; x < W; ++x) {
            float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            if (sx < 0.0f) sx = 0.0f;
            int x0 = (int)sx;
            int x1 = x0 + 1;
            float fx = sx - (float)x0;
            if (x0 >= w_ds - 1) { x0 = w_ds - 1; x1 = w_ds - 1; fx = 0.0f; }

            float tr0 = tr_r0[x0] + (tr_r0[x1] - tr_r0[x0]) * fx;
            float tr1 = tr_r1[x0] + (tr_r1[x1] - tr_r1[x0]) * fx;
            float keep = 1.0f - (tr0 + (tr1 - tr0) * fy);

            int x0_3 = x0 * 3;
            int x1_3 = x1 * 3;
            int xb = x * 3;

            float s0_r = st_r0[x0_3 + 0] + (st_r0[x1_3 + 0] - st_r0[x0_3 + 0]) * fx;
            float s1_r = st_r1[x0_3 + 0] + (st_r1[x1_3 + 0] - st_r1[x0_3 + 0]) * fx;
            float v_r  = (s0_r + (s1_r - s0_r) * fy) * bead;
            dst_row[xb + 0] = vml_clampf(vml_clampf(src_row[xb + 0], 0.0f, 2.0f) * keep + v_r, 0.0f, 2.0f);

            float s0_g = st_r0[x0_3 + 1] + (st_r0[x1_3 + 1] - st_r0[x0_3 + 1]) * fx;
            float s1_g = st_r1[x0_3 + 1] + (st_r1[x1_3 + 1] - st_r1[x0_3 + 1]) * fx;
            float v_g  = (s0_g + (s1_g - s0_g) * fy) * bead;
            dst_row[xb + 1] = vml_clampf(vml_clampf(src_row[xb + 1], 0.0f, 2.0f) * keep + v_g, 0.0f, 2.0f);

            float s0_b = st_r0[x0_3 + 2] + (st_r0[x1_3 + 2] - st_r0[x0_3 + 2]) * fx;
            float s1_b = st_r1[x0_3 + 2] + (st_r1[x1_3 + 2] - st_r1[x0_3 + 2]) * fx;
            float v_b  = (s0_b + (s1_b - s0_b) * fy) * bead;
            dst_row[xb + 2] = vml_clampf(vml_clampf(src_row[xb + 2], 0.0f, 2.0f) * keep + v_b, 0.0f, 2.0f);
        }
    }
}

VML_API void vml_upsample_add_rgb_f32(
    const float* __restrict rgb_in,
    const float* __restrict delta_ds,
    float* __restrict rgb_out,
    int H, int W,
    int h_ds, int w_ds
) {
    float scale_y = (float)h_ds / (float)H;
    float scale_x = (float)w_ds / (float)W;

    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int y = 0; y < H; ++y) {
        float sy = ((float)y + 0.5f) * scale_y - 0.5f;
        if (sy < 0.0f) sy = 0.0f;
        int y0 = (int)sy;
        int y1 = y0 + 1;
        float fy = sy - (float)y0;
        if (y0 >= h_ds - 1) { y0 = h_ds - 1; y1 = h_ds - 1; fy = 0.0f; }

        const float* d_r0 = delta_ds + (size_t)y0 * w_ds * 3;
        const float* d_r1 = delta_ds + (size_t)y1 * w_ds * 3;
        const float* src_row = rgb_in + (size_t)y * W * 3;
        float* dst_row = rgb_out + (size_t)y * W * 3;

        if (W == 2 * w_ds && w_ds >= 2 && w_ds <= 1024) {
            float dv[3072];
            int n3 = w_ds * 3;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int k = 0; k < n3; ++k) {
                dv[k] = d_r0[k] + (d_r1[k] - d_r0[k]) * fy;
            }
            dst_row[0] = src_row[0] + dv[0];
            dst_row[1] = src_row[1] + dv[1];
            dst_row[2] = src_row[2] + dv[2];
            for (int u = 0; u < w_ds - 1; ++u) {
                const float* da = dv + u * 3;
                const float* db = da + 3;
                int xb1 = (2 * u + 1) * 3;
                int xb2 = xb1 + 3;
                dst_row[xb1 + 0] = src_row[xb1 + 0] + (0.75f * da[0] + 0.25f * db[0]);
                dst_row[xb1 + 1] = src_row[xb1 + 1] + (0.75f * da[1] + 0.25f * db[1]);
                dst_row[xb1 + 2] = src_row[xb1 + 2] + (0.75f * da[2] + 0.25f * db[2]);

                dst_row[xb2 + 0] = src_row[xb2 + 0] + (0.25f * da[0] + 0.75f * db[0]);
                dst_row[xb2 + 1] = src_row[xb2 + 1] + (0.25f * da[1] + 0.75f * db[1]);
                dst_row[xb2 + 2] = src_row[xb2 + 2] + (0.25f * da[2] + 0.75f * db[2]);
            }
            int xb = (W - 1) * 3;
            int sb_i = (w_ds - 1) * 3;
            dst_row[xb + 0] = src_row[xb + 0] + dv[sb_i + 0];
            dst_row[xb + 1] = src_row[xb + 1] + dv[sb_i + 1];
            dst_row[xb + 2] = src_row[xb + 2] + dv[sb_i + 2];
            continue;
        }

        for (int x = 0; x < W; ++x) {
            float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            if (sx < 0.0f) sx = 0.0f;
            int x0 = (int)sx;
            int x1 = x0 + 1;
            float fx = sx - (float)x0;
            if (x0 >= w_ds - 1) { x0 = w_ds - 1; x1 = w_ds - 1; fx = 0.0f; }

            int x0_3 = x0 * 3;
            int x1_3 = x1 * 3;
            int xb = x * 3;

            float v0_r = d_r0[x0_3 + 0] + (d_r0[x1_3 + 0] - d_r0[x0_3 + 0]) * fx;
            float v1_r = d_r1[x0_3 + 0] + (d_r1[x1_3 + 0] - d_r1[x0_3 + 0]) * fx;
            dst_row[xb + 0] = src_row[xb + 0] + (v0_r + (v1_r - v0_r) * fy);

            float v0_g = d_r0[x0_3 + 1] + (d_r0[x1_3 + 1] - d_r0[x0_3 + 1]) * fx;
            float v1_g = d_r1[x0_3 + 1] + (d_r1[x1_3 + 1] - d_r1[x0_3 + 1]) * fx;
            dst_row[xb + 1] = src_row[xb + 1] + (v0_g + (v1_g - v0_g) * fy);

            float v0_b = d_r0[x0_3 + 2] + (d_r0[x1_3 + 2] - d_r0[x0_3 + 2]) * fx;
            float v1_b = d_r1[x0_3 + 2] + (d_r1[x1_3 + 2] - d_r1[x0_3 + 2]) * fx;
            dst_row[xb + 2] = src_row[xb + 2] + (v0_b + (v1_b - v0_b) * fy);
        }
    }
}

/* ============================================================================
 * 7. Format, Tape & Player Fused Kernels
 * ============================================================================ */
VML_API void vml_chroma_rotate_2d_f32(
    const float* __restrict i_in,
    const float* __restrict q_in,
    float* __restrict i_out,
    float* __restrict q_out,
    float cos_a, float sin_a,
    int n_pixels
) {
    #if VML_HAS_OMP
    #pragma omp parallel for simd schedule(static) if(n_pixels >= 65536)
    #endif
    for (int k = 0; k < n_pixels; ++k) {
        float iv = i_in[k];
        float qv = q_in[k];
        i_out[k] = iv * cos_a - qv * sin_a;
        q_out[k] = iv * sin_a + qv * cos_a;
    }
}

VML_API void vml_format_color_noise_f32(
    float* __restrict i_io,
    float* __restrict q_io,
    const float* __restrict n_field,
    const float* __restrict env_field,
    const float* __restrict cos_p,
    const float* __restrict sin_p,
    float amp,
    int H, int W
) {
    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int r = 0; r < H; ++r) {
        float cp = amp * cos_p[r];
        float sp = amp * sin_p[r];
        size_t off = (size_t)r * W;
        float* ip = i_io + off;
        float* qp = q_io + off;
        const float* np_row = n_field + off;
        const float* ep_row = env_field + off;

        #if VML_HAS_OMP
        #pragma omp simd
        #endif
        for (int c = 0; c < W; ++c) {
            float ne = np_row[c] * (0.55f + 0.75f * vml_fabsf(ep_row[c]));
            ip[c] += cp * ne;
            qp[c] += sp * ne;
        }
    }
}

VML_API void vml_tape_noise_floor_f32(
    const float* __restrict comp,
    const float* __restrict static_grain,
    const float* __restrict dyn_grain,
    const float* __restrict tilt,
    const float* __restrict striping_x,
    float amt,
    float do_color_amt,
    float* __restrict out,
    int H, int W
) {
    float w_stat = amt * 0.45f;
    float w_dyn  = amt * 0.55f;
    int has_tilt = (tilt != NULL && striping_x != NULL && vml_fabsf(do_color_amt) > 1e-8f);

    #if VML_HAS_OMP
    #pragma omp parallel for schedule(static) if(H * W >= 65536)
    #endif
    for (int r = 0; r < H; ++r) {
        size_t off = (size_t)r * W;
        const float* cp = comp + off;
        const float* sg = static_grain + off;
        const float* dg = dyn_grain + off;
        float* op = out + off;
        if (has_tilt) {
            const float* tp = tilt + off;
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int c = 0; c < W; ++c) {
                op[c] = cp[c] + w_stat * sg[c] + w_dyn * dg[c] + do_color_amt * tp[c] * striping_x[c];
            }
        } else {
            #if VML_HAS_OMP
            #pragma omp simd
            #endif
            for (int c = 0; c < W; ++c) {
                op[c] = cp[c] + w_stat * sg[c] + w_dyn * dg[c];
            }
        }
    }
}
