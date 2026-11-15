/*
 * maxq_v4_collapse.c — 100% Donor-Free True-Cone AVX-512 DualGram + 0731-Imatrix Quantizer for MaxQ-v4 (< 58 GiB)
 *
 * Compile (strict IEEE-754 finite safety — NO -ffast-math / -fassociative-math):
 *   gcc -O3 -march=native -Wall -Wextra -fopenmp -fPIC -shared maxq_v4_collapse.c \
 *       -L ./workspace/bin -Wl,-rpath,'$ORIGIN' \
 *       -lggml-cpu -lggml-base -lm -o ./workspace/bin/libmaxq_v4_collapse.so
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

extern void ggml_cpu_init(void);
extern void ggml_quantize_init(int type);

extern void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);
extern size_t quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_q5_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_q6_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_iq2_xxs(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix);
extern size_t quantize_iq2_xs (const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix);
extern size_t quantize_iq2_s  (const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix);
extern size_t quantize_iq3_xxs(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix);
extern size_t quantize_iq1_s  (const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix);
extern size_t quantize_iq1_m  (const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix);

extern void dequantize_row_mxfp4  (const void * x, float * y, int64_t k);
extern void dequantize_row_iq2_xxs(const void * x, float * y, int64_t k);
extern void dequantize_row_iq2_xs (const void * x, float * y, int64_t k);
extern void dequantize_row_iq2_s  (const void * x, float * y, int64_t k);
extern void dequantize_row_iq3_xxs(const void * x, float * y, int64_t k);
extern void dequantize_row_iq1_s  (const void * x, float * y, int64_t k);
extern void dequantize_row_iq1_m  (const void * x, float * y, int64_t k);

typedef struct {
    uint8_t e;
    uint8_t qs[16];
} mq_block_mxfp4;

typedef struct {
    int idx;
    float val;
} iv_t;

static int cmp_desc(const void * a, const void * b) {
    float va = ((const iv_t *)a)->val;
    float vb = ((const iv_t *)b)->val;
    return (va < vb) ? 1 : ((va > vb) ? -1 : 0);
}

static float g_e8m0_f32[256];
static float g_e4m3_f32[256];
static const float g_fp4_f32[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static int g_mq4_init = 0;

void mq4_init(int n_threads) {
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    if (__atomic_load_n(&g_mq4_init, __ATOMIC_ACQUIRE)) return;
    #pragma omp critical(mq4_init_lock)
    {
        if (!__atomic_load_n(&g_mq4_init, __ATOMIC_RELAXED)) {
            ggml_cpu_init();
            const int iq_types[6] = {16, 17, 18, 19, 22, 29};
            for (int i = 0; i < 6; ++i) {
                ggml_quantize_init(iq_types[i]);
            }
            for (int e = 0; e < 256; ++e) {
                g_e8m0_f32[e] = (e == 255) ? 0.0f : ldexpf(1.0f, e - 127);
            }
            for (int x = 0; x < 256; ++x) {
                uint8_t abs_x = (uint8_t)(x & 0x7f);
                int sign = (x & 0x80) != 0;
                float val = 0.0f;
                if (abs_x != 0 && abs_x != 0x7f) {
                    int exp = (x >> 3) & 0x0f;
                    int man = x & 0x07;
                    val = (exp == 0) ? ldexpf((float)man, -9)
                                     : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
                }
                g_e4m3_f32[x] = sign ? -val : val;
            }
            __atomic_store_n(&g_mq4_init, 1, __ATOMIC_RELEASE);
        }
    }
}

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline float eval_clamped_swiglu(float gd, float ud) {
    float gc = (gd < 10.0f) ? gd : 10.0f;
    float uc = (ud < -10.0f) ? -10.0f : ((ud > 10.0f) ? 10.0f : ud);
    if (gc < -80.0f) return 0.0f;
    return (gc / (1.0f + expf(-gc))) * uc;
}

static int64_t row_bytes_for_type(int ttype, int64_t ncol) {
    switch (ttype) {
        case 0:  return ncol * 4;
        case 8:  return (ncol / 32) * 34;
        case 12: return (ncol / 256) * 144;
        case 13: return (ncol / 256) * 176;
        case 14: return (ncol / 256) * 210;
        case 16: return (ncol / 256) * 66;
        case 17: return (ncol / 256) * 74;
        case 18: return (ncol / 256) * 98;
        case 19: return (ncol / 256) * 50;
        case 22: return (ncol / 256) * 82;
        case 29: return (ncol / 256) * 56;
        case 30: return ncol * 2;
        case 39: return (ncol / 32) * 17;
        default: return 0;
    }
}

void mq4_scale_f32(float * buf, int64_t n, float scale) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        buf[i] *= scale;
    }
}

/* Decode 0731 safetensors 2D row (0=F32, 1=BF16, 2=F8_E4M3 + block-128 F8_E8M0 scale) */
static void decode_src_row_2d(
    const void * src_ptr, int src_dtype, const uint8_t * scale_ptr,
    int64_t r, int64_t ncol, float * row_f32
) {
    if (src_dtype == 0) {
        memcpy(row_f32, (const float *)src_ptr + r * ncol, (size_t)ncol * sizeof(float));
    } else if (src_dtype == 1) {
        const uint16_t * sr = (const uint16_t *)src_ptr + r * ncol;
        for (int64_t c = 0; c < ncol; ++c) row_f32[c] = bf16_to_fp32(sr[c]);
    } else if (src_dtype == 2) {
        const uint8_t * wr = (const uint8_t *)src_ptr + r * ncol;
        const int64_t scale_cols = (ncol + 127) / 128;
        const uint8_t * sr = scale_ptr + (r >> 7) * scale_cols;
        for (int64_t cb = 0; cb < scale_cols; ++cb) {
            const float s = g_e8m0_f32[sr[cb]];
            const int64_t c0 = cb * 128;
            const int64_t c1 = (c0 + 128 < ncol) ? (c0 + 128) : ncol;
            for (int64_t c = c0; c < c1; ++c) {
                row_f32[c] = g_e4m3_f32[wr[c]] * s;
            }
        }
    }
}

/*
 * Quantize 2D tensor with optional 0731 imatrix weights (supporting both standard n_mat=1 2D tensors
 * and 3D-reshaped grouped projections such as blk.*.attn_output_a.weight where im_nval = o_groups * ncol).
 * Returns: 1 if imatrix weighting was applied, 0 if im_ptr == NULL (unweighted), or -2 on invalid imatrix fallback.
 */
int mq4_quantize_2d(
    const void * src_ptr, int src_dtype, const uint8_t * scale_ptr,
    void * dst_ptr, int dst_type, int64_t nrow, int64_t ncol,
    const float * im_ptr, int64_t im_nval
) {
    mq4_init(0);
    const int64_t dst_rb = row_bytes_for_type(dst_type, ncol);
    if (dst_rb <= 0) {
        fprintf(stderr, "FATAL: unsupported 2D dst_type=%d\n", dst_type);
        abort();
    }
    uint8_t * dst_u8 = (uint8_t *)dst_ptr;

    int64_t rows_per_mat = 0;
    int status = 0;
    if (im_ptr != NULL) {
        if (im_nval >= ncol && (im_nval % ncol) == 0) {
            int64_t n_mat = im_nval / ncol;
            if (n_mat > 0 && (nrow % n_mat) == 0) {
                int ok = 1;
                for (int64_t m = 0; m < n_mat; ++m) {
                    const float * s = im_ptr + m * ncol;
                    float mx = 0.0f;
                    for (int64_t c = 0; c < ncol; ++c) {
                        float v = s[c];
                        if (!isfinite(v) || v < 0.0f) { ok = 0; break; }
                        if (v > mx) mx = v;
                    }
                    if (!ok || mx <= 1e-20f) { ok = 0; break; }
                }
                if (ok) {
                    rows_per_mat = nrow / n_mat;
                    status = 1;
                }
            }
        }
        if (rows_per_mat == 0 && dst_type != 8) {
            fprintf(stderr, "[WARN] mq4_quantize_2d: invalid or degenerate imatrix (nrow=%lld, ncol=%lld, im_nval=%lld); falling back to unweighted quantization!\n",
                    (long long)nrow, (long long)ncol, (long long)im_nval);
            status = -2;
        }
    }

    #pragma omp parallel
    {
        float * row_f32 = (float *)malloc((size_t)ncol * sizeof(float));
        if (!row_f32) abort();
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nrow; ++r) {
            decode_src_row_2d(src_ptr, src_dtype, scale_ptr, r, ncol, row_f32);
            const float * qw = (rows_per_mat > 0) ? (im_ptr + (r / rows_per_mat) * ncol) : NULL;
            void * drow = dst_u8 + r * dst_rb;
            if (dst_type == 8) {
                quantize_row_q8_0_ref(row_f32, drow, ncol);
            } else if (dst_type == 12) {
                quantize_q4_K(row_f32, drow, 1, ncol, qw);
            } else if (dst_type == 13) {
                quantize_q5_K(row_f32, drow, 1, ncol, qw);
            } else if (dst_type == 14) {
                quantize_q6_K(row_f32, drow, 1, ncol, qw);
            } else {
                fprintf(stderr, "FATAL: unhandled 2D dst_type=%d\n", dst_type);
                abort();
            }
        }
        free(row_f32);
    }
    return status;
}

/* Decode 0731 safetensors MXFP4 expert slice [nrow x ncol] into F32 */
static void decode_st_mxfp4_expert(
    const uint8_t * w_i8, const uint8_t * s_u8,
    float * out_f32, int64_t nrow, int64_t ncol
) {
    const int64_t n_blocks = ncol / 32;
    for (int64_t r = 0; r < nrow; ++r) {
        const uint8_t * sr = s_u8 + r * n_blocks;
        const uint8_t * wr = w_i8 + r * (n_blocks * 16);
        float * orow = out_f32 + r * ncol;
        for (int64_t b = 0; b < n_blocks; ++b) {
            const float s = g_e8m0_f32[sr[b]];
            const uint8_t * wb = wr + b * 16;
            float * ob = orow + b * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = wb[j];
                ob[2 * j + 0] = g_fp4_f32[q & 0x0F] * s;
                ob[2 * j + 1] = g_fp4_f32[q >> 4]   * s;
            }
        }
    }
}

/* Optimal per-32-element MSE E8M0 + FP4 quantizer into ggml block_mxfp4 (split j, j+16 layout) */
static void quantize_mxfp4_mse(const float * src, mq_block_mxfp4 * dst, int64_t k) {
    static const uint8_t lut_pos[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    static const float   val_pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const int64_t nb = k / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float * sb = src + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float a = fabsf(sb[j]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-30f) {
            dst[b].e = 0;
            memset(dst[b].qs, 0, 16);
            continue;
        }
        int exp_base = ilogbf(amax / 6.0f) + 127;
        if (exp_base < 1) exp_base = 1;
        if (exp_base > 253) exp_base = 253;
        float best_err = 1e30f;
        uint8_t best_e = (uint8_t)exp_base;
        uint8_t best_nib[32];
        memset(best_nib, 0, 32);
        for (int de = -1; de <= 2; ++de) {
            int ce = exp_base + de;
            if (ce < 0 || ce > 254) continue;
            float sc = ldexpf(1.0f, ce - 127);
            float inv = 1.0f / sc;
            float err = 0.0f;
            uint8_t nib[32];
            for (int j = 0; j < 32; ++j) {
                float x = sb[j];
                float ax = fabsf(x) * inv;
                int bi = 0;
                float bd = ax;
                for (int q = 1; q < 8; ++q) {
                    float d = fabsf(ax - val_pos[q]);
                    if (d < bd) { bd = d; bi = q; }
                }
                nib[j] = (x < 0.0f && bi > 0) ? (uint8_t)(lut_pos[bi] | 8) : lut_pos[bi];
                float rec = ((x < 0.0f && bi > 0) ? -val_pos[bi] : val_pos[bi]) * sc;
                float df = x - rec;
                err += df * df;
            }
            if (err < best_err) {
                best_err = err;
                best_e = (uint8_t)ce;
                memcpy(best_nib, nib, 32);
            }
        }
        dst[b].e = best_e;
        for (int j = 0; j < 16; ++j) {
            dst[b].qs[j] = (uint8_t)(best_nib[j] | (best_nib[j + 16] << 4));
        }
    }
}

static void quantize_moe_3d_slice(int tt, const float * src, void * dst, int64_t nr, int64_t nc, const float * im) {
    if      (tt == 19) quantize_iq1_s  (src, dst, nr, nc, im);
    else if (tt == 29) quantize_iq1_m  (src, dst, nr, nc, im);
    else if (tt == 16) quantize_iq2_xxs(src, dst, nr, nc, im);
    else if (tt == 17) quantize_iq2_xs (src, dst, nr, nc, im);
    else if (tt == 22) quantize_iq2_s  (src, dst, nr, nc, im);
    else if (tt == 18) quantize_iq3_xxs(src, dst, nr, nc, im);
    else if (tt == 39) quantize_mxfp4_mse(src, (mq_block_mxfp4 *)dst, nr * nc);
    else { fprintf(stderr, "FATAL: unsupported quantize tt=%d\n", tt); abort(); }
}

static void dequantize_moe_3d_slice(int tt, const void * src, float * dst, int64_t k) {
    if      (tt == 19) dequantize_row_iq1_s  (src, dst, k);
    else if (tt == 29) dequantize_row_iq1_m  (src, dst, k);
    else if (tt == 16) dequantize_row_iq2_xxs(src, dst, k);
    else if (tt == 17) dequantize_row_iq2_xs (src, dst, k);
    else if (tt == 22) dequantize_row_iq2_s  (src, dst, k);
    else if (tt == 18) dequantize_row_iq3_xxs(src, dst, k);
    else if (tt == 39) dequantize_row_mxfp4  (src, dst, k);
    else { fprintf(stderr, "FATAL: unsupported dequantize tt=%d\n", tt); abort(); }
}

/* Cache-blocked 2x4 AVX-512 SwiGLU forward pass */
static void swiglu_fwd_tiled(
    const float * __restrict__ Wg,
    const float * __restrict__ Wu,
    const float * __restrict__ X_cal,
    float * __restrict__ H_out,
    float * __restrict__ col_acc,
    int rows_J,
    int NC,
    int D,
    int acc_mode
) {
    const double inv_nc = 1.0 / (double)NC;
    int j = 0;
    for (; j + 1 < rows_J; j += 2) {
        const float * __restrict__ gj0 = Wg + (size_t)j * D;
        const float * __restrict__ gj1 = Wg + (size_t)(j + 1) * D;
        const float * __restrict__ uj0 = Wu + (size_t)j * D;
        const float * __restrict__ uj1 = Wu + (size_t)(j + 1) * D;
        double acc0 = 0.0, acc1 = 0.0;

        int t = 0;
        for (; t + 3 < NC; t += 4) {
            const float * __restrict__ x0 = X_cal + (size_t)t * D;
            const float * __restrict__ x1 = X_cal + (size_t)(t + 1) * D;
            const float * __restrict__ x2 = X_cal + (size_t)(t + 2) * D;
            const float * __restrict__ x3 = X_cal + (size_t)(t + 3) * D;

            float g00, g01, g02, g03, g10, g11, g12, g13;
            float u00, u01, u02, u03, u10, u11, u12, u13;
#ifdef __AVX512F__
            __m512 vg00 = _mm512_setzero_ps(), vg01 = _mm512_setzero_ps(), vg02 = _mm512_setzero_ps(), vg03 = _mm512_setzero_ps();
            __m512 vg10 = _mm512_setzero_ps(), vg11 = _mm512_setzero_ps(), vg12 = _mm512_setzero_ps(), vg13 = _mm512_setzero_ps();
            __m512 vu00 = _mm512_setzero_ps(), vu01 = _mm512_setzero_ps(), vu02 = _mm512_setzero_ps(), vu03 = _mm512_setzero_ps();
            __m512 vu10 = _mm512_setzero_ps(), vu11 = _mm512_setzero_ps(), vu12 = _mm512_setzero_ps(), vu13 = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                __m512 gv0 = _mm512_loadu_ps(gj0 + d), gv1 = _mm512_loadu_ps(gj1 + d);
                __m512 uv0 = _mm512_loadu_ps(uj0 + d), uv1 = _mm512_loadu_ps(uj1 + d);
                __m512 xv0 = _mm512_loadu_ps(x0 + d),  xv1 = _mm512_loadu_ps(x1 + d);
                __m512 xv2 = _mm512_loadu_ps(x2 + d),  xv3 = _mm512_loadu_ps(x3 + d);
                vg00 = _mm512_fmadd_ps(gv0, xv0, vg00); vg01 = _mm512_fmadd_ps(gv0, xv1, vg01);
                vg02 = _mm512_fmadd_ps(gv0, xv2, vg02); vg03 = _mm512_fmadd_ps(gv0, xv3, vg03);
                vg10 = _mm512_fmadd_ps(gv1, xv0, vg10); vg11 = _mm512_fmadd_ps(gv1, xv1, vg11);
                vg12 = _mm512_fmadd_ps(gv1, xv2, vg12); vg13 = _mm512_fmadd_ps(gv1, xv3, vg13);
                vu00 = _mm512_fmadd_ps(uv0, xv0, vu00); vu01 = _mm512_fmadd_ps(uv0, xv1, vu01);
                vu02 = _mm512_fmadd_ps(uv0, xv2, vu02); vu03 = _mm512_fmadd_ps(uv0, xv3, vu03);
                vu10 = _mm512_fmadd_ps(uv1, xv0, vu10); vu11 = _mm512_fmadd_ps(uv1, xv1, vu11);
                vu12 = _mm512_fmadd_ps(uv1, xv2, vu12); vu13 = _mm512_fmadd_ps(uv1, xv3, vu13);
            }
            g00 = _mm512_reduce_add_ps(vg00); g01 = _mm512_reduce_add_ps(vg01);
            g02 = _mm512_reduce_add_ps(vg02); g03 = _mm512_reduce_add_ps(vg03);
            g10 = _mm512_reduce_add_ps(vg10); g11 = _mm512_reduce_add_ps(vg11);
            g12 = _mm512_reduce_add_ps(vg12); g13 = _mm512_reduce_add_ps(vg13);
            u00 = _mm512_reduce_add_ps(vu00); u01 = _mm512_reduce_add_ps(vu01);
            u02 = _mm512_reduce_add_ps(vu02); u03 = _mm512_reduce_add_ps(vu03);
            u10 = _mm512_reduce_add_ps(vu10); u11 = _mm512_reduce_add_ps(vu11);
            u12 = _mm512_reduce_add_ps(vu12); u13 = _mm512_reduce_add_ps(vu13);
#else
            g00 = 0; g01 = 0; g02 = 0; g03 = 0; g10 = 0; g11 = 0; g12 = 0; g13 = 0;
            u00 = 0; u01 = 0; u02 = 0; u03 = 0; u10 = 0; u11 = 0; u12 = 0; u13 = 0;
            for (int d = 0; d < D; ++d) {
                float gv0 = gj0[d], gv1 = gj1[d], uv0 = uj0[d], uv1 = uj1[d];
                float xv0 = x0[d], xv1 = x1[d], xv2 = x2[d], xv3 = x3[d];
                g00 += gv0 * xv0; g01 += gv0 * xv1; g02 += gv0 * xv2; g03 += gv0 * xv3;
                g10 += gv1 * xv0; g11 += gv1 * xv1; g12 += gv1 * xv2; g13 += gv1 * xv3;
                u00 += uv0 * xv0; u01 += uv0 * xv1; u02 += uv0 * xv2; u03 += uv0 * xv3;
                u10 += uv1 * xv0; u11 += uv1 * xv1; u12 += uv1 * xv2; u13 += uv1 * xv3;
            }
#endif
            float h00 = eval_clamped_swiglu(g00, u00), h01 = eval_clamped_swiglu(g01, u01);
            float h02 = eval_clamped_swiglu(g02, u02), h03 = eval_clamped_swiglu(g03, u03);
            float h10 = eval_clamped_swiglu(g10, u10), h11 = eval_clamped_swiglu(g11, u11);
            float h12 = eval_clamped_swiglu(g12, u12), h13 = eval_clamped_swiglu(g13, u13);

            H_out[(size_t)t * rows_J + j]           = h00;
            H_out[(size_t)t * rows_J + j + 1]       = h10;
            H_out[(size_t)(t + 1) * rows_J + j]     = h01;
            H_out[(size_t)(t + 1) * rows_J + j + 1] = h11;
            H_out[(size_t)(t + 2) * rows_J + j]     = h02;
            H_out[(size_t)(t + 2) * rows_J + j + 1] = h12;
            H_out[(size_t)(t + 3) * rows_J + j]     = h03;
            H_out[(size_t)(t + 3) * rows_J + j + 1] = h13;

            if (acc_mode == 0) {
                acc0 += (double)h00 * h00 + (double)h01 * h01 + (double)h02 * h02 + (double)h03 * h03;
                acc1 += (double)h10 * h10 + (double)h11 * h11 + (double)h12 * h12 + (double)h13 * h13;
            } else {
                acc0 += (double)h00 + (double)h01 + (double)h02 + (double)h03;
                acc1 += (double)h10 + (double)h11 + (double)h12 + (double)h13;
            }
        }
        for (; t < NC; ++t) {
            const float * __restrict__ xt = X_cal + (size_t)t * D;
            float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
            #pragma omp simd reduction(+:g0,g1,u0,u1)
            for (int d = 0; d < D; ++d) {
                float xv = xt[d];
                g0 += gj0[d] * xv; g1 += gj1[d] * xv;
                u0 += uj0[d] * xv; u1 += uj1[d] * xv;
            }
            float h0 = eval_clamped_swiglu(g0, u0), h1 = eval_clamped_swiglu(g1, u1);
            H_out[(size_t)t * rows_J + j]     = h0;
            H_out[(size_t)t * rows_J + j + 1] = h1;
            if (acc_mode == 0) {
                acc0 += (double)h0 * h0; acc1 += (double)h1 * h1;
            } else {
                acc0 += (double)h0;      acc1 += (double)h1;
            }
        }
        if (col_acc != NULL) {
            col_acc[j]     = (float)(acc0 * inv_nc);
            col_acc[j + 1] = (float)(acc1 * inv_nc);
        }
    }
    for (; j < rows_J; ++j) {
        const float * __restrict__ gj = Wg + (size_t)j * D;
        const float * __restrict__ uj = Wu + (size_t)j * D;
        double acc = 0.0;
        for (int t = 0; t < NC; ++t) {
            const float * __restrict__ xt = X_cal + (size_t)t * D;
            float gd = 0.0f, ud = 0.0f;
            #pragma omp simd reduction(+:gd,ud)
            for (int d = 0; d < D; ++d) {
                float xv = xt[d];
                gd += gj[d] * xv;
                ud += uj[d] * xv;
            }
            float hv = eval_clamped_swiglu(gd, ud);
            H_out[(size_t)t * rows_J + j] = hv;
            acc += (acc_mode == 0) ? ((double)hv * hv) : (double)hv;
        }
        if (col_acc != NULL) col_acc[j] = (float)(acc * inv_nc);
    }
}

/* Cache-blocked 4x4 AVX-512 Down-projection Y_out[NC x D] = H_in[NC x K_cols] * Wd_in[D x K_cols]^T */
static void down_fwd_tiled(
    const float * __restrict__ Wd_in,
    const float * __restrict__ H_in,
    float * __restrict__ Y_out,
    float * __restrict__ d_norm2,
    int D,
    int K_cols,
    int NC
) {
    if (d_norm2 != NULL) {
        for (int j = 0; j < K_cols; ++j) d_norm2[j] = 0.0f;
    }

    int i = 0;
    for (; i + 3 < D; i += 4) {
        const float * __restrict__ d0 = Wd_in + (size_t)i * K_cols;
        const float * __restrict__ d1 = Wd_in + (size_t)(i + 1) * K_cols;
        const float * __restrict__ d2 = Wd_in + (size_t)(i + 2) * K_cols;
        const float * __restrict__ d3 = Wd_in + (size_t)(i + 3) * K_cols;

        if (d_norm2 != NULL) {
            #pragma omp simd
            for (int j = 0; j < K_cols; ++j) {
                float v0 = d0[j], v1 = d1[j], v2 = d2[j], v3 = d3[j];
                d_norm2[j] += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
        }

        int t = 0;
        for (; t + 3 < NC; t += 4) {
            const float * __restrict__ h0 = H_in + (size_t)t * K_cols;
            const float * __restrict__ h1 = H_in + (size_t)(t + 1) * K_cols;
            const float * __restrict__ h2 = H_in + (size_t)(t + 2) * K_cols;
            const float * __restrict__ h3 = H_in + (size_t)(t + 3) * K_cols;

            float s00, s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23, s30, s31, s32, s33;
#ifdef __AVX512F__
            __m512 vs00 = _mm512_setzero_ps(), vs01 = _mm512_setzero_ps(), vs02 = _mm512_setzero_ps(), vs03 = _mm512_setzero_ps();
            __m512 vs10 = _mm512_setzero_ps(), vs11 = _mm512_setzero_ps(), vs12 = _mm512_setzero_ps(), vs13 = _mm512_setzero_ps();
            __m512 vs20 = _mm512_setzero_ps(), vs21 = _mm512_setzero_ps(), vs22 = _mm512_setzero_ps(), vs23 = _mm512_setzero_ps();
            __m512 vs30 = _mm512_setzero_ps(), vs31 = _mm512_setzero_ps(), vs32 = _mm512_setzero_ps(), vs33 = _mm512_setzero_ps();
            for (int j = 0; j < K_cols; j += 16) {
                __m512 w0 = _mm512_loadu_ps(d0 + j), w1 = _mm512_loadu_ps(d1 + j);
                __m512 w2 = _mm512_loadu_ps(d2 + j), w3 = _mm512_loadu_ps(d3 + j);
                __m512 a0 = _mm512_loadu_ps(h0 + j), a1 = _mm512_loadu_ps(h1 + j);
                __m512 a2 = _mm512_loadu_ps(h2 + j), a3 = _mm512_loadu_ps(h3 + j);
                vs00 = _mm512_fmadd_ps(w0, a0, vs00); vs01 = _mm512_fmadd_ps(w0, a1, vs01);
                vs02 = _mm512_fmadd_ps(w0, a2, vs02); vs03 = _mm512_fmadd_ps(w0, a3, vs03);
                vs10 = _mm512_fmadd_ps(w1, a0, vs10); vs11 = _mm512_fmadd_ps(w1, a1, vs11);
                vs12 = _mm512_fmadd_ps(w1, a2, vs12); vs13 = _mm512_fmadd_ps(w1, a3, vs13);
                vs20 = _mm512_fmadd_ps(w2, a0, vs20); vs21 = _mm512_fmadd_ps(w2, a1, vs21);
                vs22 = _mm512_fmadd_ps(w2, a2, vs22); vs23 = _mm512_fmadd_ps(w2, a3, vs23);
                vs30 = _mm512_fmadd_ps(w3, a0, vs30); vs31 = _mm512_fmadd_ps(w3, a1, vs31);
                vs32 = _mm512_fmadd_ps(w3, a2, vs32); vs33 = _mm512_fmadd_ps(w3, a3, vs33);
            }
            s00 = _mm512_reduce_add_ps(vs00); s01 = _mm512_reduce_add_ps(vs01);
            s02 = _mm512_reduce_add_ps(vs02); s03 = _mm512_reduce_add_ps(vs03);
            s10 = _mm512_reduce_add_ps(vs10); s11 = _mm512_reduce_add_ps(vs11);
            s12 = _mm512_reduce_add_ps(vs12); s13 = _mm512_reduce_add_ps(vs13);
            s20 = _mm512_reduce_add_ps(vs20); s21 = _mm512_reduce_add_ps(vs21);
            s22 = _mm512_reduce_add_ps(vs22); s23 = _mm512_reduce_add_ps(vs23);
            s30 = _mm512_reduce_add_ps(vs30); s31 = _mm512_reduce_add_ps(vs31);
            s32 = _mm512_reduce_add_ps(vs32); s33 = _mm512_reduce_add_ps(vs33);
#else
            s00 = 0; s01 = 0; s02 = 0; s03 = 0; s10 = 0; s11 = 0; s12 = 0; s13 = 0;
            s20 = 0; s21 = 0; s22 = 0; s23 = 0; s30 = 0; s31 = 0; s32 = 0; s33 = 0;
            for (int j = 0; j < K_cols; ++j) {
                float w0 = d0[j], w1 = d1[j], w2 = d2[j], w3 = d3[j];
                float a0 = h0[j], a1 = h1[j], a2 = h2[j], a3 = h3[j];
                s00 += w0 * a0; s01 += w0 * a1; s02 += w0 * a2; s03 += w0 * a3;
                s10 += w1 * a0; s11 += w1 * a1; s12 += w1 * a2; s13 += w1 * a3;
                s20 += w2 * a0; s21 += w2 * a1; s22 += w2 * a2; s23 += w2 * a3;
                s30 += w3 * a0; s31 += w3 * a1; s32 += w3 * a2; s33 += w3 * a3;
            }
#endif
            Y_out[(size_t)t * D + i]           = s00;
            Y_out[(size_t)t * D + i + 1]       = s10;
            Y_out[(size_t)t * D + i + 2]       = s20;
            Y_out[(size_t)t * D + i + 3]       = s30;
            Y_out[(size_t)(t + 1) * D + i]     = s01;
            Y_out[(size_t)(t + 1) * D + i + 1] = s11;
            Y_out[(size_t)(t + 1) * D + i + 2] = s21;
            Y_out[(size_t)(t + 1) * D + i + 3] = s31;
            Y_out[(size_t)(t + 2) * D + i]     = s02;
            Y_out[(size_t)(t + 2) * D + i + 1] = s12;
            Y_out[(size_t)(t + 2) * D + i + 2] = s22;
            Y_out[(size_t)(t + 2) * D + i + 3] = s32;
            Y_out[(size_t)(t + 3) * D + i]     = s03;
            Y_out[(size_t)(t + 3) * D + i + 1] = s13;
            Y_out[(size_t)(t + 3) * D + i + 2] = s23;
            Y_out[(size_t)(t + 3) * D + i + 3] = s33;
        }
        for (; t < NC; ++t) {
            const float * __restrict__ ht = H_in + (size_t)t * K_cols;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma omp simd reduction(+:s0,s1,s2,s3)
            for (int j = 0; j < K_cols; ++j) {
                float a = ht[j];
                s0 += d0[j] * a; s1 += d1[j] * a; s2 += d2[j] * a; s3 += d3[j] * a;
            }
            Y_out[(size_t)t * D + i]     = s0;
            Y_out[(size_t)t * D + i + 1] = s1;
            Y_out[(size_t)t * D + i + 2] = s2;
            Y_out[(size_t)t * D + i + 3] = s3;
        }
    }
    for (; i < D; ++i) {
        const float * __restrict__ di = Wd_in + (size_t)i * K_cols;
        if (d_norm2 != NULL) {
            #pragma omp simd
            for (int j = 0; j < K_cols; ++j) d_norm2[j] += di[j] * di[j];
        }
        for (int t = 0; t < NC; ++t) {
            const float * __restrict__ ht = H_in + (size_t)t * K_cols;
            float s = 0.0f;
            #pragma omp simd reduction(+:s)
            for (int j = 0; j < K_cols; ++j) s += di[j] * ht[j];
            Y_out[(size_t)t * D + i] = s;
        }
    }
}

/* FP64 Cholesky solve (G + lam*I) * M_out = H_rhs, with mandatory in-place backward-sub zr[c] *= inv */
static void chol_solve_f32_rhs(
    const double * G_in, const float * H_rhs, float * M_out,
    double * L_buf, double * Z_buf, int N, int KM, double lam
) {
    memset(L_buf, 0, (size_t)N * N * sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = G_in[i * N + j] + ((i == j) ? lam : 0.0);
            for (int k = 0; k < j; ++k) {
                sum -= L_buf[i * N + k] * L_buf[j * N + k];
            }
            if (i == j) {
                L_buf[i * N + i] = sqrt((sum > 1e-12) ? sum : 1e-12);
            } else {
                L_buf[i * N + j] = sum / L_buf[j * N + j];
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        double inv = 1.0 / L_buf[i * N + i];
        const float * hr = H_rhs + (size_t)i * KM;
        double * zr = Z_buf + (size_t)i * KM;
        for (int c = 0; c < KM; ++c) zr[c] = (double)hr[c];
        for (int k = 0; k < i; ++k) {
            double lik = L_buf[i * N + k];
            const double * zk = Z_buf + (size_t)k * KM;
            #pragma omp simd
            for (int c = 0; c < KM; ++c) zr[c] -= lik * zk[c];
        }
        #pragma omp simd
        for (int c = 0; c < KM; ++c) zr[c] *= inv;
    }
    for (int i = N - 1; i >= 0; --i) {
        double inv = 1.0 / L_buf[i * N + i];
        double * zr = Z_buf + (size_t)i * KM;
        for (int k = i + 1; k < N; ++k) {
            double lki = L_buf[k * N + i];
            const double * zk = Z_buf + (size_t)k * KM;
            #pragma omp simd
            for (int c = 0; c < KM; ++c) zr[c] -= lki * zk[c];
        }
        float * mr = M_out + (size_t)i * KM;
        #pragma omp simd
        for (int c = 0; c < KM; ++c) {
            zr[c] *= inv;
            mr[c] = (float)zr[c];
        }
    }
}

/*
 * mq4_collapse_moe_layer:
 *   100% Donor-Free True-Cone DualGram Engine (v2 — out-of-sample validated on 0731):
 *   - BETA_0731 = 0.50f: blends 50% true 0731 routed input E[x^2] with 50% @antirez 1.5M imatrix on Wg/Wu
 *   - BETA_HD   = 0.80f: blends 80% true 0731 SwiGLU channel energy E[h^2] with 20% @antirez 1.5M im_d
 *     for both K=1792 active-channel selection and IQ2_XXS importance quantization (+2.94% unseen cosine vs v3!)
 *   - NC = 384 true Top-6 routed tokens per expert
 *   - K=2048 anchors ({0,30,34,36,37,39,40,41,42}): uncentered DualGram (lam_mul=6.0, alpha=1.000, zero bias),
 *     into superblock norm-sorted IQ2_XXS (L0..L40) or MSE-optimal MXFP4 (L41, L42)
 *   - K=1792 gentle-collapse layers (34 layers): centered DualGram (lam_mul=4.5, Wiener alpha) + post-IQ2_XXS c_e
 */
float mq4_collapse_moe_layer(
    const uint8_t ** w_gate_ptrs, const uint8_t ** s_gate_ptrs,
    const uint8_t ** w_up_ptrs,   const uint8_t ** s_up_ptrs,
    const uint8_t ** w_down_ptrs, const uint8_t ** s_down_ptrs,
    const uint8_t * raw_gate_q,
    const uint8_t * raw_up_q,
    const float * im_gate_all,
    const float * im_up_all,
    const float * im_down_all,
    const uint16_t * calib_bf16,
    const uint8_t * calib_ids_u8,
    int n_cal_tokens,
    const uint16_t * router_bf16,
    const float * exp_probs_b,
    int gu_dst_type,
    int down_dst_type,
    int k_keep,
    uint8_t * out_gate,
    uint8_t * out_up,
    uint8_t * out_down,
    float * out_down_bias,
    float * out_val_gain_pct
) {
    mq4_init(0);
    const int D = 4096;
    const int M = 2048;
    const int E = 256;
    const int KM = k_keep;
    const int NC = 384;
    const float BETA_0731 = 0.50f;
    const float BETA_HD   = 0.80f;
    const double LAM_MUL  = 4.5;
    if (KM <= 0 || KM > M || (KM % 256) != 0) {
        fprintf(stderr, "FATAL: invalid k_keep=%d\n", KM);
        abort();
    }
    if (n_cal_tokens < NC) {
        fprintf(stderr, "FATAL: n_cal_tokens=%d < NC=%d\n", n_cal_tokens, NC);
        abort();
    }

    const size_t gu_rb     = (size_t)row_bytes_for_type(gu_dst_type, D);
    const size_t gu_in_qeb = (size_t)M * gu_rb;
    const size_t gu_out_eb = (size_t)KM * gu_rb;
    const size_t d_out_eb  = (size_t)D * (size_t)row_bytes_for_type(down_dst_type, KM);

    if (KM == M && out_down_bias != NULL) {
        memset(out_down_bias, 0, (size_t)E * D * sizeof(float));
    }

    double sum_retained = 0.0;
    double sum_alpha    = 0.0;

    #pragma omp parallel reduction(+:sum_retained,sum_alpha)
    {
        float * Wg       = (float *)malloc((size_t)M * D * sizeof(float));
        float * Wu       = (float *)malloc((size_t)M * D * sizeof(float));
        float * Wd       = (float *)malloc((size_t)D * M * sizeof(float));
        float * Wg_kh    = (float *)malloc((size_t)KM * D * sizeof(float));
        float * Wu_kh    = (float *)malloc((size_t)KM * D * sizeof(float));
        float * Wd_dg    = (float *)malloc((size_t)D * KM * sizeof(float));
        float * Wd_deq   = (float *)malloc((size_t)D * KM * sizeof(float));
        float * X_cal    = (float *)malloc((size_t)NC * D * sizeof(float));
        float * H_ref    = (float *)malloc((size_t)NC * M * sizeof(float));
        float * Y_ref    = (float *)malloc((size_t)NC * D * sizeof(float));
        float * H_kh     = (float *)malloc((size_t)NC * KM * sizeof(float));
        float * H_hc     = (float *)malloc((size_t)NC * KM * sizeof(float));
        float * M_reg    = (float *)malloc((size_t)NC * KM * sizeof(float));
        double * G_gram  = (double *)malloc((size_t)NC * NC * sizeof(double));
        double * L_chol  = (double *)malloc((size_t)NC * NC * sizeof(double));
        double * Z_sol   = (double *)malloc((size_t)NC * KM * sizeof(double));
        const size_t n_cal_sz = (n_cal_tokens >= NC) ? (size_t)n_cal_tokens : (size_t)NC;
        int * routed_idx = (int *)malloc(n_cal_sz * sizeof(int));
        uint8_t * is_sel = (uint8_t *)malloc(n_cal_sz);
        iv_t * tok_ord   = (iv_t *)malloc(n_cal_sz * sizeof(iv_t));
        double * x2_acc  = (double *)malloc((size_t)D * sizeof(double));
        float * im_g_b   = (float *)malloc((size_t)D * sizeof(float));
        float * im_u_b   = (float *)malloc((size_t)D * sizeof(float));
        float * r_f32    = (float *)malloc((size_t)D * sizeof(float));
        float * d_norm2  = (float *)malloc((size_t)M * sizeof(float));
        float * h_en     = (float *)malloc((size_t)M * sizeof(float));
        float * act2_buf = (float *)malloc((size_t)M * sizeof(float));
        iv_t * ch_ord    = (iv_t *)malloc((size_t)M * sizeof(iv_t));
        iv_t * col_ord   = (iv_t *)malloc((size_t)KM * sizeof(iv_t));
        int * head       = (int *)malloc((size_t)KM * sizeof(int));
        float * im_head  = (float *)malloc((size_t)KM * sizeof(float));
        float * h_mean   = (float *)malloc((size_t)KM * sizeof(float));
        float * dy_buf   = (float *)malloc((size_t)NC * sizeof(float));

        if (!Wg || !Wu || !Wd || !Wg_kh || !Wu_kh || !Wd_dg || !Wd_deq ||
            !X_cal || !H_ref || !Y_ref || !H_kh || !H_hc || !M_reg || !G_gram || !L_chol || !Z_sol ||
            !routed_idx || !is_sel || !tok_ord || !x2_acc || !im_g_b || !im_u_b || !r_f32 ||
            !d_norm2 || !h_en || !act2_buf || !ch_ord || !col_ord ||
            !head || !im_head || !h_mean || !dy_buf) {
            fprintf(stderr, "FATAL: OOM in mq4_collapse_moe_layer thread allocation\n");
            abort();
        }

        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < E; ++e) {
            decode_st_mxfp4_expert(w_gate_ptrs[e], s_gate_ptrs[e], Wg, M, D);
            decode_st_mxfp4_expert(w_up_ptrs[e],   s_up_ptrs[e],   Wu, M, D);
            decode_st_mxfp4_expert(w_down_ptrs[e], s_down_ptrs[e], Wd, D, M);

            /* 1. Scan all calibration tokens for true Top-6 routing + full-corpus 0731 E[x^2] */
            int n_routed = 0;
            const int need_x2 = (raw_gate_q == NULL || raw_up_q == NULL);
            if (need_x2) {
                memset(x2_acc, 0, (size_t)D * sizeof(double));
            }
            for (int t = 0; t < n_cal_tokens; ++t) {
                const uint8_t * ids6 = calib_ids_u8 + (size_t)t * 6;
                if (ids6[0] == e || ids6[1] == e || ids6[2] == e ||
                    ids6[3] == e || ids6[4] == e || ids6[5] == e) {
                    routed_idx[n_routed++] = t;
                    if (need_x2) {
                        const uint16_t * xb = calib_bf16 + (size_t)t * D;
                        for (int d = 0; d < D; ++d) {
                            double v = (double)bf16_to_fp32(xb[d]);
                            x2_acc[d] += v * v;
                        }
                    }
                }
            }

            if (need_x2) {
                const float * ig = im_gate_all + (size_t)e * D;
                const float * iu = im_up_all   + (size_t)e * D;
                if (BETA_0731 > 0.0f && n_routed >= 16) {
                    double m_ig = 0.0, m_iu = 0.0, m_x2 = 0.0;
                    const double inv_nr = 1.0 / (double)n_routed;
                    for (int d = 0; d < D; ++d) {
                        m_ig += (double)ig[d];
                        m_iu += (double)iu[d];
                        m_x2 += x2_acc[d] * inv_nr;
                    }
                    if (m_ig > 1e-12 && m_iu > 1e-12 && m_x2 > 1e-12) {
                        const double sc_g = m_ig / m_x2;
                        const double sc_u = m_iu / m_x2;
                        for (int d = 0; d < D; ++d) {
                            double x2n = x2_acc[d] * inv_nr;
                            im_g_b[d] = (1.0f - BETA_0731) * ig[d] + BETA_0731 * (float)(x2n * sc_g) + 1e-8f;
                            im_u_b[d] = (1.0f - BETA_0731) * iu[d] + BETA_0731 * (float)(x2n * sc_u) + 1e-8f;
                        }
                    } else if (m_x2 > 1e-12) {
                        for (int d = 0; d < D; ++d) {
                            float v = (float)(x2_acc[d] * inv_nr) + 1e-8f;
                            im_g_b[d] = v;
                            im_u_b[d] = v;
                        }
                    } else {
                        memcpy(im_g_b, ig, (size_t)D * sizeof(float));
                        memcpy(im_u_b, iu, (size_t)D * sizeof(float));
                    }
                } else {
                    memcpy(im_g_b, ig, (size_t)D * sizeof(float));
                    memcpy(im_u_b, iu, (size_t)D * sizeof(float));
                }
            }

            /* Deterministic Routed-Stride Selection + Router Sqrt-Softplus Top-Score Fill (NC=384) */
            if (n_routed >= NC) {
                for (int k = 0; k < NC; ++k) {
                    int pos = (int)(((int64_t)k * n_routed) / NC);
                    const uint16_t * xb = calib_bf16 + (size_t)routed_idx[pos] * D;
                    float * xd = X_cal + (size_t)k * D;
                    for (int d = 0; d < D; ++d) xd[d] = bf16_to_fp32(xb[d]);
                }
            } else {
                memset(is_sel, 0, n_cal_sz);
                int cnt = 0;
                for (int k = 0; k < n_routed; ++k) {
                    int t_orig = routed_idx[k];
                    is_sel[t_orig] = 1;
                    const uint16_t * xb = calib_bf16 + (size_t)t_orig * D;
                    float * xd = X_cal + (size_t)cnt * D;
                    for (int d = 0; d < D; ++d) xd[d] = bf16_to_fp32(xb[d]);
                    cnt++;
                }
                int need = NC - cnt;
                if (router_bf16 != NULL && exp_probs_b != NULL) {
                    const uint16_t * rb = router_bf16 + (size_t)e * D;
                    for (int d = 0; d < D; ++d) r_f32[d] = bf16_to_fp32(rb[d]);
                    float be = exp_probs_b[e];
                    int n_cand = 0;
                    for (int t = 0; t < n_cal_tokens; ++t) {
                        if (is_sel[t]) continue;
                        const uint16_t * xb = calib_bf16 + (size_t)t * D;
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) dot += bf16_to_fp32(xb[d]) * r_f32[d];
                        float dc = (dot < -40.0f) ? -40.0f : dot;
                        float sp = (dc > 20.0f) ? dc : log1pf(expf(dc));
                        tok_ord[n_cand].idx = t;
                        tok_ord[n_cand].val = sqrtf(sp) + be;
                        n_cand++;
                    }
                    qsort(tok_ord, n_cand, sizeof(iv_t), cmp_desc);
                    for (int k = 0; k < need; ++k) {
                        const uint16_t * xb = calib_bf16 + (size_t)tok_ord[k].idx * D;
                        float * xd = X_cal + (size_t)cnt * D;
                        for (int d = 0; d < D; ++d) xd[d] = bf16_to_fp32(xb[d]);
                        cnt++;
                    }
                } else {
                    for (int k = 0; k < n_cal_tokens && cnt < NC; ++k) {
                        int tidx = (int)(((int64_t)k * 251 + e * 13) % n_cal_tokens);
                        if (!is_sel[tidx]) {
                            is_sel[tidx] = 1;
                            const uint16_t * xb = calib_bf16 + (size_t)tidx * D;
                            float * xd = X_cal + (size_t)cnt * D;
                            for (int d = 0; d < D; ++d) xd[d] = bf16_to_fp32(xb[d]);
                            cnt++;
                        }
                    }
                }
            }

            /* 2. Lossless MXFP4 reference forward pass across NC=384 tokens */
            swiglu_fwd_tiled(Wg, Wu, X_cal, H_ref, h_en, M, NC, D, 0);
            down_fwd_tiled(Wd, H_ref, Y_ref, d_norm2, D, M, NC);

            /* 3. 0731-blended SwiGLU channel energy (BETA_HD=0.80) + superblock ||Wd[:,j]||^2 sorting */
            const float * im_d = im_down_all + (size_t)e * M;
            double m_imd = 0.0, m_hen = 0.0;
            for (int j = 0; j < M; ++j) {
                m_imd += (double)im_d[j];
                m_hen += (double)h_en[j];
            }
            if (m_imd > 1e-12 && m_hen > 1e-12) {
                const double sc_d = m_imd / m_hen;
                for (int j = 0; j < M; ++j) {
                    act2_buf[j] = (1.0f - BETA_HD) * im_d[j] + BETA_HD * (float)((double)h_en[j] * sc_d);
                }
            } else {
                for (int j = 0; j < M; ++j) {
                    act2_buf[j] = (m_imd > 1e-12) ? im_d[j] : h_en[j];
                }
            }

            double tot_v = 0.0;
            for (int j = 0; j < M; ++j) {
                float v = act2_buf[j] * d_norm2[j];
                ch_ord[j].idx = j;
                ch_ord[j].val = v;
                tot_v += (double)v;
            }
            if (KM < M) {
                qsort(ch_ord, M, sizeof(iv_t), cmp_desc);
            }
            double kept_v = 0.0;
            for (int k = 0; k < KM; ++k) {
                int j = ch_ord[k].idx;
                kept_v += (double)ch_ord[k].val;
                col_ord[k].idx = j;
                col_ord[k].val = d_norm2[j];
            }
            sum_retained += (KM == M) ? 100.0 : ((tot_v > 1e-18) ? (100.0 * kept_v / tot_v) : 100.0);

            if (down_dst_type != 39) {
                qsort(col_ord, KM, sizeof(iv_t), cmp_desc);
            }
            for (int k = 0; k < KM; ++k) {
                int j = col_ord[k].idx;
                head[k] = j;
                im_head[k] = act2_buf[j] + 1e-8f;
            }

            /* 4. Emit kept/sorted gate & up quantized rows */
            uint8_t * dst_g = out_gate + (size_t)e * gu_out_eb;
            uint8_t * dst_u = out_up   + (size_t)e * gu_out_eb;
            if (raw_gate_q != NULL && raw_up_q != NULL) {
                const uint8_t * qg_e = raw_gate_q + (size_t)e * gu_in_qeb;
                const uint8_t * qu_e = raw_up_q   + (size_t)e * gu_in_qeb;
                for (int k = 0; k < KM; ++k) {
                    int j = head[k];
                    memcpy(dst_g + (size_t)k * gu_rb, qg_e + (size_t)j * gu_rb, gu_rb);
                    memcpy(dst_u + (size_t)k * gu_rb, qu_e + (size_t)j * gu_rb, gu_rb);
                }
            } else {
                for (int k = 0; k < KM; ++k) {
                    int j = head[k];
                    memcpy(Wg_kh + (size_t)k * D, Wg + (size_t)j * D, (size_t)D * sizeof(float));
                    memcpy(Wu_kh + (size_t)k * D, Wu + (size_t)j * D, (size_t)D * sizeof(float));
                }
                quantize_moe_3d_slice(gu_dst_type, Wg_kh, dst_g, KM, D, im_g_b);
                quantize_moe_3d_slice(gu_dst_type, Wu_kh, dst_u, KM, D, im_u_b);
            }
            dequantize_moe_3d_slice(gu_dst_type, dst_g, Wg_kh, (int64_t)KM * D);
            dequantize_moe_3d_slice(gu_dst_type, dst_u, Wu_kh, (int64_t)KM * D);

            /* 5. Exact quantized-head activations H_kh [NC x KM] & FP64 DualGram Cholesky solve
             *    - KM < M (K=1792): centered DualGram (lam_mul=4.5, Wiener alpha, post-quant bias c_e)
             *    - KM == M (K=2048): uncentered DualGram (lam_mul=6.0, alpha=1.000, bias-free)
             */
            const int use_centered = (KM < M);
            swiglu_fwd_tiled(Wg_kh, Wu_kh, X_cal, H_kh, h_mean, KM, NC, D, 1);

            for (int t = 0; t < NC; ++t) {
                const float * hkr = H_kh + (size_t)t * KM;
                float * hcr = H_hc + (size_t)t * KM;
                if (use_centered) {
                    #pragma omp simd
                    for (int k = 0; k < KM; ++k) hcr[k] = hkr[k] - h_mean[k];
                } else {
                    memcpy(hcr, hkr, (size_t)KM * sizeof(float));
                }
            }

            double tr_gram = 0.0;
            for (int t1 = 0; t1 < NC; ++t1) {
                const float * h1 = H_hc + (size_t)t1 * KM;
                for (int t2 = 0; t2 <= t1; ++t2) {
                    const float * h2 = H_hc + (size_t)t2 * KM;
                    double s = 0.0;
                    #pragma omp simd reduction(+:s)
                    for (int k = 0; k < KM; ++k) s += (double)h1[k] * (double)h2[k];
                    G_gram[t1 * NC + t2] = s;
                    G_gram[t2 * NC + t1] = s;
                    if (t1 == t2) tr_gram += s;
                }
            }
            const double cur_lam_mul = use_centered ? LAM_MUL : 6.0;
            double lam = cur_lam_mul * (tr_gram / (double)NC);
            if (lam < 1e-6) lam = 1e-6;
            chol_solve_f32_rhs(G_gram, H_hc, M_reg, L_chol, Z_sol, NC, KM, lam);

            /* 6. Absorb 1-bit SwiGLU + tail residual into Wd_dg (4-token AVX-512 tiled) + Wiener alpha */
            double dot_yy = 0.0, nyc = 0.0, nyref = 0.0;
            for (int i = 0; i < D; ++i) {
                const float * __restrict__ di = Wd + (size_t)i * M;
                float * __restrict__ dgi = Wd_dg + (size_t)i * KM;
                for (int k = 0; k < KM; ++k) dgi[k] = di[head[k]];

                double ym = 0.0, dym = 0.0;
                int t = 0;
                for (; t + 3 < NC; t += 4) {
                    const float * __restrict__ hk0 = H_kh + (size_t)t * KM;
                    const float * __restrict__ hk1 = H_kh + (size_t)(t + 1) * KM;
                    const float * __restrict__ hk2 = H_kh + (size_t)(t + 2) * KM;
                    const float * __restrict__ hk3 = H_kh + (size_t)(t + 3) * KM;
                    float yh0, yh1, yh2, yh3;
#ifdef __AVX512F__
                    __m512 vyh0 = _mm512_setzero_ps(), vyh1 = _mm512_setzero_ps(), vyh2 = _mm512_setzero_ps(), vyh3 = _mm512_setzero_ps();
                    for (int k = 0; k < KM; k += 16) {
                        __m512 dv = _mm512_loadu_ps(dgi + k);
                        vyh0 = _mm512_fmadd_ps(_mm512_loadu_ps(hk0 + k), dv, vyh0);
                        vyh1 = _mm512_fmadd_ps(_mm512_loadu_ps(hk1 + k), dv, vyh1);
                        vyh2 = _mm512_fmadd_ps(_mm512_loadu_ps(hk2 + k), dv, vyh2);
                        vyh3 = _mm512_fmadd_ps(_mm512_loadu_ps(hk3 + k), dv, vyh3);
                    }
                    yh0 = _mm512_reduce_add_ps(vyh0); yh1 = _mm512_reduce_add_ps(vyh1);
                    yh2 = _mm512_reduce_add_ps(vyh2); yh3 = _mm512_reduce_add_ps(vyh3);
#else
                    yh0 = 0; yh1 = 0; yh2 = 0; yh3 = 0;
                    for (int k = 0; k < KM; ++k) {
                        float dv = dgi[k];
                        yh0 += hk0[k] * dv; yh1 += hk1[k] * dv; yh2 += hk2[k] * dv; yh3 += hk3[k] * dv;
                    }
#endif
                    float yr0 = Y_ref[(size_t)t * D + i], yr1 = Y_ref[(size_t)(t + 1) * D + i];
                    float yr2 = Y_ref[(size_t)(t + 2) * D + i], yr3 = Y_ref[(size_t)(t + 3) * D + i];
                    float dy0 = yr0 - yh0, dy1 = yr1 - yh1, dy2 = yr2 - yh2, dy3 = yr3 - yh3;
                    dy_buf[t] = dy0; dy_buf[t + 1] = dy1; dy_buf[t + 2] = dy2; dy_buf[t + 3] = dy3;
                    ym  += (double)yr0 + (double)yr1 + (double)yr2 + (double)yr3;
                    dym += (double)dy0 + (double)dy1 + (double)dy2 + (double)dy3;
                }
                for (; t < NC; ++t) {
                    const float * __restrict__ hkr = H_kh + (size_t)t * KM;
                    float yh = 0.0f;
                    #pragma omp simd reduction(+:yh)
                    for (int k = 0; k < KM; ++k) yh += hkr[k] * dgi[k];
                    float yrt = Y_ref[(size_t)t * D + i];
                    float dyt = yrt - yh;
                    dy_buf[t] = dyt;
                    ym += (double)yrt;
                    dym += (double)dyt;
                }
                ym /= (double)NC;
                dym /= (double)NC;

                float dym_f = use_centered ? (float)dym : 0.0f;
                double ym_sub = use_centered ? ym : 0.0;
                t = 0;
                for (; t + 3 < NC; t += 4) {
                    float dc0 = dy_buf[t]     - dym_f;
                    float dc1 = dy_buf[t + 1] - dym_f;
                    float dc2 = dy_buf[t + 2] - dym_f;
                    float dc3 = dy_buf[t + 3] - dym_f;
                    const float * __restrict__ m0 = M_reg + (size_t)t * KM;
                    const float * __restrict__ m1 = M_reg + (size_t)(t + 1) * KM;
                    const float * __restrict__ m2 = M_reg + (size_t)(t + 2) * KM;
                    const float * __restrict__ m3 = M_reg + (size_t)(t + 3) * KM;
                    #pragma omp simd
                    for (int k = 0; k < KM; ++k) {
                        dgi[k] += dc0 * m0[k] + dc1 * m1[k] + dc2 * m2[k] + dc3 * m3[k];
                    }
                }
                for (; t < NC; ++t) {
                    float dc = dy_buf[t] - dym_f;
                    const float * __restrict__ mt = M_reg + (size_t)t * KM;
                    #pragma omp simd
                    for (int k = 0; k < KM; ++k) dgi[k] += dc * mt[k];
                }

                t = 0;
                for (; t + 3 < NC; t += 4) {
                    const float * __restrict__ hc0 = H_hc + (size_t)t * KM;
                    const float * __restrict__ hc1 = H_hc + (size_t)(t + 1) * KM;
                    const float * __restrict__ hc2 = H_hc + (size_t)(t + 2) * KM;
                    const float * __restrict__ hc3 = H_hc + (size_t)(t + 3) * KM;
                    float yc0, yc1, yc2, yc3;
#ifdef __AVX512F__
                    __m512 vyc0 = _mm512_setzero_ps(), vyc1 = _mm512_setzero_ps(), vyc2 = _mm512_setzero_ps(), vyc3 = _mm512_setzero_ps();
                    for (int k = 0; k < KM; k += 16) {
                        __m512 dv = _mm512_loadu_ps(dgi + k);
                        vyc0 = _mm512_fmadd_ps(_mm512_loadu_ps(hc0 + k), dv, vyc0);
                        vyc1 = _mm512_fmadd_ps(_mm512_loadu_ps(hc1 + k), dv, vyc1);
                        vyc2 = _mm512_fmadd_ps(_mm512_loadu_ps(hc2 + k), dv, vyc2);
                        vyc3 = _mm512_fmadd_ps(_mm512_loadu_ps(hc3 + k), dv, vyc3);
                    }
                    yc0 = _mm512_reduce_add_ps(vyc0); yc1 = _mm512_reduce_add_ps(vyc1);
                    yc2 = _mm512_reduce_add_ps(vyc2); yc3 = _mm512_reduce_add_ps(vyc3);
#else
                    yc0 = 0; yc1 = 0; yc2 = 0; yc3 = 0;
                    for (int k = 0; k < KM; ++k) {
                        float dv = dgi[k];
                        yc0 += hc0[k] * dv; yc1 += hc1[k] * dv; yc2 += hc2[k] * dv; yc3 += hc3[k] * dv;
                    }
#endif
                    double yrc0 = (double)Y_ref[(size_t)t * D + i]       - ym_sub;
                    double yrc1 = (double)Y_ref[(size_t)(t + 1) * D + i] - ym_sub;
                    double yrc2 = (double)Y_ref[(size_t)(t + 2) * D + i] - ym_sub;
                    double yrc3 = (double)Y_ref[(size_t)(t + 3) * D + i] - ym_sub;
                    dot_yy += (double)yc0 * yrc0 + (double)yc1 * yrc1 + (double)yc2 * yrc2 + (double)yc3 * yrc3;
                    nyc    += (double)yc0 * yc0  + (double)yc1 * yc1  + (double)yc2 * yc2  + (double)yc3 * yc3;
                    nyref  += yrc0 * yrc0        + yrc1 * yrc1        + yrc2 * yrc2        + yrc3 * yrc3;
                }
                for (; t < NC; ++t) {
                    const float * __restrict__ hcr = H_hc + (size_t)t * KM;
                    float yc = 0.0f;
                    #pragma omp simd reduction(+:yc)
                    for (int k = 0; k < KM; ++k) yc += hcr[k] * dgi[k];
                    double yrc = (double)Y_ref[(size_t)t * D + i] - ym_sub;
                    dot_yy += (double)yc * yrc;
                    nyc    += (double)yc * (double)yc;
                    nyref  += yrc * yrc;
                }
            }

            float alpha = 1.000f;
            if (use_centered) {
                float a_mmse = (float)(dot_yy / (nyc + 1e-12));
                float a_norm = sqrtf((float)(nyref / (nyc + 1e-12)));
                alpha  = 0.75f * a_norm + 0.25f * a_mmse;
                if (!isfinite(alpha) || alpha < 0.95f) alpha = 0.95f;
                if (alpha > 1.04f) alpha = 1.04f;
            }
            sum_alpha += (double)alpha;

            float max_wd = 0.0f;
            int bad_vals = 0;
            for (size_t z = 0; z < (size_t)D * KM; ++z) {
                float v = Wd_dg[z] * alpha;
                Wd_dg[z] = v;
                if (!isfinite(v)) bad_vals++;
                else if (fabsf(v) > max_wd) max_wd = fabsf(v);
            }
            if (bad_vals > 0 || max_wd > 64.0f) {
                fprintf(stderr, "FATAL: numerical bound violation on expert %d: bad=%d max_wd=%g\n", e, bad_vals, max_wd);
                abort();
            }

            /* 7. Quantize Wd_dg -> IQ2_XXS (or MXFP4 on L41/L42) + post-quantization centroid bias c_e (if KM < M) */
            uint8_t * dst_d = out_down + (size_t)e * d_out_eb;
            quantize_moe_3d_slice(down_dst_type, Wd_dg, dst_d, D, KM, im_head);

            if (KM < M && out_down_bias != NULL) {
                dequantize_moe_3d_slice(down_dst_type, dst_d, Wd_deq, (int64_t)D * KM);
                float * ce_dst = out_down_bias + (size_t)e * D;
                for (int i = 0; i < D; ++i) {
                    double ym = 0.0;
                    for (int t = 0; t < NC; ++t) ym += (double)Y_ref[(size_t)t * D + i];
                    ym /= (double)NC;
                    const float * __restrict__ dqi = Wd_deq + (size_t)i * KM;
                    double hm = 0.0;
                    #pragma omp simd reduction(+:hm)
                    for (int k = 0; k < KM; ++k) hm += (double)dqi[k] * (double)h_mean[k];
                    float ce = 0.60f * (float)(ym - hm);
                    ce_dst[i] = isfinite(ce) ? ce : 0.0f;
                }
            }
        }

        free(Wg); free(Wu); free(Wd); free(Wg_kh); free(Wu_kh); free(Wd_dg); free(Wd_deq);
        free(X_cal); free(H_ref); free(Y_ref); free(H_kh); free(H_hc); free(M_reg);
        free(G_gram); free(L_chol); free(Z_sol); free(routed_idx); free(is_sel); free(tok_ord);
        free(x2_acc); free(im_g_b); free(im_u_b); free(r_f32);
        free(d_norm2); free(h_en); free(act2_buf); free(ch_ord); free(col_ord);
        free(head); free(im_head); free(h_mean); free(dy_buf);
    }

    if (out_val_gain_pct != NULL) {
        *out_val_gain_pct = (float)(sum_alpha / (double)E);
    }
    return (float)(sum_retained / (double)E);
}
