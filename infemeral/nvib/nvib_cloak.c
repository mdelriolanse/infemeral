#include "nvib_cloak.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* SIMD intrinsics - include based on available instruction sets */
#ifdef __AVX512F__
#include <immintrin.h>
#define USE_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define USE_SSE4
#endif

/* Xorshift128+ PRNG */
static inline uint64_t xorshift128_next(uint64_t state[2]) {
    uint64_t s1 = state[0];
    uint64_t s0 = state[1];
    state[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0;
    s1 ^= s0 >> 26;
    state[1] = s1;
    return state[0] + state[1];
}

/* Generate uniform random float in [0, 1) */
static inline float xorshift128_float(uint64_t state[2]) {
    uint64_t u = xorshift128_next(state);
    /* Convert to float in [0, 1) */
    return (float)(u >> 11) * (1.0f / (1ULL << 53));
}

/* Box-Muller transform: Generate standard normal random variable */
static inline float xorshift128_normal(uint64_t state[2]) {
    static float z0 = 0.0f;
    static int has_spare = 0;

    if (has_spare) {
        has_spare = 0;
        return z0;
    }

    has_spare = 1;
    float u1 = xorshift128_float(state);
    float u2 = xorshift128_float(state);

    float mag = sqrtf(-2.0f * logf(u1 + 1e-10f));
    z0 = mag * cosf(2.0f * M_PI * u2);
    return mag * sinf(2.0f * M_PI * u2);
}

/* Initialize NVIB context */
nvib_context_t* nvib_cloak_init(
    int dim,
    float beta,
    float mu_init,
    float log_sigma2_init,
    uint64_t seed
) {
    if (dim <= 0) {
        return NULL;
    }

    nvib_context_t *ctx = (nvib_context_t*)malloc(sizeof(nvib_context_t));
    if (!ctx) {
        return NULL;
    }

    ctx->dim = dim;
    ctx->beta = beta;

    /* Allocate and initialize mean vector */
    ctx->mu = (float*)malloc(dim * sizeof(float));
    if (!ctx->mu) {
        free(ctx);
        return NULL;
    }
    for (int i = 0; i < dim; i++) {
        ctx->mu[i] = mu_init;
    }

    /* Allocate and initialize log variance vector */
    ctx->log_sigma2 = (float*)malloc(dim * sizeof(float));
    if (!ctx->log_sigma2) {
        free(ctx->mu);
        free(ctx);
        return NULL;
    }
    for (int i = 0; i < dim; i++) {
        ctx->log_sigma2[i] = log_sigma2_init;
    }

    /* Initialize PRNG state */
    if (seed == 0) {
        /* Use system entropy */
        ctx->prng_state[0] = (uint64_t)time(NULL);
        ctx->prng_state[1] = (uint64_t)clock();
    } else {
        ctx->prng_state[0] = seed;
        ctx->prng_state[1] = seed ^ 0x9e3779b97f4a7c15ULL;
    }

    return ctx;
}

/* Set PRNG seed */
void nvib_cloak_set_seed(nvib_context_t *ctx, uint64_t seed) {
    if (!ctx) return;
    ctx->prng_state[0] = seed;
    ctx->prng_state[1] = seed ^ 0x9e3779b97f4a7c15ULL;
}

/* Set privacy budget beta */
void nvib_cloak_set_beta(nvib_context_t *ctx, float beta) {
    if (!ctx) return;
    ctx->beta = beta;
}

/* Scalar implementation of NVIB forward pass */
static int nvib_cloak_forward_scalar(
    nvib_context_t *ctx,
    const float *input,
    float *output
) {
    for (int i = 0; i < ctx->dim; i++) {
        /* Sample epsilon ~ N(0, 1) */
        float epsilon = xorshift128_normal(ctx->prng_state);

        /* Compute sigma = exp(0.5 * log_sigma2) */
        float sigma = expf(0.5f * ctx->log_sigma2[i]);

        /* Apply reparameterization: z = input + (1/beta) * sigma * epsilon */
        /* Higher beta = less noise = more privacy budget = better utility */
        /* Lower beta = more noise = less privacy budget = worse utility */
        float noise_scale = sigma / (ctx->beta + 1e-10f);  /* Avoid division by zero */
        output[i] = input[i] + noise_scale * epsilon;
    }

    return 0;
}

#ifdef USE_AVX512
/* AVX-512 optimized implementation (16 floats per iteration) */
static int nvib_cloak_forward_avx512(
    nvib_context_t *ctx,
    const float *input,
    float *output
) {
    const int simd_width = 16;
    int i;

    for (i = 0; i <= ctx->dim - simd_width; i += simd_width) {
        /* Load input */
        __m512 x = _mm512_loadu_ps(&input[i]);

        /* Generate epsilon ~ N(0,1) for each element */
        __m512 epsilon;
        float eps_array[16];
        for (int j = 0; j < 16; j++) {
            eps_array[j] = xorshift128_normal(ctx->prng_state);
        }
        epsilon = _mm512_loadu_ps(eps_array);

        /* Compute sigma = exp(0.5 * log_sigma2) */
        /* Use scalar exp (SVML _mm512_exp_ps may not be available) */
        float sigma_array[16];
        for (int j = 0; j < 16; j++) {
            sigma_array[j] = expf(0.5f * ctx->log_sigma2[i + j]);
        }
        __m512 sigma = _mm512_loadu_ps(sigma_array);

        /* Compute: z = input + (sigma / beta) * epsilon */
        /* Higher beta = less noise = higher similarity */
        __m512 beta_vec = _mm512_set1_ps(ctx->beta + 1e-10f);  /* Avoid division by zero */
        __m512 noise_scale = _mm512_div_ps(sigma, beta_vec);
        __m512 noise = _mm512_mul_ps(noise_scale, epsilon);
        __m512 z = _mm512_add_ps(x, noise);

        /* Store output */
        _mm512_storeu_ps(&output[i], z);
    }

    /* Handle remaining elements with scalar code */
    return nvib_cloak_forward_scalar(ctx, &input[i], &output[i]);
}
#endif

#ifdef USE_AVX2
/* AVX2 optimized implementation (8 floats per iteration) */
static int nvib_cloak_forward_avx2(
    nvib_context_t *ctx,
    const float *input,
    float *output
) {
    const int simd_width = 8;
    int i;

    for (i = 0; i <= ctx->dim - simd_width; i += simd_width) {
        /* Load input */
        __m256 x = _mm256_loadu_ps(&input[i]);

        /* Load log_sigma2 */
        __m256 log_sigma2 = _mm256_loadu_ps(&ctx->log_sigma2[i]);

        /* Generate epsilon ~ N(0,1) */
        float eps_array[8];
        for (int j = 0; j < 8; j++) {
            eps_array[j] = xorshift128_normal(ctx->prng_state);
        }
        __m256 epsilon = _mm256_loadu_ps(eps_array);

        /* Compute sigma = exp(0.5 * log_sigma2) - scalar for now */
        float sigma_array[8];
        for (int j = 0; j < 8; j++) {
            sigma_array[j] = expf(0.5f * ctx->log_sigma2[i + j]);
        }
        __m256 sigma = _mm256_loadu_ps(sigma_array);

        /* Compute: z = input + (sigma / beta) * epsilon */
        /* Higher beta = less noise = higher similarity */
        __m256 beta_vec = _mm256_set1_ps(ctx->beta + 1e-10f);  /* Avoid division by zero */
        __m256 noise_scale = _mm256_div_ps(sigma, beta_vec);
        __m256 noise = _mm256_mul_ps(noise_scale, epsilon);
        __m256 z = _mm256_add_ps(x, noise);

        /* Store output */
        _mm256_storeu_ps(&output[i], z);
    }

    /* Handle remaining elements with scalar code */
    if (i < ctx->dim) {
        nvib_context_t temp_ctx = *ctx;
        temp_ctx.dim = ctx->dim - i;
        return nvib_cloak_forward_scalar(&temp_ctx, &input[i], &output[i]);
    }
    return 0;
}
#endif

/* Main forward function with SIMD dispatch */
int nvib_cloak_forward(
    nvib_context_t *ctx,
    const float *input,
    float *output
) {
    if (!ctx || !input || !output) {
        return -1;
    }

#ifdef USE_AVX512
    return nvib_cloak_forward_avx512(ctx, input, output);
#elif defined(USE_AVX2)
    return nvib_cloak_forward_avx2(ctx, input, output);
#else
    return nvib_cloak_forward_scalar(ctx, input, output);
#endif
}

/* Free NVIB context */
void nvib_cloak_free(nvib_context_t *ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->mu) {
        free(ctx->mu);
    }
    if (ctx->log_sigma2) {
        free(ctx->log_sigma2);
    }

    free(ctx);
}
