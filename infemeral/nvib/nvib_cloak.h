#ifndef NVIB_CLOAK_H
#define NVIB_CLOAK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NVIB Context Structure */
typedef struct {
    float *mu;              /* Mean vector (4096 elements) */
    float *log_sigma2;      /* Log variance vector (4096 elements) */
    float beta;             /* Privacy budget parameter */
    uint64_t prng_state[2]; /* Xorshift128+ state */
    int dim;                /* Embedding dimension (typically 4096) */
} nvib_context_t;

/* Initialize NVIB context
 *
 * Args:
 *   dim: Embedding dimension (typically 4096)
 *   beta: Privacy budget parameter (default: 1.0)
 *   mu_init: Initial mean value (default: 0.0)
 *   log_sigma2_init: Initial log variance (default: 0.0)
 *   seed: PRNG seed (0 = use system entropy)
 *
 * Returns:
 *   Pointer to initialized context, or NULL on error
 */
nvib_context_t* nvib_cloak_init(
    int dim,
    float beta,
    float mu_init,
    float log_sigma2_init,
    uint64_t seed
);

/* Apply NVIB cloaking transformation
 *
 * Implements: z = μ + exp(0.5 * log_σ²) ⊙ ε
 * where ε ~ N(0, I) is sampled using Xorshift128+ PRNG
 *
 * Args:
 *   ctx: NVIB context (must be initialized)
 *   input: Input embedding vector (dim elements, float32)
 *   output: Output noised vector (dim elements, float32, must be pre-allocated)
 *
 * Returns:
 *   0 on success, -1 on error
 */
int nvib_cloak_forward(
    nvib_context_t *ctx,
    const float *input,
    float *output
);

/* Set PRNG seed (for deterministic behavior)
 *
 * Args:
 *   ctx: NVIB context
 *   seed: PRNG seed value
 */
void nvib_cloak_set_seed(nvib_context_t *ctx, uint64_t seed);

/* Set privacy budget beta
 *
 * Args:
 *   ctx: NVIB context
 *   beta: Privacy budget parameter (higher = less noise)
 */
void nvib_cloak_set_beta(nvib_context_t *ctx, float beta);

/* Free NVIB context
 *
 * Args:
 *   ctx: Context to free (can be NULL)
 */
void nvib_cloak_free(nvib_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* NVIB_CLOAK_H */
