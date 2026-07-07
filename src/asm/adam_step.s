/* adam_step_avx2 — Adam optimizer step kernel (AVX2 + FMA)
 * ─────────────────────────────────────────────────────────────────────────
 * void adam_step_avx2(
 *   size_t  n,          rdi   number of elements
 *   float*  w,          rsi   master weights (in/out)
 *   float*  m,          rdx   first moment   (in/out)
 *   float*  v,          rcx   second moment  (in/out)
 *   float*  grads,      r8    raw gradients  (read-only)
 *   float   clip_coef,  xmm0  gradient clip coefficient
 *   float   wd,         xmm1  weight decay
 *   float   b1,         xmm2  beta1
 *   float   b2,         xmm3  beta2
 *   float   lr_bc1,     xmm4  lr / (1 - beta1^t)  pre-computed
 *   float   inv_bc2,    xmm5  1 / (1 - beta2^t)   pre-computed
 *   float   eps         xmm6  epsilon floor
 * );
 * ─────────────────────────────────────────────────────────────────────────
 * Register allocation (YMM):
 *   ymm8  = clip_coef (broadcast)
 *   ymm9  = wd        (broadcast)
 *   ymm10 = b1        (broadcast)
 *   ymm11 = 1 - b1    (broadcast)
 *   ymm12 = lr_bc1    (broadcast)
 *   ymm13 = b2        (broadcast)
 *   ymm14 = 1 - b2    (broadcast)
 *   ymm15 = eps       (broadcast)
 *   ymm7  = inv_bc2   (broadcast)
 */

.intel_syntax noprefix
.text

.global adam_step_avx2
.type   adam_step_avx2, @function

adam_step_avx2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12

    /* Broadcast all scalar arguments from xmm into ymm */
    vbroadcastss    ymm8,  xmm0    /* clip_coef */
    vbroadcastss    ymm9,  xmm1    /* wd */
    vbroadcastss    ymm10, xmm2    /* b1 */
    vbroadcastss    ymm13, xmm3    /* b2 */
    vbroadcastss    ymm12, xmm4    /* lr_bc1 */
    vbroadcastss    ymm7,  xmm5    /* inv_bc2 */
    vbroadcastss    ymm15, xmm6    /* eps */

    /* Compute (1 - b1) via: tmp = 1.0 - b1 */
    mov     eax, 0x3F800000
    vmovd   xmm0, eax              /* xmm0 = 1.0f */
    vbroadcastss ymm0, xmm0        /* ymm0 = {1.0, 1.0, ..., 1.0} */
    vsubps  ymm11, ymm0, ymm10     /* ymm11 = 1 - b1 */
    vsubps  ymm14, ymm0, ymm13     /* ymm14 = 1 - b2 */

    xor     r12d, r12d             /* i = 0 */

.Ladam_loop8:
    /* Check: have we consumed at least 8 more elements? */
    lea     rbx, [r12 + 8]
    cmp     rbx, rdi
    ja      .Ladam_tail            /* less than 8 remain */

    /* Load 8 elements from each buffer */
    vmovups ymm0, [r8  + r12*4]   /* grads[i..i+8] */
    vmovups ymm1, [rsi + r12*4]   /* w[i..i+8] */
    vmovups ymm2, [rdx + r12*4]   /* m[i..i+8] */
    vmovups ymm3, [rcx + r12*4]   /* v[i..i+8] */

    /* g = grads * clip_coef + wd * w */
    vmulps  ymm4, ymm0, ymm8           /* ymm4 = grads * clip_coef */
    vfmadd231ps ymm4, ymm1, ymm9       /* ymm4 += wd * w */

    /* m = b1 * m + (1 - b1) * g */
    vmulps  ymm2, ymm2, ymm10          /* m = b1 * m */
    vfmadd231ps ymm2, ymm4, ymm11      /* m += (1-b1)*g */

    /* v = b2 * v + (1 - b2) * g*g */
    vmulps  ymm5, ymm4, ymm4           /* ymm5 = g*g */
    vmulps  ymm3, ymm3, ymm13          /* v = b2 * v */
    vfmadd231ps ymm3, ymm5, ymm14      /* v += (1-b2)*g*g */

    /* v_hat = v * inv_bc2 */
    vmulps  ymm5, ymm3, ymm7           /* ymm5 = v_hat */
    vsqrtps ymm5, ymm5                 /* ymm5 = sqrt(v_hat) */
    vaddps  ymm5, ymm5, ymm15          /* ymm5 += eps */

    /* step = lr_bc1 * m / (sqrt(v_hat) + eps) */
    vdivps  ymm6, ymm2, ymm5           /* ymm6 = m / denom */
    vmulps  ymm6, ymm6, ymm12          /* ymm6 = lr_bc1 * m / denom */

    /* w -= step */
    vsubps  ymm1, ymm1, ymm6

    /* Store updated w, m, v */
    vmovups [rsi + r12*4], ymm1
    vmovups [rdx + r12*4], ymm2
    vmovups [rcx + r12*4], ymm3

    add     r12, 8
    jmp     .Ladam_loop8

.Ladam_tail:
    /* Scalar tail for remaining elements (0..7) */
    /* Re-extract scalar constants from ymm into xmm */
    vextractf128    xmm8,  ymm8,  0
    vextractf128    xmm9,  ymm9,  0
    vextractf128    xmm10, ymm10, 0
    vextractf128    xmm11, ymm11, 0
    vextractf128    xmm12, ymm12, 0
    vextractf128    xmm13, ymm13, 0
    vextractf128    xmm14, ymm14, 0
    vextractf128    xmm7,  ymm7,  0
    vextractf128    xmm15, ymm15, 0

.Ladam_tail_loop:
    cmp     r12, rdi
    jge     .Ladam_done

    vmovss  xmm0, [r8  + r12*4]        /* g = grads[i] */
    vmovss  xmm1, [rsi + r12*4]        /* w[i] */
    vmovss  xmm2, [rdx + r12*4]        /* m[i] */
    vmovss  xmm3, [rcx + r12*4]        /* v[i] */

    /* g = grads * clip_coef + wd * w */
    vmulss  xmm4, xmm0, xmm8
    vfmadd231ss xmm4, xmm1, xmm9

    /* m = b1*m + (1-b1)*g */
    vmulss  xmm2, xmm2, xmm10
    vfmadd231ss xmm2, xmm4, xmm11

    /* v = b2*v + (1-b2)*g*g */
    vmulss  xmm5, xmm4, xmm4
    vmulss  xmm3, xmm3, xmm13
    vfmadd231ss xmm3, xmm5, xmm14

    /* w -= lr_bc1 * m / (sqrt(v*inv_bc2) + eps) */
    vmulss  xmm5, xmm3, xmm7
    vsqrtss xmm5, xmm5, xmm5
    vaddss  xmm5, xmm5, xmm15
    vdivss  xmm6, xmm2, xmm5
    vmulss  xmm6, xmm6, xmm12
    vsubss  xmm1, xmm1, xmm6

    vmovss  [rsi + r12*4], xmm1
    vmovss  [rdx + r12*4], xmm2
    vmovss  [rcx + r12*4], xmm3

    inc     r12
    jmp     .Ladam_tail_loop

.Ladam_done:
    vzeroupper
    pop     r12
    pop     rbx
    pop     rbp
    ret

.size adam_step_avx2, .-adam_step_avx2
