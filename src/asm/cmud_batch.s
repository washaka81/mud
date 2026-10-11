/* cmud_batch.s — AVX2 vectorized CMUD complex multiply (8 elements at once)
 *
 * void cmud_gauss_mul_batch_avx2(
 *     const float* x_re, const float* x_im,
 *     const float* w_re, const float* w_im,
 *     float* y_re, float* y_im
 * );
 *
 * For each of 8 elements:
 *   Y_R = X_R * W_R - X_I * W_I
 *   Y_I = X_R * W_I + X_I * W_R
 *
 * Uses 4 vmulps + 1 vaddps + 1 vsubps per batch (6 instructions for 8 elements).
 *
 * Args (System V AMD64):
 *   rdi=x_re, rsi=x_im, rdx=w_re, rcx=w_im, r8=y_re, r9=y_im
 *
 * AT&T syntax: vOP src1, src2, dst → dst = src2 OP src1
 */

.section .text
.global cmud_gauss_mul_batch_avx2
.type cmud_gauss_mul_batch_avx2, @function

cmud_gauss_mul_batch_avx2:
    push %rbp
    mov %rsp, %rbp

    # Load 8 complex X values
    vmovups (%rdi), %ymm0      # x_re[0..7]
    vmovups (%rsi), %ymm1      # x_im[0..7]

    # Load 8 complex W values
    vmovups (%rdx), %ymm2      # w_re[0..7]
    vmovups (%rcx), %ymm3      # w_im[0..7]

    # Y_R = X_R * W_R - X_I * W_I
    vmulps %ymm2, %ymm0, %ymm4       # ymm4 = x_re * w_re
    vmulps %ymm3, %ymm1, %ymm5       # ymm5 = x_im * w_im
    vsubps %ymm5, %ymm4, %ymm4       # ymm4 = x_re*w_re - x_im*w_im

    # Y_I = X_R * W_I + X_I * W_R
    vmulps %ymm3, %ymm0, %ymm6       # ymm6 = x_re * w_im
    vmulps %ymm2, %ymm1, %ymm7       # ymm7 = x_im * w_re
    vaddps %ymm7, %ymm6, %ymm6       # ymm6 = x_re*w_im + x_im*w_re

    # Store results
    vmovups %ymm4, (%r8)             # y_re[0..7]
    vmovups %ymm6, (%r9)             # y_im[0..7]

    vzeroupper
    pop %rbp
    ret
