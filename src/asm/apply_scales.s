/* apply_row_scales_8_avx2 — Batch apply 8 per-row PRQ scales (AVX2)
 *
 * void apply_row_scales_8_avx2(float* out, const float* scales);
 *
 * For each of 8 rows:
 *   s = scales[i]
 *   if s is NaN/Inf or |s| > 1e8: out[i] = 0.0
 *   else: out[i] *= s
 *
 * Uses packed f32 (8 values per YMM register) to leverage every computation.
 * Replaces the scalar loop in ternary_gemv_rowwise_submit.
 *
 * Args:
 *   rdi = out (8 × f32)
 *   rsi = scales (8 × f32)
 *
 * Registers:
 *   ymm0 = out values (loaded)
 *   ymm1 = scales (loaded)
 *   ymm2 = abs(scales) for NaN/Inf check
 *   ymm3 = comparison mask
 *   ymm4 = clamped scales
 */

.section .text
.global apply_row_scales_8_avx2
.type apply_row_scales_8_avx2, @function

apply_row_scales_8_avx2:
    push %rbp
    mov %rsp, %rbp

    # Load 8 out values and 8 scales
    vmovups (%rdi), %ymm0      # out[0..7]
    vmovups (%rsi), %ymm1      # scales[0..7]

    # NaN/Inf detection: abs(scale) has exponent 0xFF
    # exp_mask = 0x7F800000
    vmovdqa .exp_mask(%rip), %ymm2
    vpand %ymm2, %ymm1, %ymm3     # extract exponent bits
    vpcmpeqd %ymm2, %ymm3, %ymm3  # 0xFFFFFFFF where NaN/Inf

    # Clamp: min(max(scale, -1e8), 1e8)
    vbroadcastss .clamp_min(%rip), %ymm4   # -1e8
    vbroadcastss .clamp_max(%rip), %ymm5   # 1e8
    vmaxps %ymm4, %ymm1, %ymm4             # max(scale, -1e8)
    vminps %ymm5, %ymm4, %ymm4             # min(max(scale, -1e8), 1e8)

    # Replace NaN/Inf scales with 0 (blend: where NaN/Inf, use 0)
    vpxor %ymm6, %ymm6, %ymm6               # 0.0
    vblendvps %ymm3, %ymm6, %ymm4, %ymm4   # if NaN/Inf: 0, else: clamped

    # Multiply out *= clamped_scale
    vmulps %ymm4, %ymm0, %ymm0

    # Store result
    vmovups %ymm0, (%rdi)

    vzeroupper
    pop %rbp
    ret

.section .rodata
.align 32
.exp_mask:
    .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
    .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
.clamp_min:
    .float -100000000.0
.clamp_max:
    .float 100000000.0
