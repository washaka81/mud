/* qat_step.s — AVX2 kernel for Deep QAT Thawing (Ternary2Bit)
 * ─────────────────────────────────────────────────────────────────────────
 * void qat_step_ternary2bit_avx2(
 *   size_t  cols,       // rdi (must be multiple of 16)
 *   size_t  rows,       // rsi
 *   float*  w_fp32,     // rdx (shadow weights, in/out)
 *   const float* grad,  // rcx (gradients, read-only)
 *   u8*     w_packed,   // r8  (output ternary2bit packed)
 *   float*  scales,     // r9  (output PRQ scales)
 *   float   lr,         // xmm0
 *   float   jitter      // xmm1
 * );
 * ─────────────────────────────────────────────────────────────────────────
 */

.intel_syntax noprefix
.text

.section .rodata
.align 32
.Labs_mask:
    .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
    .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
.Lmin_clamp:
    .float -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
.Lmax_clamp:
    .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
.Lone_d:
    .long 1, 1, 1, 1, 1, 1, 1, 1
.Lneg1_d:
    .long -1, -1, -1, -1, -1, -1, -1, -1
.Ltwo_d:
    .long 2, 2, 2, 2, 2, 2, 2, 2
.Lfifteen_d:
    .long 15, 15, 15, 15, 15, 15, 15, 15
.Lshift0:
    .long 0, 4, 8, 12, 16, 20, 24, 28
.Lmagic_0707:
    .float 0.70710678
.Lmagic_1e8:
    .float 1e-8
.Lshadow_max:
    .float 512.0
.Lshadow_min:
    .float -512.0

.text
.global qat_step_ternary2bit_avx2
.type   qat_step_ternary2bit_avx2, @function

qat_step_ternary2bit_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    /* Convert cols to floats for absmean division: 1.0 / cols */
    vcvtsi2ss xmm2, xmm2, rdi
    mov eax, 0x3F800000 /* 1.0f */
    vmovd xmm3, eax
    vdivss xmm2, xmm3, xmm2 /* xmm2 = 1.0 / cols */

    vbroadcastss ymm14, xmm0 /* ymm14 = lr */
    vbroadcastss ymm15, xmm1 /* ymm15 = jitter */
    
    vmovups ymm13, [rip + .Labs_mask]
    vmovups ymm12, [rip + .Lmin_clamp]
    vmovups ymm11, [rip + .Lmax_clamp]

    vbroadcastss ymm4, [rip + .Lshadow_min] /* ymm4 = -512.0 */
    vbroadcastss ymm5, [rip + .Lshadow_max] /* ymm5 = 512.0 */

    xor r10, r10 /* row_idx = 0 */

.Lrow_loop:
    cmp r10, rsi
    jae .Lend

    vxorps ymm10, ymm10, ymm10 /* ymm10 = abs_sum */
    xor r11, r11 /* col_idx = 0 */

.Lcol_loop_pass1:
    vmovups ymm0, [rdx + r11*4] /* w_fp32 */
    vmovups ymm1, [rcx + r11*4] /* grad */
    
    vfnmadd231ps ymm0, ymm1, ymm14
    vmaxps ymm0, ymm0, ymm4
    vminps ymm0, ymm0, ymm5
    vmovups [rdx + r11*4], ymm0 /* store updated w_fp32 clamped to [-512,512] */
    
    vandps ymm0, ymm0, ymm13
    vaddps ymm10, ymm10, ymm0
    
    add r11, 8
    cmp r11, rdi
    jb .Lcol_loop_pass1

    vextractf128 xmm0, ymm10, 1
    vaddps xmm0, xmm0, xmm10
    vshufps xmm1, xmm0, xmm0, 0xEE
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 0x55
    vaddps xmm0, xmm0, xmm1 

    vmulss xmm0, xmm0, xmm2

    vmovss xmm1, [rip + .Lmagic_0707]
    vmulss xmm0, xmm0, xmm1
    
    vmovss xmm1, [rip + .Lmagic_1e8]
    vmaxss xmm0, xmm0, xmm1

    vmovss [r9 + r10*4], xmm0

    mov eax, 0x3F800000
    vmovd xmm1, eax
    vdivss xmm0, xmm1, xmm0
    vbroadcastss ymm9, xmm0 

    xor r11, r11 
    
.Lcol_loop_pass2:
    vmovups ymm0, [rdx + r11*4]
    vmovups ymm1, [rdx + r11*4 + 32]
    
    vmulps ymm0, ymm0, ymm9
    vmulps ymm1, ymm1, ymm9
    
    vaddps ymm0, ymm0, ymm15
    vaddps ymm1, ymm1, ymm15
    
    vroundps ymm0, ymm0, 0x00
    vroundps ymm1, ymm1, 0x00
    
    vmaxps ymm0, ymm0, ymm12
    vminps ymm0, ymm0, ymm11
    
    vmaxps ymm1, ymm1, ymm12
    vminps ymm1, ymm1, ymm11
    
    vcvttps2dq ymm0, ymm0
    vcvttps2dq ymm1, ymm1
    
    vmovdqu ymm2, [rip + .Lone_d]
    vpcmpeqd ymm3, ymm0, ymm2
    vpand ymm3, ymm3, ymm2 
    
    vmovdqu ymm4, [rip + .Lneg1_d]
    vpcmpeqd ymm5, ymm0, ymm4
    vmovdqu ymm6, [rip + .Lfifteen_d]
    vpand ymm5, ymm5, ymm6 
    
    vpor ymm0, ymm3, ymm5 

    vpcmpeqd ymm3, ymm1, ymm2
    vpand ymm3, ymm3, ymm2
    
    vpcmpeqd ymm5, ymm1, ymm4
    vpand ymm5, ymm5, ymm6
    
    vpor ymm1, ymm3, ymm5
    
    vmovdqu ymm2, [rip + .Lshift0]
    vpsllvd ymm0, ymm0, ymm2
    vpsllvd ymm1, ymm1, ymm2
    
    vextracti128 xmm2, ymm0, 1
    vpor xmm0, xmm0, xmm2
    vpshufd xmm2, xmm0, 0xEE
    vpor xmm0, xmm0, xmm2
    vpshufd xmm2, xmm0, 0x55
    vpor xmm0, xmm0, xmm2
    
    vextracti128 xmm2, ymm1, 1
    vpor xmm1, xmm1, xmm2
    vpshufd xmm2, xmm1, 0xEE
    vpor xmm1, xmm1, xmm2
    vpshufd xmm2, xmm1, 0x55
    vpor xmm1, xmm1, xmm2
    
    mov rax, r10
    imul rax, rdi
    add rax, r11
    shr rax, 1 
    
    vmovd [r8 + rax], xmm0
    vmovd [r8 + rax + 4], xmm1

    add r11, 16
    cmp r11, rdi
    jb .Lcol_loop_pass2

    mov rax, rdi
    shl rax, 2
    add rdx, rax
    add rcx, rax

    inc r10
    jmp .Lrow_loop

.Lend:
    vzeroupper

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
