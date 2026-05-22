.section .text
.global ternary_gemv_avx2

# System V AMD64 ABI:
# rdi: n (total weights, multiple of 32 for unrolled loop)
# rsi: x (FP32 activations)
# rdx: weights (Ternary 2-bit packed, 16 per u32)
# rcx: out (Pointer to FP32 result)
# xmm0: scale (Global layer scale)

.section .rodata
.align 32
SHIFTS_LOW:  .long 0, 2, 4, 6, 8, 10, 12, 14
SHIFTS_HIGH: .long 16, 18, 20, 22, 24, 26, 28, 30
MASK_2BIT:   .long 3, 3, 3, 3, 3, 3, 3, 3
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_TWO:     .long 2, 2, 2, 2, 2, 2, 2, 2

.section .text
ternary_gemv_avx2:
    push %rbp
    mov %rsp, %rbp
    
    # ymm8: scale
    vbroadcastss %xmm0, %ymm8
    
    # ymm0, ymm9: Accumulators (FP32) - use two for better ILP
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm9, %ymm9, %ymm9
    
    vmovdqa SHIFTS_LOW(%rip), %ymm10
    vmovdqa SHIFTS_HIGH(%rip), %ymm11
    vmovdqa MASK_2BIT(%rip), %ymm12
    vmovdqa VAL_ONE(%rip), %ymm13
    vmovdqa VAL_TWO(%rip), %ymm14

.loop:
    cmp $32, %rdi
    jl .leftover
    
    # --- BLOCK 1 (16 weights) ---
    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    vmovups (%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- BLOCK 2 (Next 16 weights) ---
    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    vmovups 64(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    vmovups 96(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    add $8, %rdx
    add $128, %rsi
    sub $32, %rdi
    jmp .loop

.leftover:
    test %rdi, %rdi
    jle .done_accum
    
    # Process remaining 16 weights if n >= 16
    cmp $16, %rdi
    jl .done_accum # Should not happen if n is multiple of 16

    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    vmovups (%rsi), %ymm5
    vaddps %ymm0, %ymm9, %ymm0 # Merge accumulators early
    vxorps %ymm9, %ymm9, %ymm9
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    sub $16, %rdi

.done_accum:
    vaddps %ymm9, %ymm0, %ymm0
    vmulps %ymm8, %ymm0, %ymm0

    # Horizontal reduction
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    
    vaddss (%rcx), %xmm0, %xmm0
    vmovss %xmm0, (%rcx)
    
    vzeroupper
    pop %rbp
    ret
