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
    
    # ymm0, ymm9, ymm13, ymm14: Accumulators (FP32) - four for better ILP
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm9, %ymm9, %ymm9
    vxorps %ymm13, %ymm13, %ymm13
    vxorps %ymm14, %ymm14, %ymm14
    
    vmovdqa SHIFTS_LOW(%rip), %ymm10
    vmovdqa SHIFTS_HIGH(%rip), %ymm11
    vmovdqa MASK_2BIT(%rip), %ymm12

.loop:
    cmp $64, %rdi
    jl .leftover

    # L1/L2 cache prefetch: 256 bytes ahead for activations, 64 bytes for packed weights
    prefetcht0 256(%rsi)
    prefetchnta 64(%rdx)

    # --- BLOCK 1 (16 weights → acc ymm0) ---
    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups (%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- BLOCK 2 (16 weights → acc ymm9) ---
    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 64(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 96(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    # --- BLOCK 3 (16 weights → acc ymm13) ---
    vpbroadcastd 8(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 128(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 160(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13

    # --- BLOCK 4 (16 weights → acc ymm14) ---
    vpbroadcastd 12(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 192(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 224(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14

    add $16, %rdx
    add $256, %rsi
    sub $64, %rdi
    jmp .loop

.leftover:
    test %rdi, %rdi
    jle .done_accum

    cmp $32, %rdi
    jl .leftover16

    # Process 32 weights with 2 accumulators
    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups (%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 64(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 96(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    add $8, %rdx
    add $128, %rsi
    sub $32, %rdi

.leftover16:
    cmp $16, %rdi
    jl .done_accum

    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups (%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    sub $16, %rdi

.done_accum:
    vaddps %ymm9, %ymm0, %ymm0
    vaddps %ymm13, %ymm0, %ymm0
    vaddps %ymm14, %ymm0, %ymm0
    vmulps %ymm8, %ymm0, %ymm0

    # Horizontal reduction
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    
    
    vmovss %xmm0, (%rcx)
    
    vzeroupper
    pop %rbp
    ret

.global ternary_gemv_i8act_avx2
ternary_gemv_i8act_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    sub $8, %rsp                 # align to 16 bytes

    # ymm8: scale
    vbroadcastss %xmm0, %ymm8
    
    # ymm0, ymm9, ymm13, ymm14: Accumulators (FP32) - four for better ILP
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm9, %ymm9, %ymm9
    vxorps %ymm13, %ymm13, %ymm13
    vxorps %ymm14, %ymm14, %ymm14
    vxorps %xmm3, %xmm3, %xmm3       # scalar tail accumulator
    
    vmovdqa SHIFTS_LOW(%rip), %ymm10
    vmovdqa SHIFTS_HIGH(%rip), %ymm11
    vmovdqa MASK_2BIT(%rip), %ymm12

.loop_i8:
    cmp $64, %rdi
    jl .leftover_i8

    # L1/L2 cache prefetch: 256 bytes ahead for activations, 64 bytes for packed weights
    prefetcht0 256(%rsi)
    prefetchnta 64(%rdx)

    # --- BLOCK 1 (16 weights → acc ymm0) ---
    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd (%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 8(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- BLOCK 2 (16 weights → acc ymm9) ---
    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 16(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 24(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    # --- BLOCK 3 (16 weights → acc ymm13) ---
    vpbroadcastd 8(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 32(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 40(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13

    # --- BLOCK 4 (16 weights → acc ymm14) ---
    vpbroadcastd 12(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 48(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 56(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14

    add $16, %rdx
    add $64, %rsi
    sub $64, %rdi
    jmp .loop_i8

.leftover_i8:
    test %rdi, %rdi
    jle .done_accum_i8

    cmp $32, %rdi
    jl .leftover16_i8

    # Process 32 weights with 2 accumulators
    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd (%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 8(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 16(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 24(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    add $8, %rdx
    add $32, %rsi
    sub $32, %rdi
    jmp .leftover_i8

.leftover16_i8:
    cmp $16, %rdi
    jl .scalar_tail_i8

    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd (%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3
    vpcmpeqd VAL_TWO(%rip), %ymm2, %ymm4
    vpmovsxbd 8(%rsi), %ymm5
    vcvtdq2ps %ymm5, %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    add $4, %rdx
    add $16, %rsi
    sub $16, %rdi
    jmp .leftover_i8

.scalar_tail_i8:
    vxorps %xmm15, %xmm15, %xmm15
    test %rdi, %rdi
    jle .done_accum_i8

    mov (%rdx), %eax                  # eax = current u32
    mov %rcx, -48(%rbp)               # save output ptr
    xor %ecx, %ecx                    # cl = bit offset
    xor %r8d, %r8d                    # r8 = element index

.tail_loop_i8:
    mov %eax, %r9d
    shr %cl, %r9d                     # shift u32 right by bit offset
    and $3, %r9d
    cmp $1, %r9d
    je .tail_plus1_i8
    cmp $2, %r9d
    je .tail_minus1_i8
    jmp .tail_next_i8

.tail_plus1_i8:
    movsbl (%rsi,%r8), %r10d
    cvtsi2ss %r10d, %xmm4
    vaddss %xmm4, %xmm15, %xmm15
    jmp .tail_next_i8

.tail_minus1_i8:
    movsbl (%rsi,%r8), %r10d
    cvtsi2ss %r10d, %xmm4
    vsubss %xmm4, %xmm15, %xmm15

.tail_next_i8:
    add $2, %ecx
    inc %r8d
    cmp %r8d, %edi
    jne .tail_loop_i8

.tail_done_i8:
    mov -48(%rbp), %rcx               # restore output ptr
    # NOTE: do NOT vaddss here — it would zero upper 128 bits of ymm0 (AVX VEX semantics)
    # Scaled tail is added after horizontal reduction in .done_accum_i8

.done_accum_i8:
    vaddps %ymm9, %ymm0, %ymm0
    vaddps %ymm13, %ymm0, %ymm0
    vaddps %ymm14, %ymm0, %ymm0
    vmulps %ymm8, %ymm0, %ymm0

    # Horizontal reduction
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    
    # Add scaled scalar tail (preserved in xmm15[0])
    vmulss %xmm8, %xmm15, %xmm15  # xmm15[0] *= combined_scale
    vaddss %xmm15, %xmm0, %xmm0   # add to final scalar

    
    vmovss %xmm0, (%rcx)

    vzeroupper
    add $8, %rsp
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    pop %rbp
    ret

