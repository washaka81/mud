.section .text
.global ternary_gemv_4rows_avx2

# System V AMD64 ABI:
# rdi: n (length of x, multiple of 32)
# rsi: x (FP32 activations)
# rdx: weights (Pointer to 4 rows of weights)
# rcx: out (Pointer to 4 FP32 results)
# xmm0: scale (Global layer scale)
# r8:  stride (Number of u32 blocks per row = n / 16)

.section .rodata
.align 32
SHIFTS_LOW:  .long 0, 2, 4, 6, 8, 10, 12, 14
SHIFTS_HIGH: .long 16, 18, 20, 22, 24, 26, 28, 30
MASK_2BIT:   .long 3, 3, 3, 3, 3, 3, 3, 3
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_TWO:     .long 2, 2, 2, 2, 2, 2, 2, 2

.section .text
ternary_gemv_4rows_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    
    # ymm15: scale
    vbroadcastss %xmm0, %ymm15
    
    # Accumulators
    vxorps %ymm0, %ymm0, %ymm0 # Row 0
    vxorps %ymm1, %ymm1, %ymm1 # Row 1
    vxorps %ymm2, %ymm2, %ymm2 # Row 2
    vxorps %ymm3, %ymm3, %ymm3 # Row 3
    
    vmovdqa SHIFTS_LOW(%rip), %ymm10
    vmovdqa SHIFTS_HIGH(%rip), %ymm11
    vmovdqa MASK_2BIT(%rip), %ymm12
    vmovdqa VAL_ONE(%rip), %ymm13
    vmovdqa VAL_TWO(%rip), %ymm14

    # Pointers to row starts
    mov %r8, %rax
    shl $2, %rax           # rax = stride * 4 (bytes per row)
    lea (%rdx, %rax), %r12
    lea (%r12, %rax), %r13
    lea (%r13, %rax), %r14

    mov %rdi, %r9          # Loop counter (n)

.loop:
    # Prefetch next iterations
    prefetcht0 256(%rsi)        # Prefetch activations
    prefetcht0 256(%rdx)        # Prefetch weights row 0

    # Load 16 activations (64 bytes)
    vmovups (%rsi), %ymm4       # x[0..7]
    vmovups 32(%rsi), %ymm5     # x[8..15]

    # Row 0
    vpbroadcastd (%rdx), %ymm6
    # Unpack 0-7
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm4, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm7, %ymm4, %ymm7
    vsubps %ymm7, %ymm0, %ymm0
    # Unpack 8-15
    vpsrlvd %ymm11, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm0, %ymm0

    # Row 1
    vpbroadcastd (%r12), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm4, %ymm8
    vaddps %ymm8, %ymm1, %ymm1
    vpand %ymm7, %ymm4, %ymm7
    vsubps %ymm7, %ymm1, %ymm1
    vpsrlvd %ymm11, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm1, %ymm1
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm1, %ymm1

    # Row 2
    vpbroadcastd (%r13), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm4, %ymm8
    vaddps %ymm8, %ymm2, %ymm2
    vpand %ymm7, %ymm4, %ymm7
    vsubps %ymm7, %ymm2, %ymm2
    vpsrlvd %ymm11, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm2, %ymm2
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm2, %ymm2

    # Row 3
    vpbroadcastd (%r14), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm4, %ymm8
    vaddps %ymm8, %ymm3, %ymm3
    vpand %ymm7, %ymm4, %ymm7
    vsubps %ymm7, %ymm3, %ymm3
    vpsrlvd %ymm11, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm3, %ymm3
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm3, %ymm3

    add $4, %rdx
    add $4, %r12
    add $4, %r13
    add $4, %r14
    add $64, %rsi
    sub $16, %r9
    jnz .loop

.done_accum:
    # Scale (ymm15 still holds scale)
    vmulps %ymm15, %ymm0, %ymm0
    vmulps %ymm15, %ymm1, %ymm1
    vmulps %ymm15, %ymm2, %ymm2
    vmulps %ymm15, %ymm3, %ymm3

    # Reduction and output
    # Row 0
    vextractf128 $1, %ymm0, %xmm4
    vaddps %xmm4, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm4
    vaddps %xmm4, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm4
    vaddps %xmm4, %xmm0, %xmm0
    vmovss %xmm0, (%rcx)

    # Row 1
    vextractf128 $1, %ymm1, %xmm4
    vaddps %xmm4, %xmm1, %xmm1
    vshufps $0xEE, %xmm1, %xmm1, %xmm4
    vaddps %xmm4, %xmm1, %xmm1
    vshufps $0x11, %xmm1, %xmm1, %xmm4
    vaddps %xmm4, %xmm1, %xmm1
    vmovss %xmm1, 4(%rcx)

    # Row 2
    vextractf128 $1, %ymm2, %xmm4
    vaddps %xmm4, %xmm2, %xmm2
    vshufps $0xEE, %xmm2, %xmm2, %xmm4
    vaddps %xmm4, %xmm2, %xmm2
    vshufps $0x11, %xmm2, %xmm2, %xmm4
    vaddps %xmm4, %xmm2, %xmm2
    vmovss %xmm2, 8(%rcx)

    # Row 3
    vextractf128 $1, %ymm3, %xmm4
    vaddps %xmm4, %xmm3, %xmm3
    vshufps $0xEE, %xmm3, %xmm3, %xmm4
    vaddps %xmm4, %xmm3, %xmm3
    vshufps $0x11, %xmm3, %xmm3, %xmm4
    vaddps %xmm4, %xmm3, %xmm3
    vmovss %xmm3, 12(%rcx)

    vzeroupper
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret
