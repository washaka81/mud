.section .text
.global ternary_gemv_avx2

# System V AMD64 ABI:
# rdi: n (total weights, multiple of 32 for unrolled loop)
# rsi: x (FP32 activations)
# rdx: weights (Ternary ELUT 4-bit packed, 8 per u32)
# rcx: out (Pointer to FP32 result)
# xmm0: scale (Global layer scale)

.section .rodata
.align 32
SHIFTS_ELUT: .long 0, 4, 8, 12, 16, 20, 24, 28
MASK_ELUT:   .long 15, 15, 15, 15, 15, 15, 15, 15
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_MINUS_ONE: .long 15, 15, 15, 15, 15, 15, 15, 15

.section .text
ternary_gemv_avx2:
    push %rbp
    mov %rsp, %rbp
    
    # ymm8: scale
    vbroadcastss %xmm0, %ymm8
    
    # YMM Accumulators
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm9, %ymm9, %ymm9
    vxorps %ymm13, %ymm13, %ymm13
    vxorps %ymm14, %ymm14, %ymm14
    
    vmovdqa SHIFTS_ELUT(%rip), %ymm10
    vmovdqa VAL_ONE(%rip), %ymm11
    vmovdqa MASK_ELUT(%rip), %ymm12
    vmovdqa VAL_MINUS_ONE(%rip), %ymm15

.loop:
    cmp $64, %rdi
    jl .leftover

    # Prefetch tuned for i7-1260P (Alder Lake P-core):
    #   L1d=48KiB, L2=1.25MiB/core, dual-ch DDR4/LPDDR — hide ~80-120ns load.
    # Main loop steps 256 B of x + 32 B of W; distance covers ~2–4 iters.
    # prefetcht0 → L1 for reused activations; NTA for one-shot weight stream
    # (avoids polluting L2 with W that is never re-read in GEMV).
    prefetcht0 768(%rsi)
    prefetcht1 1536(%rsi)
    prefetchnta 256(%rdx)
    prefetchnta 512(%rdx)
    prefetchnta 1024(%rdx)


    # --- BLOCK 1 (8 weights -> acc %ymm0) ---
    vpbroadcastd 0(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 0(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- BLOCK 2 (8 weights -> acc %ymm9) ---
    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    # --- BLOCK 3 (8 weights -> acc %ymm13) ---
    vpbroadcastd 8(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 64(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13

    # --- BLOCK 4 (8 weights -> acc %ymm14) ---
    vpbroadcastd 12(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 96(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14

    # --- BLOCK 5 (8 weights -> acc %ymm0) ---
    vpbroadcastd 16(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 128(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- BLOCK 6 (8 weights -> acc %ymm9) ---
    vpbroadcastd 20(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 160(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    # --- BLOCK 7 (8 weights -> acc %ymm13) ---
    vpbroadcastd 24(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 192(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13

    # --- BLOCK 8 (8 weights -> acc %ymm14) ---
    vpbroadcastd 28(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 224(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14

    add $32, %rdx
    add $256, %rsi
    sub $64, %rdi
    jmp .loop

.leftover:
    test %rdi, %rdi
    jle .done_accum

    cmp $32, %rdi
    jl .leftover16


    # --- LEFTOVER BLOCK 1 (8 weights -> acc %ymm0) ---
    vpbroadcastd 0(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 0(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- LEFTOVER BLOCK 2 (8 weights -> acc %ymm9) ---
    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    # --- LEFTOVER BLOCK 3 (8 weights -> acc %ymm13) ---
    vpbroadcastd 8(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 64(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm13, %ymm13
    vsubps %ymm7, %ymm13, %ymm13

    # --- LEFTOVER BLOCK 4 (8 weights -> acc %ymm14) ---
    vpbroadcastd 12(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 96(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14

    add $16, %rdx
    add $128, %rsi
    sub $32, %rdi
    jmp .leftover

.leftover16:
    test %rdi, %rdi
    jle .done_accum

    cmp $16, %rdi
    jl .leftover8


    # --- LEFTOVER16 BLOCK 1 (8 weights -> acc %ymm0) ---
    vpbroadcastd 0(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 0(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    # --- LEFTOVER16 BLOCK 2 (8 weights -> acc %ymm9) ---
    vpbroadcastd 4(%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups 32(%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9

    add $8, %rdx
    add $64, %rsi
    sub $16, %rdi
    jmp .leftover16

.leftover8:
    test %rdi, %rdi
    jle .done_accum

    vpbroadcastd (%rdx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm15, %ymm2, %ymm4
    vmovups (%rsi), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    add $4, %rdx
    add $32, %rsi
    sub $8, %rdi
    jmp .leftover8

.done_accum:
    vmulps %ymm8, %ymm0, %ymm0
    vmulps %ymm8, %ymm9, %ymm9
    vmulps %ymm8, %ymm13, %ymm13
    vmulps %ymm8, %ymm14, %ymm14

    vaddps %ymm9, %ymm0, %ymm0
    vaddps %ymm14, %ymm13, %ymm13
    vaddps %ymm13, %ymm0, %ymm0

    # Horizontal reduce: extract + 2× vhaddps (tighter than shufps ladder)
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0

    # NaN/Inf guard: IEEE exponent all-ones → 0.0 (training safety)
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne .store_out
    vxorps %xmm0, %xmm0, %xmm0
.store_out:
    vmovss %xmm0, (%rcx)

    vzeroupper
    pop %rbp
    ret
