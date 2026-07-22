.section .text
.global ternary_gemv_4rows_avx2

# Optimized Ternary GEMV — 4 rows × shared activation load (ELUT 4-bit)
# System V AMD64:
#   rdi = n (cols; should be multiple of 8; remainder < 8 ignored safely)
#   rsi = x (f32 activations)
#   rdx = weights row0 (u32 packed; rows spaced by stride u32s)
#   rcx = out (4× f32)
#   xmm0 = scale
#   r8  = stride in u32 elements between consecutive rows
#
# Hot-path improvement (2026-07): main loop processes **16 cols / iter**
# (2× u32 words × 4 rows, 2× YMM of activations) to cut loop overhead ~2×
# vs previous 8-col loop. Fallback 8-col path for tail.

.section .rodata
.align 32
SHIFTS_ELUT: .long 0, 4, 8, 12, 16, 20, 24, 28
MASK_ELUT:   .long 15, 15, 15, 15, 15, 15, 15, 15
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_MINUS_ONE: .long 15, 15, 15, 15, 15, 15, 15, 15

.section .text
ternary_gemv_4rows_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14

    vbroadcastss %xmm0, %ymm15

    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1
    vxorps %ymm2, %ymm2, %ymm2
    vxorps %ymm3, %ymm3, %ymm3

    vmovdqa SHIFTS_ELUT(%rip), %ymm10
    vmovdqa MASK_ELUT(%rip), %ymm12
    vmovdqa VAL_ONE(%rip), %ymm13
    vmovdqa VAL_MINUS_ONE(%rip), %ymm14

    mov %r8, %rax
    shl $2, %rax              # stride in bytes
    lea (%rdx, %rax), %r12    # row1
    lea (%r12, %rax), %r13    # row2
    lea (%r13, %rax), %r14    # row3

    mov %rdi, %r9

# ── 16 cols / iter: load x[0:8] and x[8:16], one u32 weight word each ────────
.loop16:
    cmp $16, %r9
    jl .loop8

    # Prefetch: i7-1260P — T0 for x (reused), NTA for streamed W rows
    prefetcht0 768(%rsi)
    prefetcht1 1536(%rsi)
    prefetchnta 256(%rdx)
    prefetchnta 256(%r12)
    prefetchnta 256(%r13)
    prefetchnta 256(%r14)
    prefetchnta 512(%rdx)
    prefetchnta 512(%r12)

    vmovups 0(%rsi), %ymm4
    vmovups 32(%rsi), %ymm5

    # ── word0 @ x[0:8] ──
    # Row 0
    vpbroadcastd (%rdx), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm4, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm7, %ymm4, %ymm7
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

    # ── word1 @ x[8:16] ──
    # Row 0
    vpbroadcastd 4(%rdx), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm0, %ymm0

    # Row 1
    vpbroadcastd 4(%r12), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm1, %ymm1
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm1, %ymm1

    # Row 2
    vpbroadcastd 4(%r13), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm2, %ymm2
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm2, %ymm2

    # Row 3
    vpbroadcastd 4(%r14), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm5, %ymm8
    vaddps %ymm8, %ymm3, %ymm3
    vpand %ymm7, %ymm5, %ymm7
    vsubps %ymm7, %ymm3, %ymm3

    add $8, %rdx
    add $8, %r12
    add $8, %r13
    add $8, %r14
    add $64, %rsi
    sub $16, %r9
    jmp .loop16

# ── 8 cols / iter (tail) ────────────────────────────────────────────────────
.loop8:
    cmp $8, %r9
    jl .done_accum

    prefetcht0 512(%rsi)
    prefetchnta 256(%rdx)

    vmovups (%rsi), %ymm4

    # Row 0
    vpbroadcastd (%rdx), %ymm6
    vpsrlvd %ymm10, %ymm6, %ymm7
    vpand %ymm12, %ymm7, %ymm7
    vpcmpeqd %ymm13, %ymm7, %ymm8
    vpcmpeqd %ymm14, %ymm7, %ymm7
    vpand %ymm8, %ymm4, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm7, %ymm4, %ymm7
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

    add $4, %rdx
    add $4, %r12
    add $4, %r13
    add $4, %r14
    add $32, %rsi
    sub $8, %r9
    jmp .loop8

.done_accum:
    vmulps %ymm15, %ymm0, %ymm0
    vmulps %ymm15, %ymm1, %ymm1
    vmulps %ymm15, %ymm2, %ymm2
    vmulps %ymm15, %ymm3, %ymm3

    # Horizontal reduce + NaN/Inf kill → store
    # Row 0
    vextractf128 $1, %ymm0, %xmm4
    vaddps %xmm4, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 1f
    vxorps %xmm0, %xmm0, %xmm0
1:  vmovss %xmm0, (%rcx)

    # Row 1
    vextractf128 $1, %ymm1, %xmm4
    vaddps %xmm4, %xmm1, %xmm1
    vhaddps %xmm1, %xmm1, %xmm1
    vhaddps %xmm1, %xmm1, %xmm1
    vmovd %xmm1, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 2f
    vxorps %xmm1, %xmm1, %xmm1
2:  vmovss %xmm1, 4(%rcx)

    # Row 2
    vextractf128 $1, %ymm2, %xmm4
    vaddps %xmm4, %xmm2, %xmm2
    vhaddps %xmm2, %xmm2, %xmm2
    vhaddps %xmm2, %xmm2, %xmm2
    vmovd %xmm2, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 3f
    vxorps %xmm2, %xmm2, %xmm2
3:  vmovss %xmm2, 8(%rcx)

    # Row 3
    vextractf128 $1, %ymm3, %xmm4
    vaddps %xmm4, %xmm3, %xmm3
    vhaddps %xmm3, %xmm3, %xmm3
    vhaddps %xmm3, %xmm3, %xmm3
    vmovd %xmm3, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 4f
    vxorps %xmm3, %xmm3, %xmm3
4:  vmovss %xmm3, 12(%rcx)

    vzeroupper
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret
