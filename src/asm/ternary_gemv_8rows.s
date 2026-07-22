.section .text
.global ternary_gemv_8rows_avx2

# Ternary GEMV — 8 rows × shared activation load (ELUT 4-bit)
# Target: i7-1260P Alder Lake (AVX2 only, no AVX-512)
#
# System V AMD64:
#   rdi = n (cols; multiple of 8 preferred; remainder <8 ignored)
#   rsi = x (f32 activations)
#   rdx = weights row0 (u32 packed; rows spaced by stride u32s)
#   rcx = out (8× f32)
#   xmm0 = scale (applied to all 8 outputs)
#   r8  = stride in u32 elements between consecutive rows
#
# Rationale (ASM_VULKAN_BOTTLENECK / T11):
#   4-row kernel reuses x across 4 rows; 8-row doubles that reuse so each
#   activation load feeds twice the matmul work — biggest win on BW-bound
#   GEMV (Bonsai FWD, Iris Xe host with MUD_GPU_GEMV=0).
# Main loop: 16 cols/iter (same 2× unroll as 4rows).

.section .rodata
.align 32
SHIFTS_ELUT8: .long 0, 4, 8, 12, 16, 20, 24, 28
MASK_ELUT8:   .long 15, 15, 15, 15, 15, 15, 15, 15
VAL_ONE8:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_MINUS_ONE8: .long 15, 15, 15, 15, 15, 15, 15, 15

.section .text
ternary_gemv_8rows_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    # save out ptr + scale (scale only needed at reduce)
    push %rcx
    sub $8, %rsp
    vmovss %xmm0, (%rsp)

    # ymm0..ymm7 = row accumulators
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1
    vxorps %ymm2, %ymm2, %ymm2
    vxorps %ymm3, %ymm3, %ymm3
    vxorps %ymm4, %ymm4, %ymm4
    vxorps %ymm5, %ymm5, %ymm5
    vxorps %ymm6, %ymm6, %ymm6
    vxorps %ymm7, %ymm7, %ymm7

    vmovdqa SHIFTS_ELUT8(%rip), %ymm11
    vmovdqa MASK_ELUT8(%rip), %ymm12
    vmovdqa VAL_ONE8(%rip), %ymm13
    vmovdqa VAL_MINUS_ONE8(%rip), %ymm14

    # Build 8 row base pointers (stride in bytes)
    mov %r8, %rax
    shl $2, %rax              # stride_bytes
    lea (%rdx, %rax), %r12    # row1
    lea (%r12, %rax), %r13    # row2
    lea (%r13, %rax), %r14    # row3
    lea (%r14, %rax), %r15    # row4
    lea (%r15, %rax), %rbx    # row5
    lea (%rbx, %rax), %r10    # row6
    lea (%r10, %rax), %r11    # row7
    # rdx = row0

    mov %rdi, %r9             # remaining cols

# ── macro-ish: ACC_WORD(row_ptr_reg, acc_ymm, x_ymm, offset_bytes) ──────────
# Implemented inline per row for assembler simplicity.

.loop16:
    cmp $16, %r9
    jl .loop8

    # Sparse prefetch (heavy NTA on all 8 rows thrashed L2 on 1260P hot path)
    prefetcht0 512(%rsi)
    prefetchnta 256(%rdx)     # row0 stream
    prefetchnta 256(%r15)     # row4 — covers mid pack

    vmovups 0(%rsi), %ymm9    # x[0:8]
    vmovups 32(%rsi), %ymm10  # x[8:16]

    # ── word0 @ x[0:8] ──
    # Row 0 → ymm0
    vpbroadcastd 0(%rdx), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm0, %ymm0

    # Row 1 → ymm1
    vpbroadcastd 0(%r12), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm1, %ymm1
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm1, %ymm1

    # Row 2 → ymm2
    vpbroadcastd 0(%r13), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm2, %ymm2
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm2, %ymm2

    # Row 3 → ymm3
    vpbroadcastd 0(%r14), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm3, %ymm3
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm3, %ymm3

    # Row 4 → ymm4
    vpbroadcastd 0(%r15), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm4, %ymm4
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm4, %ymm4

    # Row 5 → ymm5
    vpbroadcastd 0(%rbx), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm5, %ymm5
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm5, %ymm5

    # Row 6 → ymm6
    vpbroadcastd 0(%r10), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm6, %ymm6
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm6, %ymm6

    # Row 7 → ymm7
    vpbroadcastd 0(%r11), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm7, %ymm7
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm7, %ymm7

    # ── word1 @ x[8:16] ──
    # Row 0
    vpbroadcastd 4(%rdx), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm0, %ymm0

    # Row 1
    vpbroadcastd 4(%r12), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm1, %ymm1
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm1, %ymm1

    # Row 2
    vpbroadcastd 4(%r13), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm2, %ymm2
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm2, %ymm2

    # Row 3
    vpbroadcastd 4(%r14), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm3, %ymm3
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm3, %ymm3

    # Row 4
    vpbroadcastd 4(%r15), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm4, %ymm4
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm4, %ymm4

    # Row 5
    vpbroadcastd 4(%rbx), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm5, %ymm5
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm5, %ymm5

    # Row 6
    vpbroadcastd 4(%r10), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm6, %ymm6
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm6, %ymm6

    # Row 7
    vpbroadcastd 4(%r11), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm10, %ymm8
    vaddps %ymm8, %ymm7, %ymm7
    vpand %ymm15, %ymm10, %ymm15
    vsubps %ymm15, %ymm7, %ymm7

    add $8, %rdx
    add $8, %r12
    add $8, %r13
    add $8, %r14
    add $8, %r15
    add $8, %rbx
    add $8, %r10
    add $8, %r11
    add $64, %rsi
    sub $16, %r9
    jmp .loop16

# ── 8 cols / iter (tail) ────────────────────────────────────────────────────
.loop8:
    cmp $8, %r9
    jl .done_accum

    prefetcht0 512(%rsi)
    prefetchnta 128(%rdx)

    vmovups (%rsi), %ymm9

    # Row 0
    vpbroadcastd (%rdx), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm0, %ymm0
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm0, %ymm0

    # Row 1
    vpbroadcastd (%r12), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm1, %ymm1
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm1, %ymm1

    # Row 2
    vpbroadcastd (%r13), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm2, %ymm2
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm2, %ymm2

    # Row 3
    vpbroadcastd (%r14), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm3, %ymm3
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm3, %ymm3

    # Row 4
    vpbroadcastd (%r15), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm4, %ymm4
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm4, %ymm4

    # Row 5
    vpbroadcastd (%rbx), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm5, %ymm5
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm5, %ymm5

    # Row 6
    vpbroadcastd (%r10), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm6, %ymm6
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm6, %ymm6

    # Row 7
    vpbroadcastd (%r11), %ymm8
    vpsrlvd %ymm11, %ymm8, %ymm15
    vpand %ymm12, %ymm15, %ymm15
    vpcmpeqd %ymm13, %ymm15, %ymm8
    vpcmpeqd %ymm14, %ymm15, %ymm15
    vpand %ymm8, %ymm9, %ymm8
    vaddps %ymm8, %ymm7, %ymm7
    vpand %ymm15, %ymm9, %ymm15
    vsubps %ymm15, %ymm7, %ymm7

    add $4, %rdx
    add $4, %r12
    add $4, %r13
    add $4, %r14
    add $4, %r15
    add $4, %rbx
    add $4, %r10
    add $4, %r11
    add $32, %rsi
    sub $8, %r9
    jmp .loop8

.done_accum:
    # scale = (%rsp); out = 8(%rsp) after sub $8
    vmovss (%rsp), %xmm15
    vbroadcastss %xmm15, %ymm15
    mov 8(%rsp), %rcx

    vmulps %ymm15, %ymm0, %ymm0
    vmulps %ymm15, %ymm1, %ymm1
    vmulps %ymm15, %ymm2, %ymm2
    vmulps %ymm15, %ymm3, %ymm3
    vmulps %ymm15, %ymm4, %ymm4
    vmulps %ymm15, %ymm5, %ymm5
    vmulps %ymm15, %ymm6, %ymm6
    vmulps %ymm15, %ymm7, %ymm7

    # Horizontal reduce + NaN/Inf kill → store (same pattern as 4rows)
    # Row 0
    vextractf128 $1, %ymm0, %xmm8
    vaddps %xmm8, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 1f
    vxorps %xmm0, %xmm0, %xmm0
1:  vmovss %xmm0, 0(%rcx)

    # Row 1
    vextractf128 $1, %ymm1, %xmm8
    vaddps %xmm8, %xmm1, %xmm1
    vhaddps %xmm1, %xmm1, %xmm1
    vhaddps %xmm1, %xmm1, %xmm1
    vmovd %xmm1, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 2f
    vxorps %xmm1, %xmm1, %xmm1
2:  vmovss %xmm1, 4(%rcx)

    # Row 2
    vextractf128 $1, %ymm2, %xmm8
    vaddps %xmm8, %xmm2, %xmm2
    vhaddps %xmm2, %xmm2, %xmm2
    vhaddps %xmm2, %xmm2, %xmm2
    vmovd %xmm2, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 3f
    vxorps %xmm2, %xmm2, %xmm2
3:  vmovss %xmm2, 8(%rcx)

    # Row 3
    vextractf128 $1, %ymm3, %xmm8
    vaddps %xmm8, %xmm3, %xmm3
    vhaddps %xmm3, %xmm3, %xmm3
    vhaddps %xmm3, %xmm3, %xmm3
    vmovd %xmm3, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 4f
    vxorps %xmm3, %xmm3, %xmm3
4:  vmovss %xmm3, 12(%rcx)

    # Row 4
    vextractf128 $1, %ymm4, %xmm8
    vaddps %xmm8, %xmm4, %xmm4
    vhaddps %xmm4, %xmm4, %xmm4
    vhaddps %xmm4, %xmm4, %xmm4
    vmovd %xmm4, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 5f
    vxorps %xmm4, %xmm4, %xmm4
5:  vmovss %xmm4, 16(%rcx)

    # Row 5
    vextractf128 $1, %ymm5, %xmm8
    vaddps %xmm8, %xmm5, %xmm5
    vhaddps %xmm5, %xmm5, %xmm5
    vhaddps %xmm5, %xmm5, %xmm5
    vmovd %xmm5, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 6f
    vxorps %xmm5, %xmm5, %xmm5
6:  vmovss %xmm5, 20(%rcx)

    # Row 6
    vextractf128 $1, %ymm6, %xmm8
    vaddps %xmm8, %xmm6, %xmm6
    vhaddps %xmm6, %xmm6, %xmm6
    vhaddps %xmm6, %xmm6, %xmm6
    vmovd %xmm6, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 7f
    vxorps %xmm6, %xmm6, %xmm6
7:  vmovss %xmm6, 24(%rcx)

    # Row 7
    vextractf128 $1, %ymm7, %xmm8
    vaddps %xmm8, %xmm7, %xmm7
    vhaddps %xmm7, %xmm7, %xmm7
    vhaddps %xmm7, %xmm7, %xmm7
    vmovd %xmm7, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 8f
    vxorps %xmm7, %xmm7, %xmm7
8:  vmovss %xmm7, 28(%rcx)

    vzeroupper
    add $8, %rsp
    pop %rcx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    pop %rbp
    ret
