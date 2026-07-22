.section .text
.global lm_head_avx2
.global lm_head_logits_avx2
.type lm_head_avx2, @function
.type lm_head_logits_avx2, @function

# ─────────────────────────────────────────────────────────────────────────────
# Shared layout helpers (System V AMD64):
#   vocab_size → r12, hidden → r13, regs → r14, weights → r15, stride_bytes → rbx
# ─────────────────────────────────────────────────────────────────────────────

# lm_head_logits_avx2(vocab_size, hidden, regs, weights, out_logits)
# rdi=vocab, rsi=hidden, rdx=regs, rcx=weights, r8=out_logits (*mut f32[vocab])
# Fills full logit vector (needed for Top-P / Softmax). No argmax.
lm_head_logits_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    mov %rdi, %r12              # vocab
    mov %rsi, %r13              # hidden
    mov %rdx, %r14              # regs
    mov %rcx, %r15              # weights
    mov %r8, %r10               # out_logits (keep r10; r8 used as index)

    mov %r13, %rbx
    shl $2, %rbx                # stride_bytes = hidden * 4

    xor %r9, %r9                # row index
    test %r12, %r12
    jz .ll_done

.ll_vocab_loop:
    # Prefetch next weight rows (streaming LM head matrix)
    lea (%r15, %rbx), %rax
    prefetchnta (%rax)
    lea (%rax, %rbx), %rax
    prefetchnta (%rax)

    # --- inlined dual-acc FMA dot(regs, weight_row) ---
    mov %r14, %rsi
    mov %r15, %rdx
    mov %r13, %rdi
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1

.ll_dot16:
    cmp $16, %rdi
    jl .ll_dot8
    prefetcht0 128(%rsi)
    prefetcht0 128(%rdx)
    vmovups (%rsi), %ymm2
    vmovups (%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm3, %ymm0
    vmovups 32(%rsi), %ymm2
    vmovups 32(%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm3, %ymm1
    add $64, %rsi
    add $64, %rdx
    sub $16, %rdi
    jmp .ll_dot16

.ll_dot8:
    cmp $8, %rdi
    jl .ll_dot_tail
    vmovups (%rsi), %ymm2
    vmovups (%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm3, %ymm0
    add $32, %rsi
    add $32, %rdx
    sub $8, %rdi

.ll_dot_tail:
    test %rdi, %rdi
    jz .ll_hsum
.ll_dot1:
    vmovss (%rsi), %xmm2
    vmulss (%rdx), %xmm2, %xmm2
    vaddss %xmm2, %xmm0, %xmm0
    add $4, %rsi
    add $4, %rdx
    dec %rdi
    jnz .ll_dot1

.ll_hsum:
    vaddps %ymm1, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    # L-08: non-finite logit → 0 (avoids poisoning Top-P / Softmax)
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 1f
    vxorps %xmm0, %xmm0, %xmm0
1:
    vmovss %xmm0, (%r10, %r9, 4)

    add %rbx, %r15
    inc %r9
    cmp %r12, %r9
    jb .ll_vocab_loop

.ll_done:
    vzeroupper
    pop %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret

# lm_head_avx2(vocab_size, hidden, regs, weights) -> best vocab index (argmax)
# rdi=vocab, rsi=hidden, rdx=regs, rcx=weights
lm_head_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    mov %rdi, %r12
    mov %rsi, %r13
    mov %rdx, %r14
    mov %rcx, %r15

    mov %r13, %rbx
    shl $2, %rbx

    vmovss .neg_inf(%rip), %xmm15
    xor %r8, %r8                # best_id
    xor %r9, %r9                # row

    test %r12, %r12
    jz .lm_done

.lm_vocab_loop:
    lea (%r15, %rbx), %rax
    prefetchnta (%rax)
    lea (%rax, %rbx), %rax
    prefetchnta (%rax)

    mov %r14, %rsi
    mov %r15, %rdx
    mov %r13, %rdi
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1

.lm_dot16:
    cmp $16, %rdi
    jl .lm_dot8
    prefetcht0 128(%rsi)
    prefetcht0 128(%rdx)
    vmovups (%rsi), %ymm2
    vmovups (%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm3, %ymm0
    vmovups 32(%rsi), %ymm2
    vmovups 32(%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm3, %ymm1
    add $64, %rsi
    add $64, %rdx
    sub $16, %rdi
    jmp .lm_dot16

.lm_dot8:
    cmp $8, %rdi
    jl .lm_dot_tail
    vmovups (%rsi), %ymm2
    vmovups (%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm3, %ymm0
    add $32, %rsi
    add $32, %rdx
    sub $8, %rdi

.lm_dot_tail:
    test %rdi, %rdi
    jz .lm_hsum
.lm_dot1:
    vmovss (%rsi), %xmm2
    vmulss (%rdx), %xmm2, %xmm2
    vaddss %xmm2, %xmm0, %xmm0
    add $4, %rsi
    add $4, %rdx
    dec %rdi
    jnz .lm_dot1

.lm_hsum:
    vaddps %ymm1, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vcomiss %xmm15, %xmm0
    jbe .lm_skip_update
    vmovaps %xmm0, %xmm15
    mov %r9, %r8

.lm_skip_update:
    add %rbx, %r15
    inc %r9
    cmp %r12, %r9
    jb .lm_vocab_loop

.lm_done:
    mov %r8, %rax
    vzeroupper
    pop %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret

.section .rodata
.align 4
.neg_inf:
    .long 0xFF800000
