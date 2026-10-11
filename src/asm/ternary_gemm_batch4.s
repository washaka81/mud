.section .text
.global ternary_gemm_batch4_avx2

# Optimized Ternary GEMM for Batch=4 (4-bit ELUT format)
# Multiplies W [out_dim, in_dim] with X [4, in_dim] to produce OUT [4, out_dim].
# W is packed 8 ternary values per u32 (4-bit nibbles).

.section .rodata
.align 32
SHIFTS_ELUT: .long 0, 4, 8, 12, 16, 20, 24, 28
MASK_ELUT:   .long 15, 15, 15, 15, 15, 15, 15, 15
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_MINUS_ONE: .long 15, 15, 15, 15, 15, 15, 15, 15

.section .text
ternary_gemm_batch4_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    mov %rsi, %r10
    shl $2, %r10        # r10 = in_dim * 4 (bytes)
    lea (%rdx, %r10), %r11      # r11 = Token 1
    lea (%r11, %r10), %r12      # r12 = Token 2
    lea (%r12, %r10), %r13      # r13 = Token 3

    mov %rdi, %r15
    shl $2, %r15        # r15 = out_dim * 4 (bytes offset between tokens)

    # ymm10: SHIFTS, ymm12: MASK, ymm11: ONE, ymm13: MINUS_ONE
    vmovdqa SHIFTS_ELUT(%rip), %ymm10
    vmovdqa MASK_ELUT(%rip), %ymm12
    vmovdqa VAL_ONE(%rip), %ymm11
    vmovdqa VAL_MINUS_ONE(%rip), %ymm13

    xor %rbx, %rbx      # rbx = row counter

.row_loop:
    cmp %rdi, %rbx
    jge .done

    vmovss (%r9), %xmm0
    vbroadcastss %xmm0, %ymm8   # ymm8 = scale
    add $4, %r9

    # Accumulators for the 4 tokens
    vxorps %ymm0, %ymm0, %ymm0  # Token 0
    vxorps %ymm9, %ymm9, %ymm9  # Token 1
    vxorps %ymm14, %ymm14, %ymm14 # Token 2
    vxorps %ymm15, %ymm15, %ymm15 # Token 3

    mov %rsi, %r14  # Loop counter = in_dim

.inner_loop:
    cmp $16, %r14
    jl .leftover

    # Prefetch next packed weights + activation streams (memory-bound path)
    prefetchnta 256(%rcx)
    prefetcht0 512(%rdx)
    prefetcht0 512(%r11)
    prefetcht0 512(%r12)
    prefetcht0 512(%r13)

    # BLOCK 0
    vpbroadcastd 0(%rcx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm13, %ymm2, %ymm4
    vmovups 0(%rdx), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    vmovups 0(%r11), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    vmovups 0(%r12), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14
    vmovups 0(%r13), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm15, %ymm15
    vsubps %ymm7, %ymm15, %ymm15
    # BLOCK 1
    vpbroadcastd 4(%rcx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm13, %ymm2, %ymm4
    vmovups 32(%rdx), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    vmovups 32(%r11), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    vmovups 32(%r12), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14
    vmovups 32(%r13), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm15, %ymm15
    vsubps %ymm7, %ymm15, %ymm15
    add $8, %rcx
    add $64, %rdx
    add $64, %r11
    add $64, %r12
    add $64, %r13
    sub $16, %r14
    jmp .inner_loop

.leftover:
    test %r14, %r14
    jle .end_inner

    # leftover block (8 elements)
    vpbroadcastd 0(%rcx), %ymm1
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    vpcmpeqd %ymm11, %ymm2, %ymm3
    vpcmpeqd %ymm13, %ymm2, %ymm4
    vmovups 0(%rdx), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    vmovups 0(%r11), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm9, %ymm9
    vsubps %ymm7, %ymm9, %ymm9
    vmovups 0(%r12), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm14, %ymm14
    vsubps %ymm7, %ymm14, %ymm14
    vmovups 0(%r13), %ymm5
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    vaddps %ymm6, %ymm15, %ymm15
    vsubps %ymm7, %ymm15, %ymm15
    add $4, %rcx
    add $32, %rdx
    add $32, %r11
    add $32, %r12
    add $32, %r13
    sub $8, %r14
    jmp .leftover

.end_inner:
    # horizontal sum + scale; L-08: mask non-finite with andn of exp-all-ones compare
    # token 0
    vextractf128 $1, %ymm0, %xmm4
    vaddps %xmm4, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vmulss %xmm8, %xmm0, %xmm0
    # if exp==0xFF → zero: use scalar path via ucomiss self
    vucomiss %xmm0, %xmm0
    jnp .b4_t0_ok
    vxorps %xmm0, %xmm0, %xmm0
.b4_t0_ok:
    vmovss %xmm0, (%r8, %rbx, 4)
    # token 1
    vextractf128 $1, %ymm9, %xmm4
    vaddps %xmm4, %xmm9, %xmm9
    vhaddps %xmm9, %xmm9, %xmm9
    vhaddps %xmm9, %xmm9, %xmm9
    vmulss %xmm8, %xmm9, %xmm9
    vucomiss %xmm9, %xmm9
    jnp .b4_t1_ok
    vxorps %xmm9, %xmm9, %xmm9
.b4_t1_ok:
    mov %r8, %rax
    add %r15, %rax
    vmovss %xmm9, (%rax, %rbx, 4)
    # token 2
    vextractf128 $1, %ymm14, %xmm4
    vaddps %xmm4, %xmm14, %xmm14
    vhaddps %xmm14, %xmm14, %xmm14
    vhaddps %xmm14, %xmm14, %xmm14
    vmulss %xmm8, %xmm14, %xmm14
    vucomiss %xmm14, %xmm14
    jnp .b4_t2_ok
    vxorps %xmm14, %xmm14, %xmm14
.b4_t2_ok:
    add %r15, %rax
    vmovss %xmm14, (%rax, %rbx, 4)
    # token 3
    vextractf128 $1, %ymm15, %xmm4
    vaddps %xmm4, %xmm15, %xmm15
    vhaddps %xmm15, %xmm15, %xmm15
    vhaddps %xmm15, %xmm15, %xmm15
    vmulss %xmm8, %xmm15, %xmm15
    vucomiss %xmm15, %xmm15
    jnp .b4_t3_ok
    vxorps %xmm15, %xmm15, %xmm15
.b4_t3_ok:
    add %r15, %rax
    vmovss %xmm15, (%rax, %rbx, 4)

    # Reset X pointers
    sub %r10, %rdx
    sub %r10, %r11
    sub %r10, %r12
    sub %r10, %r13
    inc %rbx
    jmp .row_loop
.done:
    pop %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    vzeroupper
    ret
