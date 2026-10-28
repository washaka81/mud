.section .text
.global dot_product_avx2
.global sum_squares_avx2

# dot_product_avx2(n: usize, a: *const f32, b: *const f32) -> f32
dot_product_avx2:
    push %rbp
    mov %rsp, %rbp
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm3, %ymm3, %ymm3
    vxorps %xmm4, %xmm4, %xmm4
.dot_loop:
    cmp $16, %rdi
    jl .dot_leftover
    prefetcht0 128(%rsi)
    prefetcht0 128(%rdx)
    vmovups (%rsi), %ymm1
    vmovups (%rdx), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm0
    vmovups 32(%rsi), %ymm1
    vmovups 32(%rdx), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm3
    add $64, %rsi
    add $64, %rdx
    sub $16, %rdi
    jmp .dot_loop
.dot_leftover:
    cmp $8, %rdi
    jl .dot_done
    vmovups (%rsi), %ymm1
    vmovups (%rdx), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm0
    add $32, %rsi
    add $32, %rdx
    sub $8, %rdi
.dot_done:
    # scalar leftover 0..7
    # scalar leftover: accumulate into xmm4, then fold into ymm0 at hsum
    vxorps %xmm4, %xmm4, %xmm4
    test %rdi, %rdi
    jz .dot_hsum
.dot_scalar:
    vmovss (%rsi), %xmm1
    vmulss (%rdx), %xmm1, %xmm1
    vaddss %xmm1, %xmm4, %xmm4
    add $4, %rsi
    add $4, %rdx
    dec %rdi
    jnz .dot_scalar
    vaddss %xmm4, %xmm0, %xmm0
.dot_hsum:
    vaddps %ymm3, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    # L-08: NaN/Inf → 0.0 (IEEE exp all-ones)
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne .dot_ret
    vxorps %xmm0, %xmm0, %xmm0
.dot_ret:
    vzeroupper
    pop %rbp
    ret

# sum_squares_avx2(n: usize, x: *const f32) -> f32
sum_squares_avx2:
    push %rbp
    mov %rsp, %rbp
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm2, %ymm2, %ymm2
.ss_loop:
    cmp $16, %rdi
    jl .ss_leftover
    vmovups (%rsi), %ymm1
    vfmadd231ps %ymm1, %ymm1, %ymm0
    vmovups 32(%rsi), %ymm1
    vfmadd231ps %ymm1, %ymm1, %ymm2
    add $64, %rsi
    sub $16, %rdi
    jmp .ss_loop
.ss_leftover:
    cmp $8, %rdi
    jl .ss_done
    vmovups (%rsi), %ymm1
    vfmadd231ps %ymm1, %ymm1, %ymm0
    add $32, %rsi
    sub $8, %rdi
.ss_done:
    # scalar leftover 0..7
    test %rdi, %rdi
    jz .ss_hsum
.ss_scalar:
    vmovss (%rsi), %xmm1
    vmulss %xmm1, %xmm1, %xmm1
    vaddss %xmm1, %xmm0, %xmm0
    add $4, %rsi
    dec %rdi
    jnz .ss_scalar
.ss_hsum:
    vaddps %ymm2, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0
    # L-08: NaN/Inf → 0.0
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne .ss_ret
    vxorps %xmm0, %xmm0, %xmm0
.ss_ret:
    vzeroupper
    pop %rbp
    ret

# P-08 purge (2026-08-03): removed dead kernels peak_abs_avx2,
# apply_gradient_avx2, hadamard_transform_avx2 — compiled but never declared
# via extern "C" in asm/mod.rs (0 callers). The Rust-side ezop::peak_abs_avx2
# (inline intrinsics) is the live path. Removed their .rodata constants too
# (.abs_mask, .one, .min_grad, .max_grad — only used by the deleted kernels).
