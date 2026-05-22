.section .text
.global dot_product_avx2
.global sum_squares_avx2

# dot_product_avx2(n: usize, a: *const f32, b: *const f32) -> f32
dot_product_avx2:
    push %rbp
    mov %rsp, %rbp
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm3, %ymm3, %ymm3
.dot_loop:
    cmp $16, %rdi
    jl .dot_leftover
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
    vaddps %ymm3, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
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
    vaddps %ymm2, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vzeroupper
    pop %rbp
    ret
