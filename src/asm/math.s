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

.global peak_abs_avx2
# peak_abs_avx2(n: usize, x: *const f32) -> f32
peak_abs_avx2:
    push %rbp
    mov %rsp, %rbp
    vxorps %ymm0, %ymm0, %ymm0    # ymm0 will store the max values
    
    # Mask to clear the sign bit (absolute value)
    vmovups .abs_mask(%rip), %ymm4

.peak_loop:
    cmp $16, %rdi
    jl .peak_leftover
    vmovups (%rsi), %ymm1
    vmovups 32(%rsi), %ymm2
    
    vandps %ymm4, %ymm1, %ymm1    # abs(x1)
    vandps %ymm4, %ymm2, %ymm2    # abs(x2)
    
    vmaxps %ymm1, %ymm0, %ymm0    # update max
    vmaxps %ymm2, %ymm0, %ymm0
    
    add $64, %rsi
    sub $16, %rdi
    jmp .peak_loop

.peak_leftover:
    cmp $8, %rdi
    jl .peak_done
    vmovups (%rsi), %ymm1
    vandps %ymm4, %ymm1, %ymm1
    vmaxps %ymm1, %ymm0, %ymm0
    add $32, %rsi
    sub $8, %rdi

.peak_done:
    # Horizontal max across ymm0
    vextractf128 $1, %ymm0, %xmm1
    vmaxps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vmaxps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vmaxps %xmm1, %xmm0, %xmm0
    
    vzeroupper
    pop %rbp
    ret

.global apply_gradient_avx2
# apply_gradient_avx2(n: usize, weight: *mut f32, grad: *const f32, alpha: f32, decay: f32)
apply_gradient_avx2:
    push %rbp
    mov %rsp, %rbp
    vbroadcastss %xmm0, %ymm0    # ymm0 = alpha (correct xmm0)
    vbroadcastss %xmm1, %ymm1    # ymm1 = decay (correct xmm1)
    
    # 1.0 - decay
    vmovups .one(%rip), %ymm5
    vsubps %ymm1, %ymm5, %ymm3    # ymm3 = 1.0 - decay (was using ymm3 for result)

.grad_loop:
    cmp $8, %rdi
    jl .grad_done
    prefetcht0 128(%rsi)
    prefetcht0 128(%rdx)
    vmovups (%rsi), %ymm2        # ymm2 = weight
    vmovups (%rdx), %ymm4        # ymm4 = grad
    
    # 🛡️ CLAMPING: Clamp gradient to [-1.0, 1.0] before applying
    vmaxps .min_grad(%rip), %ymm4, %ymm4
    vminps .max_grad(%rip), %ymm4, %ymm4
    
    # 🛡️ SANITIZATION: weight = weight * (1.0 - decay) + alpha * grad
    vmulps %ymm3, %ymm2, %ymm2
    vfmadd231ps %ymm4, %ymm0, %ymm2
    
    vmovups %ymm2, (%rsi)
    add $32, %rsi
    add $32, %rdx
    sub $8, %rdi
    jmp .grad_loop

.grad_done:
    vzeroupper
    pop %rbp
    ret

.global hadamard_transform_avx2
# hadamard_transform_avx2(n: usize, x: *mut f32)
# Iterative In-place Fast Walsh-Hadamard Transform
hadamard_transform_avx2:
    push %rbp
    mov %rsp, %rbp
    
    # rdi = n (must be power of 2)
    # rsi = x (data pointer)
    
    # We use r8 for 's' (step size), r9 for 'i' (outer loop), r10 for 'j' (inner loop)
    mov $1, %r8          # s = 1

.h_stage_loop:
    cmp %rdi, %r8
    jge .h_done          # if s >= n, done
    
    xor %r9, %r9         # i = 0

.h_outer_loop:
    mov %r9, %rax
    add %r8, %rax        # i + s
    cmp %rdi, %rax
    jge .h_stage_next    # if i + s >= n, next stage
    
    xor %r10, %r10       # j = 0

.h_inner_loop:
    # Vectorized path if s >= 8
    cmp $8, %r8
    jl .h_scalar_path
    
    # AVX2 Path
.h_avx_loop:
    # a = x[i + j], b = x[i + j + s]
    mov %r9, %rax
    add %r10, %rax       # i + j
    shl $2, %rax         # * 4
    add %rsi, %rax       # &x[i+j]
    
    mov %r8, %rbx
    shl $2, %rbx         # s * 4
    
    vmovups (%rax), %ymm1       # ymm1 = a
    vmovups (%rax, %rbx), %ymm2 # ymm2 = b
    
    vaddps %ymm2, %ymm1, %ymm3  # ymm3 = a + b
    vsubps %ymm2, %ymm1, %ymm4  # ymm4 = a - b
    
    vmovups %ymm3, (%rax)
    vmovups %ymm4, (%rax, %rbx)
    
    add $8, %r10
    cmp %r8, %r10
    jl .h_avx_loop
    
    # Advance i to next block: i += 2*s
    mov %r8, %rax
    shl $1, %rax         # 2*s
    add %rax, %r9
    jmp .h_outer_loop

.h_scalar_path:
    # a = x[i + j], b = x[i + j + s]
    mov %r9, %rax
    add %r10, %rax       # i + j
    shl $2, %rax
    add %rsi, %rax
    
    mov %r8, %rbx
    shl $2, %rbx
    
    vmovss (%rax), %xmm1
    vmovss (%rax, %rbx), %xmm2
    
    vaddss %xmm2, %xmm1, %xmm3
    vsubss %xmm2, %xmm1, %xmm4
    
    vmovss %xmm3, (%rax)
    vmovss %xmm4, (%rax, %rbx)
    
    inc %r10
    cmp %r8, %r10
    jl .h_scalar_path
    
    # Advance i
    mov %r8, %rax
    shl $1, %rax
    add %rax, %r9
    jmp .h_outer_loop

.h_stage_next:
    shl $1, %r8          # s *= 2
    jmp .h_stage_loop

.h_done:
    vzeroupper
    pop %rbp
    ret

.section .rodata
.align 32
.abs_mask:
    .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
.one:
    .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
.min_grad:
    .float -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
.max_grad:
    .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
