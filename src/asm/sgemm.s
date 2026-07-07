.section .text
.global sgemm_abt_avx2
.global sgemm_avx2

# sgemm_abt_avx2(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32)
# Computes C = A * B^T
# %rdi = m, %rsi = n, %rdx = k, %rcx = a, %r8 = b, %r9 = c
sgemm_abt_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    mov %rdi, %r10      # m
    mov %rsi, %r11      # n
    mov %rdx, %r12      # k
    mov %rcx, %r13      # a
    mov %r8,  %r14      # b
    mov %r9,  %r15      # c

    xor %rax, %rax
.L_abt_i_loop:
    cmp %r10, %rax
    jge .L_abt_done

    xor %rbx, %rbx
.L_abt_j_loop:
    cmp %r11, %rbx
    jge .L_abt_j_done

    mov %rax, %rcx
    imul %r12, %rcx
    lea (%r13, %rcx, 4), %rcx

    mov %rbx, %rdx
    imul %r12, %rdx
    lea (%r14, %rdx, 4), %rdx

    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm3, %ymm3, %ymm3

    xor %r8, %r8
.L_abt_p_loop:
    mov %r12, %r9
    sub %r8, %r9
    cmp $16, %r9
    jl .L_abt_p_leftover

    vmovups (%rcx, %r8, 4), %ymm1
    vmovups (%rdx, %r8, 4), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm0

    vmovups 32(%rcx, %r8, 4), %ymm1
    vmovups 32(%rdx, %r8, 4), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm3

    add $16, %r8
    jmp .L_abt_p_loop

.L_abt_p_leftover:
    cmp $8, %r9
    jl .L_abt_p_scalar

    vmovups (%rcx, %r8, 4), %ymm1
    vmovups (%rdx, %r8, 4), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm0

    add $8, %r8

.L_abt_p_scalar:
    vaddps %ymm3, %ymm0, %ymm0
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0

.L_abt_p_scalar_loop:
    cmp %r12, %r8
    jge .L_abt_p_store
    vmovss (%rcx, %r8, 4), %xmm1
    vmovss (%rdx, %r8, 4), %xmm2
    vmulss %xmm1, %xmm2, %xmm1
    vaddss %xmm1, %xmm0, %xmm0
    inc %r8
    jmp .L_abt_p_scalar_loop

.L_abt_p_store:
    mov %rax, %rcx
    imul %r11, %rcx
    add %rbx, %rcx
    vmovss %xmm0, (%r15, %rcx, 4)

    inc %rbx
    jmp .L_abt_j_loop

.L_abt_j_done:
    inc %rax
    jmp .L_abt_i_loop

.L_abt_done:
    vzeroupper
    pop %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret


# sgemm_avx2(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32)
# Computes C = A * B using IKJ order for true AVX2 vectorization
# %rdi = m, %rsi = n, %rdx = k, %rcx = a, %r8 = b, %r9 = c
sgemm_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    mov %rdi, %r10      # m
    mov %rsi, %r11      # n
    mov %rdx, %r12      # k
    mov %rcx, %r13      # a
    mov %r8,  %r14      # b
    mov %r9,  %r15      # c

    xor %rax, %rax      # i = 0
.L_ikj_i_loop:
    cmp %r10, %rax
    jge .L_ikj_done

    # Zero C[i, 0..n-1]
    mov %rax, %rcx
    imul %r11, %rcx     # rcx = i * n
    lea (%r15, %rcx, 4), %rdi  # rdi = &C[i, 0]
    
    vxorps %ymm0, %ymm0, %ymm0
    xor %rbx, %rbx      # j = 0
.L_ikj_zero_loop:
    mov %r11, %rcx
    sub %rbx, %rcx      # remaining = n - j
    cmp $8, %rcx
    jl .L_ikj_zero_scalar
    vmovups %ymm0, (%rdi, %rbx, 4)
    add $8, %rbx
    jmp .L_ikj_zero_loop
.L_ikj_zero_scalar:
    cmp %r11, %rbx
    jge .L_ikj_zero_done
    vmovss %xmm0, (%rdi, %rbx, 4)
    inc %rbx
    jmp .L_ikj_zero_scalar
.L_ikj_zero_done:

    # Loop p = 0..k-1
    xor %rbx, %rbx      # p = 0
.L_ikj_p_loop:
    cmp %r12, %rbx
    jge .L_ikj_p_done

    # a_val = A[i, p]
    mov %rax, %rcx
    imul %r12, %rcx
    add %rbx, %rcx      # i * k + p
    vbroadcastss (%r13, %rcx, 4), %ymm0  # ymm0 = a_val

    # b_row = &B[p, 0]
    mov %rbx, %rcx
    imul %r11, %rcx
    lea (%r14, %rcx, 4), %rsi   # rsi = &B[p, 0]
    
    # c_row = &C[i, 0]
    mov %rax, %rcx
    imul %r11, %rcx
    lea (%r15, %rcx, 4), %rdi   # rdi = &C[i, 0]

    xor %rcx, %rcx      # j = 0
.L_ikj_j_loop:
    mov %r11, %r8
    sub %rcx, %r8       # remaining = n - j
    cmp $16, %r8
    jl .L_ikj_j_leftover

    # Process 16 elements
    vmovups (%rdi, %rcx, 4), %ymm1
    vmovups (%rsi, %rcx, 4), %ymm2
    vfmadd231ps %ymm0, %ymm2, %ymm1
    vmovups %ymm1, (%rdi, %rcx, 4)

    vmovups 32(%rdi, %rcx, 4), %ymm1
    vmovups 32(%rsi, %rcx, 4), %ymm2
    vfmadd231ps %ymm0, %ymm2, %ymm1
    vmovups %ymm1, 32(%rdi, %rcx, 4)

    add $16, %rcx
    jmp .L_ikj_j_loop

.L_ikj_j_leftover:
    cmp $8, %r8
    jl .L_ikj_j_scalar

    # Process 8 elements
    vmovups (%rdi, %rcx, 4), %ymm1
    vmovups (%rsi, %rcx, 4), %ymm2
    vfmadd231ps %ymm0, %ymm2, %ymm1
    vmovups %ymm1, (%rdi, %rcx, 4)

    add $8, %rcx

.L_ikj_j_scalar:
    cmp %r11, %rcx
    jge .L_ikj_j_done
    
    vmovss (%rdi, %rcx, 4), %xmm1
    vmovss (%rsi, %rcx, 4), %xmm2
    vfmadd231ps %xmm0, %xmm2, %xmm1
    vmovss %xmm1, (%rdi, %rcx, 4)
    
    inc %rcx
    jmp .L_ikj_j_scalar

.L_ikj_j_done:
    inc %rbx
    jmp .L_ikj_p_loop

.L_ikj_p_done:
    inc %rax
    jmp .L_ikj_i_loop

.L_ikj_done:
    vzeroupper
    pop %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret
