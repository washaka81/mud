.text
.globl mamba_scan_avx2

# void mamba_scan_avx2(size_t n, size_t d_state, const float* x, const float* a_bar, const float* b_bar, const float* c, const float* dt, float* state, float* out)
# rdi: n (hidden_size)
# rsi: d_state (usually 16)
# rdx: x
# rcx: a_bar (precomputed exp(dt*A))
# r8:  b_bar (precomputed dt*B)
# r9:  c
# stack: dt (16(%rbp)), state (24(%rbp)), out (32(%rbp))

mamba_scan_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    # Arguments from stack
    mov 24(%rbp), %r10    # state
    mov 32(%rbp), %r11    # out

    xor %r12, %r12        # i = 0 (loop over n)
.loop_n:
    # Check if we have at least 2 channels left for MIMO
    mov %r12, %rax
    add $2, %rax
    cmp %rdi, %rax
    jg .loop_n_scalar

    # --- MIMO 2x Block ---
    vbroadcastss (%rdx, %r12, 4), %ymm0       # ymm0 = x[i]
    vbroadcastss 4(%rdx, %r12, 4), %ymm6      # ymm6 = x[i+1]
    
    mov %r12, %rax
    imul %rsi, %rax
    shl $2, %rax                              # rax = i * d_state * 4
    
    lea (%r10, %rax), %r14                    # state_0
    lea (%rcx, %rax), %r15                    # a_bar_0
    lea (%r8, %rax), %rbx                     # b_bar_0
    
    mov %rsi, %rax
    shl $2, %rax                              # stride = d_state * 4
    
    # We will use offset addressing for the second channel: (ptr, %rax)
    vxorps %ymm5, %ymm5, %ymm5                # ymm5 = partial_sum_0
    vxorps %ymm11, %ymm11, %ymm11             # ymm11 = partial_sum_1
    
    xor %r13, %r13                            # j = 0
.loop_d_mimo:
    cmp %rsi, %r13
    jge .next_n_mimo

    vmovups (%r14, %r13, 4), %ymm1            # state_0
    vmovups (%r15, %r13, 4), %ymm2            # a_bar_0
    vmulps %ymm2, %ymm1, %ymm1                # h_0 * a_0
    vmovups (%rbx, %r13, 4), %ymm3            # b_bar_0
    vfmadd231ps %ymm0, %ymm3, %ymm1           # h_0 += x_0 * b_0
    vmovups %ymm1, (%r14, %r13, 4)            # store state_0
    
    # Channel i+1
    # We will compute the second channel using offset addressing directly to avoid register exhaustion.
    # relying on the CPU's Out-Of-Order execution to hide latency.
    
    vmovups (%r14, %r13, 4), %ymm1            # state_0
    vmovups (%r15, %r13, 4), %ymm2            # a_bar_0
    vmulps %ymm2, %ymm1, %ymm1
    vmovups (%rbx, %r13, 4), %ymm3
    vfmadd231ps %ymm0, %ymm3, %ymm1
    vmovups %ymm1, (%r14, %r13, 4)
    vmovups (%r9, %r13, 4), %ymm4             # c
    vfmadd231ps %ymm1, %ymm4, %ymm5           # sum_0
    
    # Reload original n and out pointers just in case, but they are in rdi and r11.
    push %r14
    push %r15
    push %rbx
    add %rax, %r14
    add %rax, %r15
    add %rax, %rbx
    
    vmovups (%r14, %r13, 4), %ymm7            # state_1
    vmovups (%r15, %r13, 4), %ymm8            # a_bar_1
    vmulps %ymm8, %ymm7, %ymm7
    vmovups (%rbx, %r13, 4), %ymm9
    vfmadd231ps %ymm6, %ymm9, %ymm7
    vmovups %ymm7, (%r14, %r13, 4)
    vfmadd231ps %ymm7, %ymm4, %ymm11          # sum_1 (using same c in ymm4)
    
    pop %rbx
    pop %r15
    pop %r14
    
    add $8, %r13
    jmp .loop_d_mimo

.next_n_mimo:
    # Horizontal sum ymm5
    vextractf128 $1, %ymm5, %xmm1
    vaddps %xmm1, %xmm5, %xmm5
    vhaddps %xmm5, %xmm5, %xmm5
    vhaddps %xmm5, %xmm5, %xmm5
    vmovss %xmm5, (%r11, %r12, 4)
    
    # Horizontal sum ymm11
    vextractf128 $1, %ymm11, %xmm1
    vaddps %xmm1, %xmm11, %xmm11
    vhaddps %xmm11, %xmm11, %xmm11
    vhaddps %xmm11, %xmm11, %xmm11
    vmovss %xmm11, 4(%r11, %r12, 4)
    
    add $2, %r12
    jmp .loop_n

.loop_n_scalar:
    cmp %rdi, %r12
    jge .done

    vbroadcastss (%rdx, %r12, 4), %ymm0
    mov %r12, %rax
    imul %rsi, %rax
    shl $2, %rax
    
    lea (%r10, %rax), %r14
    lea (%rcx, %rax), %r15
    lea (%r8, %rax), %rbx

    vxorps %ymm5, %ymm5, %ymm5
    
    xor %r13, %r13
.loop_d_scalar:
    cmp %rsi, %r13
    jge .next_n_scalar

    vmovups (%r14, %r13, 4), %ymm1
    vmovups (%r15, %r13, 4), %ymm2
    vmulps %ymm2, %ymm1, %ymm1
    
    vmovups (%rbx, %r13, 4), %ymm3
    vfmadd231ps %ymm0, %ymm3, %ymm1
    
    vmovups %ymm1, (%r14, %r13, 4)
    
    vmovups (%r9, %r13, 4), %ymm4
    vfmadd231ps %ymm1, %ymm4, %ymm5
    
    add $8, %r13
    jmp .loop_d_scalar

.next_n_scalar:
    vextractf128 $1, %ymm5, %xmm6
    vaddps %xmm6, %xmm5, %xmm5
    vhaddps %xmm5, %xmm5, %xmm5
    vhaddps %xmm5, %xmm5, %xmm5
    vmovss %xmm5, (%r11, %r12, 4)
    
    inc %r12
    jmp .loop_n

.done:
    vzeroupper
    pop %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret

.globl mamba_delta_fold_avx2

# void mamba_delta_fold_avx2(size_t len, float* state, float decay)
# rdi: len (must be multiple of 8)
# rsi: state
# xmm0: decay

mamba_delta_fold_avx2:
    vbroadcastss %xmm0, %ymm0      # ymm0 = [decay, decay, ..., decay]
    xor %rcx, %rcx                 # i = 0
.fold_loop:
    cmp %rdi, %rcx
    jge .fold_done
    
    vmulps (%rsi, %rcx, 4), %ymm0, %ymm1   # ymm1 = state[i..i+7] * decay
    vmovups %ymm1, (%rsi, %rcx, 4)         # state[i..i+7] = ymm1
    
    add $8, %rcx
    jmp .fold_loop
.fold_done:
    vzeroupper
    ret
