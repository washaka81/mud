.section .text
.global lm_head_avx2

# lm_head_avx2(vocab_size: usize, hidden: usize, regs: *const f32, weights: *const f32) -> usize
# Returns the vocabulary index with the highest logit (dot product with registers)
#
# rdi = vocab_size
# rsi = hidden (must be multiple of 16)
# rdx = regs (pointer to hidden f32 values)
# rcx = weights (pointer to vocab_size * hidden f32 values)
#
# Strategy: For each vocab row, compute dot product with regs using AVX2+FMA,
# track the maximum logit and its index.

lm_head_avx2:
    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    
    # Save parameters
    mov %rdi, %r12        # r12 = vocab_size
    mov %rsi, %r13        # r13 = hidden
    mov %rdx, %r14        # r14 = regs
    mov %rcx, %r15        # r15 = weights
    
    # Initialize max_logit = -INF in xmm1, best_id = 0
    vmovss .neg_inf(%rip), %xmm1
    xor %r8, %r8          # r8 = best_id = 0
    
    xor %r9, %r9          # r9 = vocab index (v)
    
.lm_vocab_loop:
    cmp %r12, %r9
    jge .lm_done
    
    # Compute dot product for this vocab row
    # regs is in r14, current weight row is at r15 + v * hidden * 4
    mov %r9, %rax
    imul %r13, %rax       # v * hidden
    shl $2, %rax          # * 4 (bytes per f32)
    lea (%r15, %rax), %rcx  # rcx = &weights[v * hidden]
    
    # Save xmm1 (max_logit) - dot_product_avx2 uses ymm1 which corrupts xmm1
    # Align stack to 16 bytes before call (we're at 8-byte alignment after function entry)
    sub $24, %rsp         # 24 bytes: 16 for xmm1 + 8 for alignment
    vmovaps %xmm1, 8(%rsp)
    
    # dot_product_avx2(hidden, regs, weight_row)
    mov %r13, %rdi        # n = hidden
    mov %r14, %rsi        # a = regs
    mov %rcx, %rdx        # b = weight_row
    call dot_product_avx2
    
    # Restore xmm1 (max_logit)
    vmovaps 8(%rsp), %xmm1
    add $24, %rsp
    
    # xmm0 now contains the logit for this vocab row
    # Compare with current max (in xmm1)
    vcomiss %xmm0, %xmm1  # compare max_logit with current logit
    jae .lm_skip_update   # if max_logit >= logit, skip
    
    # Update max_logit and best_id
    vmovaps %xmm0, %xmm1   # max_logit = logit
    mov %r9, %r8          # best_id = v
    
.lm_skip_update:
    inc %r9
    jmp .lm_vocab_loop
    
.lm_done:
    # Return best_id in rax
    mov %r8, %rax
    
    vzeroupper
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    ret

.section .rodata
.align 16
.neg_inf:
    .long 0xFF800000  # -INF in IEEE 754
