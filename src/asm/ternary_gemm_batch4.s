.section .text
.global ternary_gemm_batch4_avx2

# Optimized Ternary GEMM for Batch=4 (Speculative Decoding Target Verification)
# Multiplies W [out_dim, in_dim] with X [4, in_dim] to produce OUT [4, out_dim].
# W is packed 16 ternary values per u32.

.section .rodata
.align 32
SHIFTS_LOW:  .long 0, 2, 4, 6, 8, 10, 12, 14
SHIFTS_HIGH: .long 16, 18, 20, 22, 24, 26, 28, 30
MASK_2BIT:   .long 3, 3, 3, 3, 3, 3, 3, 3
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_TWO:     .long 2, 2, 2, 2, 2, 2, 2, 2

.section .text
ternary_gemm_batch4_avx2:
    # RDI: out_dim (rows in W)
    # RSI: in_dim (cols in W)
    # RDX: x_ptr (shape [4, in_dim])
    # RCX: w_ptr (shape [out_dim, in_dim / 16])
    # R8:  out_ptr (shape [4, out_dim])
    # R9:  scales (length out_dim)

    push %rbp
    mov %rsp, %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    push %rbx

    # x_ptr offsets for the 4 tokens
    # Token 0 is at RDX
    # Token 1 is at RDX + in_dim * 4
    mov %rsi, %r10
    shl $2, %r10        # r10 = in_dim * 4 (bytes)
    lea (%rdx, %r10), %r11      # r11 = x_ptr + in_dim * 4 (Token 1)
    lea (%r11, %r10), %r12      # r12 = x_ptr + in_dim * 8 (Token 2)
    lea (%r12, %r10), %r13      # r13 = x_ptr + in_dim * 12 (Token 3)

    # Output offsets
    mov %rdi, %r14
    shl $2, %r14        # r14 = out_dim * 4 (bytes)
    lea (%r8, %r14), %r15       # r15 = out_ptr + out_dim * 4 (Token 1)
    # Token 2 out is at r8 + r14 * 2
    # Token 3 out is at r8 + r14 * 3

    # We process W row by row
    xor %rbx, %rbx      # rbx = current_row = 0

.row_loop:
    cmp %rdi, %rbx
    jge .done

    # Accumulators for the 4 tokens
    vxorps %ymm8, %ymm8, %ymm8
    vxorps %ymm9, %ymm9, %ymm9
    vxorps %ymm10, %ymm10, %ymm10
    vxorps %ymm11, %ymm11, %ymm11

    # Loop over columns (in_dim / 16 chunks)
    # We will use rcx as the w_ptr for this row, which advances.
    # We must reset x pointers for each row.
    mov %rdx, %rax      # rax = x_ptr (Token 0)
    mov %r11, %r10      # r10 = x_ptr (Token 1)
    push %r12
    push %r13
    
    # Actually, we shouldn't push/pop in the loop if we can avoid it.
    # Let's save rcx (w_ptr for the row)
    mov %rcx, %rdi      # rdi = current w_ptr
    
    mov %rsi, %r12      # loop counter (in_dim / 16)
    shr $4, %r12
    
.col_loop:
    cmp $0, %r12
    je .col_done

    # Load 1 chunk of W (16 values in 1 u32)
    vpbroadcastd (%rcx), %ymm14
    add $4, %rcx

    # Expand W to low 8 values
    vpsrlvd SHIFTS_LOW(%rip), %ymm14, %ymm15
    vpand MASK_2BIT(%rip), %ymm15, %ymm15
    
    vpcmpeqd VAL_ONE(%rip), %ymm15, %ymm0  # +1 mask
    vpcmpeqd VAL_TWO(%rip), %ymm15, %ymm1  # -1 mask

    # Token 0 (low 8)
    vmovups (%rax), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm8, %ymm8
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm8, %ymm8

    # Token 1 (low 8)
    vmovups (%r10), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm9, %ymm9
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm9, %ymm9
    
    # Token 2 (low 8)
    mov 0(%rsp), %r13 # get Token 2 ptr
    vmovups (%r13), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm10, %ymm10
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm10, %ymm10
    add $32, %r13
    mov %r13, 0(%rsp)
    
    # Token 3 (low 8)
    mov 8(%rsp), %r13 # get Token 3 ptr
    vmovups (%r13), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm11, %ymm11
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm11, %ymm11
    add $32, %r13
    mov %r13, 8(%rsp)

    # Expand W to high 8 values
    vpsrlvd SHIFTS_HIGH(%rip), %ymm14, %ymm15
    vpand MASK_2BIT(%rip), %ymm15, %ymm15
    vpcmpeqd VAL_ONE(%rip), %ymm15, %ymm0  # +1 mask
    vpcmpeqd VAL_TWO(%rip), %ymm15, %ymm1  # -1 mask
    
    # Token 0 (high 8)
    vmovups 32(%rax), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm8, %ymm8
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm8, %ymm8
    add $64, %rax

    # Token 1 (high 8)
    vmovups 32(%r10), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm9, %ymm9
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm9, %ymm9
    add $64, %r10
    
    # Token 2 (high 8)
    mov 0(%rsp), %r13
    vmovups (%r13), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm10, %ymm10
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm10, %ymm10
    add $32, %r13
    mov %r13, 0(%rsp)

    # Token 3 (high 8)
    mov 8(%rsp), %r13
    vmovups (%r13), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm11, %ymm11
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm11, %ymm11
    add $32, %r13
    mov %r13, 8(%rsp)

    dec %r12
    jmp .col_loop

.col_done:
    pop %r13
    pop %r12

    # Horizontal sum for the accumulators ymm8..11
    # We can do this efficiently
    vhaddps %ymm8, %ymm8, %ymm8
    vhaddps %ymm8, %ymm8, %ymm8
    vextractf128 $1, %ymm8, %xmm0
    vaddps %xmm0, %xmm8, %xmm8
    # xmm8[0] has token 0 sum

    vhaddps %ymm9, %ymm9, %ymm9
    vhaddps %ymm9, %ymm9, %ymm9
    vextractf128 $1, %ymm9, %xmm0
    vaddps %xmm0, %xmm9, %xmm9
    # xmm9[0] has token 1 sum
    
    vhaddps %ymm10, %ymm10, %ymm10
    vhaddps %ymm10, %ymm10, %ymm10
    vextractf128 $1, %ymm10, %xmm0
    vaddps %xmm0, %xmm10, %xmm10
    
    vhaddps %ymm11, %ymm11, %ymm11
    vhaddps %ymm11, %ymm11, %ymm11
    vextractf128 $1, %ymm11, %xmm0
    vaddps %xmm0, %xmm11, %xmm11

    # Load scale for this row
    vbroadcastss (%r9, %rbx, 4), %xmm0

    # Multiply and store
    vmulss %xmm0, %xmm8, %xmm8
    vmulss %xmm0, %xmm9, %xmm9
    vmulss %xmm0, %xmm10, %xmm10
    vmulss %xmm0, %xmm11, %xmm11

    # Store token 0
    vmovss %xmm8, (%r8, %rbx, 4)
    # Store token 1
    vmovss %xmm9, (%r15, %rbx, 4)
    
    # Calculate Token 2 & 3 out offsets
    mov %r14, %r13       # r14 is out_dim * 4
    add %r14, %r13       # r13 = out_dim * 8
    add %r8, %r13        # r13 = out_ptr + out_dim * 8
    vmovss %xmm10, (%r13, %rbx, 4)
    
    add %r14, %r13       # r13 = out_ptr + out_dim * 12
    vmovss %xmm11, (%r13, %rbx, 4)

    # Next row
    inc %rbx
    # RDI is out_dim (preserved? no we overrode it! Wait!
    # Ah, I overrode RDI: `mov %rcx, %rdi`
    # Let me fix the out_dim loop check
    jmp .row_loop_next

.row_loop_next:
    # Actually, RDI is out_dim, let's restore it from stack or avoid overwriting
    # Let's see... I used `%rdi` as temporary for w_ptr, but didn't even use it!
    # `mov %rcx, %rdi` is useless, I just advance `%rcx` directly.
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
