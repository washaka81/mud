.section .text
.global ternary_gemm_batch4_avx2

# Optimized Ternary GEMM for Batch=4 (Speculative Decoding Target Verification)
# Multiplies W [out_dim, in_dim] with X [4, in_dim] to produce OUT [4, out_dim].
# W is packed 16 ternary values per u32.

# Fixes applied:
#   1. Token 2/3 x-pointer swap fixed (use indexed addressing instead of stack slots)
#   2. vhaddps reduction replaced with shuffle pattern (lower latency)
#   3. Added prefetcht0 for next iteration's data

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

    # Reset x pointers for this row
    # rax = Token 0 advancing, r10 = Token 1 advancing
    mov %rdx, %rax
    mov %r11, %r10
    # rbp = column float-index (0, 16, 32, ...) for T2/T3 indexed addressing
    xor %rbp, %rbp

.col_loop:
    cmp %rsi, %rbp      # if current_float_index >= in_dim, done
    jae .col_done

    # Prefetch next iteration's activations
    prefetcht0 64(%rax)
    prefetcht0 64(%r10)

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

    # Token 2 (low 8) — using r12 (T2 base) + rbp * 4 as index
    vmovups (%r12, %rbp, 4), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm10, %ymm10
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm10, %ymm10

    # Token 3 (low 8) — using r13 (T3 base) + rbp * 4 as index
    vmovups (%r13, %rbp, 4), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm11, %ymm11
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm11, %ymm11

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

    # Token 2 (high 8) — at rbp + 8 floats = (rbp + 8) * 4 = rbp*4 + 32
    vmovups 32(%r12, %rbp, 4), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm10, %ymm10
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm10, %ymm10

    # Token 3 (high 8)
    vmovups 32(%r13, %rbp, 4), %ymm2
    vpand %ymm0, %ymm2, %ymm3
    vaddps %ymm3, %ymm11, %ymm11
    vpand %ymm1, %ymm2, %ymm3
    vsubps %ymm3, %ymm11, %ymm11

    add $16, %rbp        # Advance float index by 16 (one W chunk)
    jmp .col_loop

.col_done:
    # r12/r13 preserved (never modified in loop) — no push/pop needed

    # Horizontal reduction using shuffle pattern (lower latency than vhaddps)
    vextractf128 $1, %ymm8, %xmm0
    vaddps %xmm0, %xmm8, %xmm8
    vshufps $0xEE, %xmm8, %xmm8, %xmm0
    vaddps %xmm0, %xmm8, %xmm8
    vshufps $0x11, %xmm8, %xmm8, %xmm0
    vaddps %xmm0, %xmm8, %xmm8

    vextractf128 $1, %ymm9, %xmm0
    vaddps %xmm0, %xmm9, %xmm9
    vshufps $0xEE, %xmm9, %xmm9, %xmm0
    vaddps %xmm0, %xmm9, %xmm9
    vshufps $0x11, %xmm9, %xmm9, %xmm0
    vaddps %xmm0, %xmm9, %xmm9

    vextractf128 $1, %ymm10, %xmm0
    vaddps %xmm0, %xmm10, %xmm10
    vshufps $0xEE, %xmm10, %xmm10, %xmm0
    vaddps %xmm0, %xmm10, %xmm10
    vshufps $0x11, %xmm10, %xmm10, %xmm0
    vaddps %xmm0, %xmm10, %xmm10

    vextractf128 $1, %ymm11, %xmm0
    vaddps %xmm0, %xmm11, %xmm11
    vshufps $0xEE, %xmm11, %xmm11, %xmm0
    vaddps %xmm0, %xmm11, %xmm11
    vshufps $0x11, %xmm11, %xmm11, %xmm0
    vaddps %xmm0, %xmm11, %xmm11

    # Load scale for this row
    vbroadcastss (%r9, %rbx, 4), %xmm0

    # Multiply by scale
    vmulss %xmm0, %xmm8, %xmm8
    vmulss %xmm0, %xmm9, %xmm9
    vmulss %xmm0, %xmm10, %xmm10
    vmulss %xmm0, %xmm11, %xmm11

    # Store token 0
    vmovss %xmm8, (%r8, %rbx, 4)
    # Store token 1
    vmovss %xmm9, (%r15, %rbx, 4)

    # Calculate Token 2 & 3 out offsets using rax (T0 pointer no longer needed)
    mov %r14, %rax       # rax = out_dim * 4
    add %r14, %rax       # rax = out_dim * 8
    add %r8, %rax        # rax = out_ptr + out_dim * 8
    vmovss %xmm10, (%rax, %rbx, 4)

    add %r14, %rax       # rax = out_ptr + out_dim * 12
    vmovss %xmm11, (%rax, %rbx, 4)

    # Next row
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
