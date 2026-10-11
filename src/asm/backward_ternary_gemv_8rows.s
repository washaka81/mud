.section .text
.global backward_ternary_gemv_8rows_dx_avx2

# Backward pass for Ternary ELUT 4-bit GEMV — 8-row batched grad_x accumulation (STE).
#
# Forward GEMV (8-row): y[r] = scale[r] * sum_j( w[r,j] * x[j] ) for r=0..7
#
# Backward grad_x (8-row batched):
#   dx[j] += sum_r( scale_gy[r] * dequant(w[r, j]) )
#   where scale_gy[r] = scale[r] * grad_y[r]
#
# Processes 8 rows simultaneously: for each 8-element word position,
# loads grad_x[0:8] ONCE, accumulates contributions from all 8 rows,
# stores once. This reduces grad_x memory traffic by 8× vs single-row kernel.
#
# Weight layout: row r's word w is at weights[r * stride + w]
#   (row-major, stride = n_in / 8 u32 words per row)
#
# ABI (System V AMD64):
#   rdi = n           (elements per row; MUST be multiple of 8 for vector path)
#   rsi = scale_gy    (*const f32, 8 scale×grad_y values, contiguous)
#   rdx = weights     (*const u32, row0 base; rows spaced by r8 u32 words)
#   rcx = grad_x      (*mut f32, n elements, accumulate across all 8 rows)
#   r8  = stride      (u32 word stride between rows = n/8)
#
# Register allocation:
#   r12 = n (element count, preserved)
#   r13 = scale_gy ptr (preserved)
#   r14 = grad_x ptr (advances through elements in loop)
#   r15 = weights row0 base (preserved)
#   rbx = stride in bytes (stride_u32 * 4, preserved)
#   r10 = word count (n / 8, countdown)
#   r11 = word byte offset within row (advances in loop)
#   (r11d used for byte offset within row — 0, 4, 8, ... for each u32 word)
#   Stack frame: 64 bytes for 8 row base pointers (pushed after callee-saved saves)
#
#   ymm0  = grad_x accumulator [c..c+8]
#   ymm1  = scale_gy broadcast for current row
#   ymm2  = weight word broadcast
#   ymm3  = shifted nibbles
#   ymm4  = masked 4-bit values
#   ymm5  = plus_mask
#   ymm6  = minus_mask
#   ymm7  = plus_val
#   ymm8  = minus_val
#   ymm9  = net contribution
#   ymm10 = SHIFTS [0,4,8,12,16,20,24,28]
#   ymm11 = MASK [0x0F × 8]
#   ymm12 = VAL_ONE [1 × 8]
#   ymm13 = exp_mask (NaN/Inf guard, L-08)
#   ymm14 = VAL_MINUS_ONE [0xF × 8]
#   rax/rcx/rdx/rsi/r8/r9 = scratch (row pointer computation, load/store)
backward_ternary_gemv_8rows_dx_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    push %r10
    push %r11
    # Allocate 64 bytes for 8 row base pointers
    sub $64, %rsp

    mov %rdi, %r12                    # r12 = n
    mov %rsi, %r13                    # r13 = scale_gy ptr
    mov %rdx, %r15                    # r15 = weights row0 base
    mov %rcx, %r14                    # r14 = grad_x ptr
    mov %r8, %rbx                     # rbx = stride (u32 words)
    shl $2, %rbx                      # rbx = stride in bytes

    # Word count = n / 8
    mov %r12, %rax
    shr $3, %rax
    mov %rax, %r10                    # r10 = word count
    test %rax, %rax
    jz .done                          # n < 8: nothing to do

    # Precompute all 8 row base pointers on stack.
    # Row r base = r15 + r * rbx
    # Valid scale factors in x86-64: 1, 2, 4, 8 — use combinations of lea + add.
    mov %r15, 0(%rsp)                 # row0 = r15
    lea (%r15, %rbx), %rax
    mov %rax, 8(%rsp)                 # row1 = r15 + rbx
    lea (%r15, %rbx, 2), %rax
    mov %rax, 16(%rsp)                # row2 = r15 + 2*rbx
    lea (%r15, %rbx, 2), %rax
    add %rbx, %rax
    mov %rax, 24(%rsp)                # row3 = r15 + 3*rbx
    lea (%r15, %rbx, 4), %rax
    mov %rax, 32(%rsp)                # row4 = r15 + 4*rbx
    lea (%r15, %rbx, 4), %rax
    add %rbx, %rax
    mov %rax, 40(%rsp)                # row5 = r15 + 5*rbx
    lea (%r15, %rbx, 4), %rax
    lea (%rax, %rbx, 2), %rax
    mov %rax, 48(%rsp)                # row6 = r15 + 6*rbx
    lea (%r15, %rbx, 4), %rax
    lea (%rax, %rbx, 2), %rax
    add %rbx, %rax
    mov %rax, 56(%rsp)                # row7 = r15 + 7*rbx

    # Load shared constants
    vmovdqu btg8_shifts(%rip), %ymm10
    vmovdqu btg8_mask(%rip), %ymm11
    vmovdqu btg8_val_one(%rip), %ymm12
    vmovdqu btg8_val_minus(%rip), %ymm14
    vmovdqu btg8_exp_mask(%rip), %ymm13

    xor %r11, %r11                  # r11 = byte offset within row (0)

.align 32
.word_loop:
    # Load grad_x[0:8] into accumulator
    vmovups (%r14), %ymm0

.row_loop:
    # Unrolled 8-row processing — each row uses same weight extraction pattern
    # Row 0: addr = row0_base + r11
    vmovss (%r13), %xmm1               # scale_gy[0]
    vbroadcastss %xmm1, %ymm1
    mov 0(%rsp), %rax                  # rax = row0 base
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 1
    vmovss 4(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 8(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 2
    vmovss 8(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 16(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 3
    vmovss 12(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 24(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 4
    vmovss 16(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 32(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 5
    vmovss 20(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 40(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 6
    vmovss 24(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 48(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # Row 7
    vmovss 28(%r13), %xmm1
    vbroadcastss %xmm1, %ymm1
    mov 56(%rsp), %rax
    vpbroadcastd (%rax, %r11), %ymm2
    vpsrlvd %ymm10, %ymm2, %ymm3
    vpand %ymm11, %ymm3, %ymm4
    vpcmpeqd %ymm12, %ymm4, %ymm5
    vpcmpeqd %ymm14, %ymm4, %ymm6
    vpand %ymm1, %ymm5, %ymm7
    vpand %ymm1, %ymm6, %ymm8
    vsubps %ymm8, %ymm7, %ymm9
    vaddps %ymm9, %ymm0, %ymm0

    # --- NaN/Inf guard (L-08) ---
    vpand %ymm13, %ymm0, %ymm9
    vpcmpeqd %ymm13, %ymm9, %ymm9
    vpandn %ymm0, %ymm9, %ymm0

    # --- Store accumulated grad_x ---
    vmovups %ymm0, (%r14)

    add $32, %r14                      # grad_x += 8 f32
    add $4, %r11                        # word offset += 1 u32 word (4 bytes)
    dec %r10
    jnz .word_loop

.done:
    vzeroupper
    add $64, %rsp
    pop %r11
    pop %r10
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    pop %rbp
    ret

# Note: n MUST be a multiple of 8. All production ternary weight dimensions
# are powers of 2 (n_in = 128, 512, 2048, ...), so n%8 == 0 always holds.
# For n%8 != 0, callers should fall back to backward_ternary_gemv_dx_avx2
# for the remainder rows/elements.

# Constants (32-byte aligned for 256-bit loads)
.align 32
btg8_shifts:     .long 0, 4, 8, 12, 16, 20, 24, 28
btg8_mask:       .long 15, 15, 15, 15, 15, 15, 15, 15
btg8_val_one:    .long 1, 1, 1, 1, 1, 1, 1, 1
btg8_val_minus:  .long 15, 15, 15, 15, 15, 15, 15, 15
btg8_exp_mask:   .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
