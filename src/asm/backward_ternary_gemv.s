.section .text
.global backward_ternary_gemv_dx_avx2

# Backward pass for Ternary ELUT 4-bit GEMV — grad_x accumulation (STE).
#
# Forward GEMV:  y[i] = scale[i] * sum_j( w[i,j] * x[j] )
#               where w[i,j] ∈ {-1, 0, +1} (ELUT 4-bit nibble: 0x1=+1, 0xF=-1, else=0)
#
# Backward grad_x:  dx[j] += scale * grad_y * dequant(w[j])
# (transposed GEMV: each upstream grad_y * scale flows back through each ternary weight)
#
# Uses the same nibble-extraction technique as ternary_gemv_avx2 forward:
#   vpbroadcastd → nibble broadcast to all 8 lanes
#   vpsrlvd + vpand → extract 4-bit value per lane
#   vpcmpeqd → generate ±1 masks (VAL_ONE=0x1, VAL_MINUS_ONE=0xF per lane)
#   vpand + vpsub → accumulate signed contributions into grad_x
#
# ABI (System V AMD64):
#   rdi = n           (element count; caller handles n%8 remainder or
#                      ensures n is a multiple of 8)
#   xmm0 = scale_gy   (f32 scalar = scale * grad_y for this row)
#   rsi = weights     (*const u32, n/8 packed words; 8 nibbles per word)
#   rdx = grad_x      (*mut f32, accumulate scale_gy * dequant(w[j]) into existing values)
#
# Register allocation:
#   r12 = word count (n / 8) for vector loop
#   r13 = weights ptr
#   r14 = grad_x ptr
#   r15 = scalar tail element count (n % 8)
#   rbx = nibble index within current u32 word (scalar tail)
#   ymm0 = scale_gy (broadcast to 8 lanes)
#   ymm1 = weights word (broadcast u32 → 8× same)
#   ymm2 = shifted nibbles
#   ymm3 = masked 4-bit values
#   ymm4 = plus_mask   (nibble == 0x1 → all-ones)
#   ymm5 = minus_mask  (nibble == 0xF → all-ones)
#   ymm6 = grad_x[0:8] (loaded)
#   ymm7 = plus_val = plus_mask & scale_gy
#   ymm8 = minus_val = minus_mask & scale_gy
#   ymm9 = net = plus_val - minus_val
#   ymm10 = SHIFTS_ELUT (0,4,8,...,28)
#   ymm11 = MASK_ELUT (0x0F per lane)
#   ymm12 = VAL_ONE (1 per lane)
#   ymm15 = VAL_MINUS_ONE (0xF per lane)
#   ymm13 = exp_mask for NaN/Inf guard (L-08)
#   xmm0 preserved as scale_gy throughout
backward_ternary_gemv_dx_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    sub $8, %rsp

    mov %rdi, %r12                    # r12 = n (preserve original)
    mov %rsi, %r13                    # r13 = weights ptr
    mov %rdx, %r14                    # r14 = grad_x ptr

    # Word count = n / 8
    mov %r12, %rax
    shr $3, %rax                      # rax = n / 8
    mov %rax, %r12                    # r12 = word count
    test %rax, %rax
    jz .scalar_tail                   # no full words → all scalar

    # Broadcast scale_gy to all 8 lanes
    vbroadcastss %xmm0, %ymm0          # ymm0 = scale_gy[0:8]

    # Load shared constants
    vmovdqu btg_shifts(%rip), %ymm10   # ymm10 = [0,4,8,12,16,20,24,28]
    vmovdqu btg_mask(%rip), %ymm11     # ymm11 = [0x0F × 8]
    vmovdqu btg_val_one(%rip), %ymm12  # ymm12 = [1 × 8]
    vmovdqu btg_val_minus(%rip), %ymm15 # ymm15 = [0xF × 8]
    vmovdqu btg_exp_mask(%rip), %ymm13  # ymm13 = [0x7F800000 × 8]

.align 32
.vector_loop:
    # --- Load 1 weight word, broadcast to 8 lanes ---
    vpbroadcastd (%r13), %ymm1         # ymm1 = [w,w,w,w,w,w,w,w]
    vmovups (%r14), %ymm6              # ymm6 = grad_x[0:8]

    # --- Extract nibbles: shift each lane by its position, then mask ---
    vpsrlvd %ymm10, %ymm1, %ymm2       # ymm2 = shifted nibbles
    vpand %ymm11, %ymm2, %ymm3          # ymm3 = 4-bit nibble values [0..15]

    # --- Generate ±1 masks ---
    vpcmpeqd %ymm12, %ymm3, %ymm4      # ymm4 = plus_mask (all-ones where nibble==1)
    vpcmpeqd %ymm15, %ymm3, %ymm5       # ymm5 = minus_mask (all-ones where nibble==15)

    # --- Compute signed contributions ---
    vpand %ymm0, %ymm4, %ymm7          # ymm7 = scale_gy where w=+1, 0 elsewhere
    vpand %ymm0, %ymm5, %ymm8          # ymm8 = scale_gy where w=-1, 0 elsewhere
    vsubps %ymm8, %ymm7, %ymm9         # ymm9 = plus - minus = signed contribution

    # --- Accumulate into grad_x ---
    vaddps %ymm6, %ymm9, %ymm6         # grad_x += signed_contribution

    # --- NaN/Inf guard (L-08): zero non-finite lanes ---
    vpand %ymm13, %ymm6, %ymm9
    vpcmpeqd %ymm13, %ymm9, %ymm9
    vpandn %ymm6, %ymm9, %ymm6

    vmovups %ymm6, (%r14)              # store updated grad_x

    add $4, %r13                       # weights += 1 u32 word (4 bytes)
    add $32, %r14                      # grad_x += 8 f32 (32 bytes)
    dec %r12
    jnz .vector_loop

.scalar_tail:
    # --- Scalar tail for remaining (n % 8) elements ---
    mov %rdi, %r15                      # r15 = original n
    and $7, %r15                        # r15 = n % 8
    test %r15, %r15
    jz .done

    xor %rbx, %rbx                    # rbx = nibble index = 0

.scalar_loop:
    # Extract nibble rbx from word at *r13
    # eax = u32 word
    mov (%r13), %eax
    # shift amount = rbx * 4
    mov %ebx, %ecx
    shl $2, %cl                          # cl = rbx * 4
    shr %cl, %eax                      # shift word right by nibble_idx * 4
    and $0xF, %eax                      # eax = nibble value [0..15]

    # Ternary decode: 1→+scale_gy, 15→-scale_gy, else→0
    # Start with 0.0
    vxorps %xmm6, %xmm6, %xmm6
    cmp $1, %eax
    je .s_plus
    cmp $15, %eax
    je .s_minus
    jmp .s_accum

.s_plus:
    movss %xmm0, %xmm6                  # xmm6 = +scale_gy
    jmp .s_accum

.s_minus:
    # xmm6 = 0.0, compute 0.0 - scale_gy = -scale_gy
    vsubss %xmm0, %xmm6, %xmm6          # xmm6 = 0.0 - scale_gy = -scale_gy

.s_accum:
    # grad_x[i] += contribution
    addss (%r14), %xmm6                  # xmm6 = grad_x[i] + contribution

    # L-08: NaN/Inf guard (scalar) — zero if non-finite
    movd %xmm6, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne .s_store
    vxorps %xmm6, %xmm6, %xmm6          # zero it

.s_store:
    movss %xmm6, (%r14)                  # store grad_x[i]

    # Advance nibble/element
    inc %rbx
    add $4, %r14                        # grad_x += 1 f32
    dec %r15
    jz .done

    # If consumed 8 nibbles in this word, advance to next word
    cmp $8, %rbx
    jne .scalar_loop
    xor %rbx, %rbx
    add $4, %r13                        # weights += 1 u32 word
    jmp .scalar_loop

.done:
    vzeroupper
    add $8, %rsp
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    pop %rbp
    ret

# Constants (32-byte aligned for 256-bit loads)
.align 32
btg_shifts:     .long 0, 4, 8, 12, 16, 20, 24, 28
btg_mask:       .long 15, 15, 15, 15, 15, 15, 15, 15
btg_val_one:    .long 1, 1, 1, 1, 1, 1, 1, 1
btg_val_minus:  .long 15, 15, 15, 15, 15, 15, 15, 15
btg_exp_mask:   .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
