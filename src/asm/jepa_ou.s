.section .text
.global jepa_ou_avx2

# JEPA OU EMA update: z_next[i] = alpha * y_norm[i] + (1 - alpha) * z[i]
# = z[i] + alpha * (y_norm[i] - z[i])
#
# Uses 2 instructions in vector loop:
#   1. vsubps: diff = y_norm - z
#   2. vfmadd231ps: z_next = z + alpha * diff
#
# System V AMD64 ABI:
#   rdi = n             (element count)
#   rsi = z_ptr         (*const f32, current z values)
#   rdx = y_norm_ptr    (*const f32, normalized block output)
#   rcx = z_next_ptr    (*mut f32, output)
#   r8  = alpha_d       (f64; alpha = 0.1 for pure OU per AGENTS.md §5)
#
# Register allocation:
#   r12 = n (countdown)
#   r13 = z ptr
#   r14 = y_norm ptr
#   r15 = z_next ptr
#   ymm0 = z[8]
#   ymm1 = y_norm[8]
#   ymm2 = diff = y_norm - z
#   ymm2 = z_next = z + alpha * diff
#   ymm3 = alpha (broadcast)
#   ymm4 = exp_mask (NaN/Inf guard)
jepa_ou_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    sub $8, %rsp

    mov %rdi, %r12
    mov %rsi, %r13
    mov %rdx, %r14
    mov %rcx, %r15

    # Load alpha from r8d (float bits in low 32 bits of r8)
    vmovd %r8d, %xmm3
    vbroadcastss %xmm3, %ymm3        # ymm3 = alpha (0.1)

    # Load exp_mask for NaN/Inf guard
    vmovdqu jepa_exp_mask(%rip), %ymm4

.vector_loop:
    cmp $8, %r12
    jl .scalar_remainder

    vmovups (%r13), %ymm0              # ymm0 = z[0:8]
    vmovups (%r14), %ymm1              # ymm1 = y_norm[0:8]
    vsubps %ymm0, %ymm1, %ymm2         # ymm2 = y_norm - z (GAS reversed)
    vfmadd231ps %ymm2, %ymm3, %ymm0    # ymm0 = z + alpha * (y_norm - z)
                                       #   GAS: vfmadd231ps %op1, %op2, %op3
                                       #   → op3 = op3 + op1 * op2
                                       #   → ymm0 = ymm0 + ymm2 * ymm3

    # L-08: NaN/Inf guard — zero out non-finite lanes
    vpand %ymm4, %ymm0, %ymm2
    vpcmpeqd %ymm4, %ymm2, %ymm2
    vpandn %ymm0, %ymm2, %ymm0        # ymm0 = ~mask & z_next

    vmovups %ymm0, (%r15)

    add $32, %r13
    add $32, %r14
    add $32, %r15
    sub $8, %r12
    jmp .vector_loop

.scalar_remainder:
.scalar_loop:
    cmp $0, %r12
    je .done

    vmovss (%r13), %xmm0              # z[i]
    vmovss (%r14), %xmm1              # y_norm[i]

    # z_next = z + alpha * (y_norm - z)
    vsubss %xmm0, %xmm1, %xmm2        # xmm2 = y_norm - z
    vfmadd231ss %xmm2, %xmm3, %xmm0   # xmm0 = z + alpha * (y_norm - z)

    # NaN/Inf guard (scalar)
    vmovss jepa_exp_mask(%rip), %xmm2
    vpand %xmm2, %xmm0, %xmm1
    vpcmpeqd %xmm2, %xmm1, %xmm1
    vpandn %xmm0, %xmm1, %xmm0

    vmovss %xmm0, (%r15)

    add $4, %r13
    add $4, %r14
    add $4, %r15
    dec %r12
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

# Constants (8-element arrays for 256-bit aligned loads)
.align 32
jepa_alpha:     .float 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
jepa_exp_mask:  .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
