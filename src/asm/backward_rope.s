.section .text
.global backward_rope_asm

# Backward RoPE rotation (same split layout as apply_rope_asm):
#   dx0 = dy0 * cos + dy1 * sin
#   dx1 = -dy0 * sin + dy1 * cos
#
# ABI (System V AMD64):
#   rdi = n              (head_dim; first half = real, second half = imag)
#   rsi = grad_y_ptr    (*mut f32, in-place: [re0..re(half-1), im0..im(half-1)])
#   rdx = cos_tbl       (*const f32, half_d elements)
#   rcx = sin_tbl       (*const f32, half_d elements)
#
# Register allocation:
#   ymm0 = grad_real[8]
#   ymm1 = grad_imag[8]
#   ymm2 = cos[8]
#   ymm3 = sin[8]
#   ymm4 = dx_real (result)
#   ymm5 = dx_imag (result)
#   ymm6 = temp (neg_mask, exp_mask, comparison)
#   ymm7 = temp
backward_rope_asm:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15

    mov %rdi, %r12                    # r12 = n (head_dim)
    mov %rsi, %r13                    # r13 = grad_y ptr (in-place)
    mov %rdx, %r14                    # r14 = cos ptr
    mov %rcx, %r15                    # r15 = sin ptr

    # Compute half_d and byte offset to imag half
    mov %r12, %rax
    shr $1, %rax                      # half = n / 2
    mov %rax, %r8                     # r8 = half (for scalar tail)
    shl $2, %r8                       # r8 = half * sizeof(float) (byte offset)
    shr $3, %rax                      # rax = half / 8 (vector blocks)

    test %rax, %rax
    jz .scalar_tail

    # Preload constants
    vmovaps bw_neg_zero(%rip), %ymm9  # ymm9 = -0.0
    vmovaps bw_exp_mask(%rip), %ymm10 # ymm10 = exp_mask

.align 32
.vector_loop:
    vmovups (%r13), %ymm0              # grad_real[0:8]
    vmovups (%r13, %r8), %ymm1         # grad_imag[0:8]
    vmovups (%r14), %ymm2              # cos[0:8]
    vmovups (%r15), %ymm3              # sin[0:8]

    # dx_real = grad_real * cos + grad_imag * sin
    vmulps %ymm2, %ymm0, %ymm4         # grad_real * cos
    vfmadd231ps %ymm3, %ymm1, %ymm4    # ymm4 += grad_imag * sin

    # dx_imag = grad_imag * cos - grad_real * sin
    # = grad_imag * cos + grad_real * (-sin)
    vmulps %ymm2, %ymm1, %ymm5         # grad_imag * cos
    vxorps %ymm9, %ymm3, %ymm6         # ymm6 = -sin
    vfmadd231ps %ymm6, %ymm0, %ymm5    # ymm5 += grad_real * (-sin)

    # NaN/Inf guard (L-08): non-finite → 0
    vpand %ymm10, %ymm4, %ymm7
    vpcmpeqd %ymm10, %ymm7, %ymm7
    vpandn %ymm4, %ymm7, %ymm4
    vpand %ymm10, %ymm5, %ymm7
    vpcmpeqd %ymm10, %ymm7, %ymm7
    vpandn %ymm5, %ymm7, %ymm5

    vmovups %ymm4, (%r13)              # store dx_real
    vmovups %ymm5, (%r13, %r8)         # store dx_imag

    add $32, %r13                       # grad_y += 8
    add $32, %r14                       # cos += 8
    add $32, %r15                       # sin += 8
    dec %rax
    jnz .vector_loop

    # Scalar tail for remaining pairs (0..7)
.scalar_tail:
    mov %r12, %rax
    shr $1, %rax                      # half
    and $7, %rax                      # remaining pairs
    test %rax, %rax
    jz .done

    vmovss bw_neg_zero(%rip), %xmm9   # -0.0 for scalar negation
    vmovdqu bw_exp_mask(%rip), %xmm10 # exp_mask for NaN guard

.scalar_loop:
    movss (%r13), %xmm0               # grad_real[i]
    movss (%r13, %r8), %xmm1          # grad_imag[i]
    movss (%r14), %xmm2               # cos[i]
    movss (%r15), %xmm3               # sin[i]

    # dx_real = grad_real * cos + grad_imag * sin
    movss %xmm0, %xmm4                # copy grad_real
    mulss %xmm2, %xmm4                # grad_real * cos
    movss %xmm1, %xmm5                # grad_imag
    mulss %xmm3, %xmm5                # grad_imag * sin
    addss %xmm5, %xmm4                # dx_real

    # dx_imag = grad_imag * cos - grad_real * sin
    movss %xmm1, %xmm5                # grad_imag (fresh copy)
    mulss %xmm2, %xmm5                # grad_imag * cos
    movss %xmm0, %xmm6                # grad_real (fresh copy)
    mulss %xmm3, %xmm6                # grad_real * sin
    subss %xmm6, %xmm5                # dx_imag = grad_imag*cos - grad_real*sin

    # NaN/Inf guard (L-08) for dx_real
    movss %xmm4, %xmm7
    vpand %xmm10, %xmm7, %xmm7
    vpcmpeqd %xmm10, %xmm7, %xmm7
    vpandn %xmm4, %xmm7, %xmm4

    # NaN/Inf guard (L-08) for dx_imag
    movss %xmm5, %xmm7
    vpand %xmm10, %xmm7, %xmm7
    vpcmpeqd %xmm10, %xmm7, %xmm7
    vpandn %xmm5, %xmm7, %xmm5

    movss %xmm4, (%r13)               # store dx_real
    movss %xmm5, (%r13, %r8)          # store dx_imag

    add $4, %r13
    add $4, %r14
    add $4, %r15
    dec %rax
    jnz .scalar_loop

.done:
    vzeroupper
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    pop %rbp
    ret

# Constants
.align 32
bw_neg_zero: .float -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0
bw_exp_mask: .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
