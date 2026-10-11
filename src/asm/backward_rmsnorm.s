.section .text
.global backward_rmsnorm_avx2

# Backward RMSNorm (elementwise part):
#   grad_x[i] = grad_y[i] * weight[i] - y_unscaled[i] * mean_dy_xnorm
#
# Uses vfmadd231ps to fuse the multiply-subtract into one FMA:
#   grad_x = grad_y*weight + y_unscaled*(-mean)
# = grad_y*weight - y_unscaled*mean
#
# ABI (System V AMD64):
#   rdi = n               (element count)
#   rsi = grad_y_ptr      (*const f32)
#   rdx = y_unscaled_ptr  (*const f32, pre-norm x*inv_rms)
#   rcx = weight_ptr      (*const f32)
#   r8  = mean_bits       (u32, f32 bits: mean_dy_xnorm)
#   r9  = grad_x_ptr       (*mut f32, output)
#
# Register allocation:
#   ymm0 = grad_y[8]
#   ymm1 = accumulator (grad_x result)
#   ymm2 = weight[8]
#   ymm3 = neg_mean (broadcast)
backward_rmsnorm_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15

    mov %rdi, %r12                    # r12 = n
    mov %rsi, %r13                    # r13 = grad_y ptr
    mov %rdx, %r14                    # r14 = y_unscaled ptr
    mov %rcx, %r10                    # r10 = weight ptr (r10 is caller-saved, safe)
    mov %r9, %r15                     # r15 = grad_x output ptr

    # Broadcast mean_dy_xnorm and negate it (for FMA subtract)
    vmovd %r8d, %xmm3                 # xmm3 = mean_dy_xnorm bits
    vxorps bw_neg_zero(%rip), %xmm3, %xmm3  # xmm3 = -mean
    vbroadcastss %xmm3, %ymm3         # ymm3 = -mean (8 lanes)

.align 32
.vector_loop:
    cmp $8, %r12
    jl .scalar_remainder

    vmovups (%r13), %ymm0              # grad_y[0:8]
    vmovups (%r14), %ymm1              # y_unscaled[0:8]
    vmovups (%r10), %ymm2              # weight[0:8]

    # Step 1: ymm1 = y_unscaled * (-mean)
    vmulps %ymm3, %ymm1, %ymm1
    # Step 2: grad_x = ymm1 + grad_y * weight
    # vfmadd231ps %op1, %op2, %op3 → op3 = op3 + op2 * op1
    # %ymm2=weight, %ymm0=grad_y, %ymm1=result
    vfmadd231ps %ymm2, %ymm0, %ymm1

    vmovups %ymm1, (%r15)              # store grad_x

    add $32, %r13                       # grad_y += 8
    add $32, %r14                       # y_unscaled += 8
    add $32, %r10                       # weight += 8
    add $32, %r15                       # grad_x += 8
    sub $8, %r12
    jmp .vector_loop

.scalar_remainder:
    test %r12, %r12
    jz .done

    # Scalar path uses x87-free SIMD scalar ops
    vmovss bw_neg_zero(%rip), %xmm6   # load -0.0 mask
    vmovd %r8d, %xmm4                 # xmm4 = mean_bits (rebroadcast for scalar)
    vxorps %xmm4, %xmm6, %xmm4         # xmm4 = -mean (scalar version)

.scalar_loop:
    cmp $0, %r12
    je .done

    vmovss (%r13), %xmm0              # grad_y[i]
    vmovss (%r14), %xmm1              # y_unscaled[i]
    vmovss (%r10), %xmm2              # weight[i]

    # grad_x = grad_y*weight - y_unscaled*mean
    # = grad_y*weight + y_unscaled*(-mean)
    vmulss %xmm4, %xmm1, %xmm5         # tmp = y_unscaled * (-mean)
    vfmadd231ss %xmm2, %xmm0, %xmm5    # xmm5 = xmm5 + grad_y * weight

    # L-08: NaN/Inf guard — if not finite, set to 0
    vmovss bw_exp_mask(%rip), %xmm6
    vpand %xmm6, %xmm5, %xmm7
    vpcmpeqd %xmm6, %xmm7, %xmm7
    vpandn %xmm5, %xmm7, %xmm5

    vmovss %xmm5, (%r15)

    add $4, %r13
    add $4, %r14
    add $4, %r10
    add $4, %r15
    dec %r12
    jmp .scalar_loop

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
