.section .text
.global mhc_gate_backward_avx2

# mHC gate backward: gate = sigmoid(v_jepa), d_v = gate*(1-gate) * d_gate
# Forward: gate = 1 / (1 + exp(-v_jepa))
# Backward: d_v = gate * (1 - gate) * d_gate
#
# ABI (System V AMD64):
#   rdi = n
#   rsi = gate_ptr      (*const f32, sigmoid output values)
#   rdx = grad_gate_ptr (*const f32, upstream gradient w.r.t. gate)
#   rcx = grad_v_ptr    (*mut f32, output: gradient w.r.t. v_jepa)
#
# Register allocation:
#   ymm0 = gate[8]
#   ymm1 = d_gate[8]
#   ymm2 = 1 - gate
#   ymm3 = gate * (1 - gate)
#   ymm4 = result = gate * (1 - gate) * d_gate
#   ymm9 = 1.0 (constant, preserved across loops)
#   ymm10 = exp_mask (L-08 NaN guard)
#   rax = vector block counter
mhc_gate_backward_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15

    mov %rdi, %r12                    # r12 = n (preserved)
    mov %rsi, %r13                    # r13 = gate ptr
    mov %rdx, %r14                    # r14 = grad_gate ptr
    mov %rcx, %r15                    # r15 = grad_v ptr (output)

    # Preload constants
    vmovaps bw_one(%rip), %ymm9       # ymm9 = 1.0
    vmovaps bw_exp_mask(%rip), %ymm10 # ymm10 = exp_mask

    mov %rdi, %rax
    shr $3, %rax                      # rax = n / 8 (vector blocks)

    test %rax, %rax
    jz .scalar_tail

.align 32
.vector_loop:
    vmovups (%r13), %ymm0              # gate[0:8]
    vmovups (%r14), %ymm1              # d_gate[0:8]

    # ymm2 = 1.0 - gate
    vsubps %ymm0, %ymm9, %ymm2          # = ymm9 - ymm0 = 1.0 - gate
    # ymm3 = gate * (1 - gate)  [sigmoid derivative]
    vmulps %ymm2, %ymm0, %ymm3
    # ymm4 = d_gate * gate * (1 - gate)
    vmulps %ymm1, %ymm3, %ymm4

    # NaN/Inf guard (L-08)
    vpand %ymm10, %ymm4, %ymm5
    vpcmpeqd %ymm10, %ymm5, %ymm5
    vpandn %ymm4, %ymm5, %ymm4

    vmovups %ymm4, (%r15)

    add $32, %r13
    add $32, %r14
    add $32, %r15
    dec %rax
    jnz .vector_loop

    # Scalar tail for remaining elements (n % 8)
.scalar_tail:
    mov %r12, %rax                    # rax = original n
    and $7, %rax                       # remaining elements
    test %rax, %rax
    jz .done

    vmovss bw_one(%rip), %xmm9
    vmovdqu bw_exp_mask(%rip), %xmm10

.scalar_loop:
    movss (%r13), %xmm0               # gate
    movss (%r14), %xmm1               # d_gate

    # 1 - gate
    subss %xmm0, %xmm9                # xmm9 = 1.0 - gate
    # gate * (1 - gate)
    mulss %xmm9, %xmm0                # xmm0 = gate * (1-gate)
    # d_gate * gate * (1 - gate)
    mulss %xmm1, %xmm0                # xmm0 = d_gate * gate * (1-gate)

    # Reset xmm9 to 1.0 for next iteration
    vmovss bw_one(%rip), %xmm9

    # NaN guard (L-08): non-finite → 0
    movaps %xmm0, %xmm5
    vpand %xmm10, %xmm5, %xmm5
    vpcmpeqd %xmm10, %xmm5, %xmm5
    vpandn %xmm0, %xmm5, %xmm0

    movss %xmm0, (%r15)

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
bw_one: .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
bw_exp_mask: .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
