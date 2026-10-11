.section .text
.global rms_norm_scale_asm

# System V AMD64 ABI:
# rdi: n (size_t)
# rsi: x (float*)
# xmm0: eps (float) -> Result will be in xmm0

rms_norm_scale_asm:
    push %rbp
    mov %rsp, %rbp
    
    # ymm1 = acumulador de sumas de cuadrados
    vxorps %ymm1, %ymm1, %ymm1
    
    mov %rdi, %rax
    shr $3, %rax         # n / 8 (bloques de 8 floats)

.loop:
    test %rax, %rax
    jz .done
    
    vmovups (%rsi), %ymm2
    vmulps %ymm2, %ymm2, %ymm2     # x^2
    vaddps %ymm2, %ymm1, %ymm1     # acc += x^2
    
    add $32, %rsi
    dec %rax
    jmp .loop

.done:
    # Reducción horizontal de ymm1 a xmm1
    vextractf128 $1, %ymm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vhaddps %xmm1, %xmm1, %xmm1
    vhaddps %xmm1, %xmm1, %xmm1

    # L-08: non-finite sum_sq → return 0 scale (fail-safe)
    vmovd %xmm1, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    je .rms_zero

    # mean(x^2) = sum / n  (n==0 → zero)
    test %rdi, %rdi
    jz .rms_zero
    vcvtsi2ss %rdi, %xmm3, %xmm3
    vdivss %xmm3, %xmm1, %xmm1

    # eps floor 1e-8 if eps non-positive or non-finite
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    je .rms_eps_floor
    vxorps %xmm2, %xmm2, %xmm2
    vcomiss %xmm2, %xmm0
    jbe .rms_eps_floor
    jmp .rms_have_eps
.rms_eps_floor:
    movl $0x322BCC77, %eax         # 1e-8f
    vmovd %eax, %xmm0
.rms_have_eps:
    vaddss %xmm0, %xmm1, %xmm1

    # 1.0 / sqrt(mean + eps)
    vsqrtss %xmm1, %xmm1, %xmm1
    movl $0x3f800000, %eax         # 1.0f
    vmovd %eax, %xmm0
    vdivss %xmm1, %xmm0, %xmm0

    # L-08: sanitize final scale
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne .rms_done
.rms_zero:
    vxorps %xmm0, %xmm0, %xmm0
.rms_done:
    vzeroupper
    pop %rbp
    ret
