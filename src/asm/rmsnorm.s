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
    vshufps $0xEE, %xmm1, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vshufps $0x11, %xmm1, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    
    # xmm1[0] ahora tiene la suma total de cuadrados
    
    # Calcular mean(x^2) = sum / n
    vcvtsi2ss %rdi, %xmm3, %xmm3
    vdivss %xmm3, %xmm1, %xmm1     # mean = sum / n
    
    # mean + eps
    vaddss %xmm0, %xmm1, %xmm1
    
    # 1.0 / sqrt(mean + eps)
    vsqrtss %xmm1, %xmm1, %xmm1
    mov $0x3f800000, %eax          # 1.0f
    vmovd %eax, %xmm0
    vdivss %xmm1, %xmm0, %xmm0     # xmm0 = scale
    
    vzeroupper
    pop %rbp
    ret
