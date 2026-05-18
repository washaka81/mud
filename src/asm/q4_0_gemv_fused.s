.section .text
.global q4_0_gemv_fused_asm

# System V AMD64 ABI: rdi=n, rsi=x, rdx=weights, rcx=out, xmm0=scale

q4_0_gemv_fused_asm:
    push %rbp
    mov %rsp, %rbp
    
    # ymm8 = scale
    vbroadcastss %xmm0, %ymm8
    
    # Limpiar ymm0 DESPUES de usar xmm0
    vxorps %ymm0, %ymm0, %ymm0
    
    # bias 8.0f
    mov $0x41000000, %eax
    vmovd %eax, %xmm2
    vbroadcastss %xmm2, %ymm2

    # máscara 0x0F
    mov $0x0F, %eax
    vmovd %eax, %xmm1
    vpbroadcastd %xmm1, %ymm1

.loop:
    test %rdi, %rdi
    jle .done

    # Cargar d (FP16)
    movzwl (%rdx), %eax
    vmovd %eax, %xmm6
    vcvtph2ps %xmm6, %xmm6
    vbroadcastss %xmm6, %ymm6
    vmulps %ymm8, %ymm6, %ymm6      # d * scale

    # Cargar 8 pesos
    vpmovzxbd 2(%rdx), %ymm4
    vpand %ymm1, %ymm4, %ymm4
    vcvtdq2ps %ymm4, %ymm4
    vsubps %ymm2, %ymm4, %ymm4
    vmulps %ymm6, %ymm4, %ymm4

    # Cargar x y FMA
    vmovups (%rsi), %ymm5
    vfmadd231ps %ymm5, %ymm4, %ymm0

    add $18, %rdx
    add $32, %rsi
    sub $8, %rdi
    jmp .loop

.done:
    vmovups %ymm0, (%rcx)
    vzeroupper
    pop %rbp
    ret
