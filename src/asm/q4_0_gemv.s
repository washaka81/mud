.section .text
.global q4_0_gemv_asm

# rdi=n (total elements, e.g. 1536)
# rsi=x (activations FP32)
# rdx=weights (BlockQ4_0)
# rcx=out (Puntero a float de salida)

q4_0_gemv_asm:
    push %rbp
    mov %rsp, %rbp
    
    vxorps %ymm0, %ymm0, %ymm0
    
    # Constantes
    mov $0x41000000, %eax
    vmovd %eax, %xmm2
    vbroadcastss %xmm2, %ymm2      # ymm2 = 8.0f (bias)

    mov $0x0F, %eax
    vmovd %eax, %xmm1
    vpbroadcastd %xmm1, %ymm1      # ymm1 = 0x0F (máscara)

.loop:
    cmp $32, %rdi
    jl .done

    # 1. Cargar factor de escala d
    movzwl (%rdx), %eax
    vmovd %eax, %xmm6
    vcvtph2ps %xmm6, %xmm6
    vbroadcastss %xmm6, %ymm6      # ymm6 = d

    # 2. Cargar 16 bytes de pesos (32 elementos)
    vmovdqu 2(%rdx), %xmm3         # xmm3 = [qs0..qs15]

    # --- LAYOUT GGUF Q4_0 ---
    # Elementos 0..15  = bajos de qs[0..15]
    # Elementos 16..31 = altos de qs[0..15]

    # A. Elementos 0..7 (Bajos bytes 0..7) -> x[0..7]
    vpmovzxbd %xmm3, %ymm4
    vpand %ymm1, %ymm4, %ymm4
    vcvtdq2ps %ymm4, %ymm4
    vsubps %ymm2, %ymm4, %ymm4
    vmulps %ymm6, %ymm4, %ymm4
    vmovups (%rsi), %ymm5
    vfmadd231ps %ymm5, %ymm4, %ymm0

    # B. Elementos 8..15 (Bajos bytes 8..15) -> x[8..15]
    vpsrldq $8, %xmm3, %xmm7
    vpmovzxbd %xmm7, %ymm4
    vpand %ymm1, %ymm4, %ymm4
    vcvtdq2ps %ymm4, %ymm4
    vsubps %ymm2, %ymm4, %ymm4
    vmulps %ymm6, %ymm4, %ymm4
    vmovups 32(%rsi), %ymm5
    vfmadd231ps %ymm5, %ymm4, %ymm0

    # C. Elementos 16..23 (Altos bytes 0..7) -> x[16..23]
    vmovdqa %xmm3, %xmm7
    vpsrlw $4, %xmm7, %xmm7
    vpmovzxbd %xmm7, %ymm4
    vpand %ymm1, %ymm4, %ymm4
    vcvtdq2ps %ymm4, %ymm4
    vsubps %ymm2, %ymm4, %ymm4
    vmulps %ymm6, %ymm4, %ymm4
    vmovups 64(%rsi), %ymm5
    vfmadd231ps %ymm5, %ymm4, %ymm0

    # D. Elementos 24..31 (Altos bytes 8..15) -> x[24..31]
    vpsrldq $8, %xmm3, %xmm7
    vpsrlw $4, %xmm7, %xmm7
    vpmovzxbd %xmm7, %ymm4
    vpand %ymm1, %ymm4, %ymm4
    vcvtdq2ps %ymm4, %ymm4
    vsubps %ymm2, %ymm4, %ymm4
    vmulps %ymm6, %ymm4, %ymm4
    vmovups 96(%rsi), %ymm5
    vfmadd231ps %ymm5, %ymm4, %ymm0

    # Avanzar punteros
    add $18, %rdx
    add $128, %rsi                 # 32 hilos * 4 bytes = 128
    sub $32, %rdi
    jmp .loop

.done:
    # Reducción horizontal a escalar
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    
    vaddss (%rcx), %xmm0, %xmm0
    vmovss %xmm0, (%rcx)

    vzeroupper
    pop %rbp
    ret
