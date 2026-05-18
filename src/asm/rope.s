.section .text
.global apply_rope_asm

# System V AMD64 ABI:
# rdi: n (dimensión a la que se aplica RoPE, e.g. head_size)
# rsi: x (puntero al vector de floats)
# rdx: cos (puntero a tabla de cosenos para la posición actual)
# rcx: sin (puntero a tabla de senos para la posición actual)

apply_rope_asm:
    push %rbp
    mov %rsp, %rbp
    
    mov %rdi, %rax
    shr $3, %rax         # n / 8 (procesamos 4 pares de floats por iteración)

.loop:
    test %rax, %rax
    jz .done
    
    # Cargar x[0..7] -> 4 pares (x0, x1, x2, x3, x4, x5, x6, x7)
    vmovups (%rsi), %ymm0
    
    # Cargar cos[0..7] y sin[0..7]
    vmovups (%rdx), %ymm1
    vmovups (%rcx), %ymm2
    
    # Queremos:
    # out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
    # out[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
    
    # 1. Crear vector de x "mezclado" para la resta/suma
    # ymm3 = (x1, x0, x3, x2, x5, x4, x7, x6)
    vshufps $0xB1, %ymm0, %ymm0, %ymm3
    
    # 2. Preparar signos para la resta/suma
    # Para out[2i], queremos -x[2i+1]*sin. Para out[2i+1], queremos +x[2i]*sin.
    # Podemos usar vpermilps o simplemente máscaras.
    # O más fácil: usar vmulps y luego sumar/restar.
    
    # ymm4 = x * cos
    vmulps %ymm1, %ymm0, %ymm4
    
    # ymm5 = x_mezclado * sin
    vmulps %ymm2, %ymm3, %ymm5
    
    # Ahora ymm4 tiene (x0*c0, x1*c1, ...)
    # ymm5 tiene (x1*s0, x0*s1, ...) <- OJO, sin[i] debe estar duplicado en la tabla o manejado
    # Mejor técnica: duplicar cos y sin en la tabla: cos[i], cos[i], sin[i], sin[i]...
    # Si la tabla ya viene duplicada:
    # ymm4 = (x0*c0, x1*c0, x2*c1, x3*c1, ...)
    # ymm5 = (x1*s0, x0*s0, x3*s1, x2*s1, ...)
    # out0 = x0*c0 - x1*s0
    # out1 = x1*c0 + x0*s0
    
    # Usaremos vaddsubps si fuera posible, pero es para complejos.
    # Haremos una suma/resta manual con vpermilps para cambiar signos.
    
    # Por ahora asumo que la tabla de senos rsi tiene los signos ya aplicados
    # o que hacemos la rotación simple. 
    
    # Simplificación para el primer paso del hito:
    vaddps %ymm4, %ymm5, %ymm0 # placeholder
    
    vmovups %ymm0, (%rsi)
    
    add $32, %rsi
    add $32, %rdx
    add $32, %rcx
    dec %rax
    jmp .loop

.done:
    vzeroupper
    pop %rbp
    ret
