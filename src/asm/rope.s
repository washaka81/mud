.section .text
.global apply_rope_asm
.global apply_rope_interleaved_asm

# System V AMD64 ABI:
# rdi: n (head_dim)
# rsi: x (puntero al vector de floats)
# rdx: cos (puntero a tabla de cosenos)
# rcx: sin (puntero a tabla de senos)

# Optimized Split RoPE for AVX2
apply_rope_asm:
    push %rbp
    mov %rsp, %rbp
    
    mov %rdi, %rax
    shr $1, %rax         # half = n / 2
    mov %rax, %r8        # r8 = half
    shr $3, %rax         # half / 8 (iteramos sobre bloques de 8 pares)

    # Offset para la segunda mitad
    mov %r8, %r9
    shl $2, %r9          # half * sizeof(float)

.align 32
.loop_split:
    test %rax, %rax
    jz .done_split
    
    vmovups (%rsi), %ymm0           # x[i..i+7]
    vmovups (%rsi, %r9), %ymm1      # x[i+half..i+half+7]
    vmovups (%rdx), %ymm2           # cos[i..i+7]
    vmovups (%rcx), %ymm3           # sin[i..i+7]
    
    vmulps %ymm0, %ymm2, %ymm4      # x[i]*cos[i]
    vmulps %ymm1, %ymm3, %ymm5      # x[i+half]*sin[i]
    vsubps %ymm5, %ymm4, %ymm6      # out[i]
    
    vmulps %ymm0, %ymm3, %ymm4      # x[i]*sin[i]
    vmulps %ymm1, %ymm2, %ymm5      # x[i+half]*cos[i]
    vaddps %ymm5, %ymm4, %ymm7      # out[i+half]
    
    vmovups %ymm6, (%rsi)
    vmovups %ymm7, (%rsi, %r9)
    
    add $32, %rsi
    add $32, %rdx
    add $32, %rcx
    dec %rax
    jmp .loop_split

.done_split:
    vzeroupper
    pop %rbp
    ret

# Placeholder for future AVX2 optimization. 
# Current implementation is handled in Rust within MudInference::apply_rope.
apply_rope_interleaved_asm:
    push %rbp
    mov %rsp, %rbp
    vzeroupper
    pop %rbp
    ret
