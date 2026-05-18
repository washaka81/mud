.section .text
.global ternary_gemv_avx2

# System V AMD64 ABI:
# rdi: n (total weights, multiple of 16)
# rsi: x (FP32 activations)
# rdx: weights (Ternary 2-bit packed, 16 per u32)
# rcx: out (Pointer to FP32 result)
# xmm0: scale (Global layer scale)

# Constants for bit extraction
# We need shifts: 0, 2, 4, 6, 8, 10, 12, 14
.section .rodata
.align 32
SHIFTS_LOW:  .long 0, 2, 4, 6, 8, 10, 12, 14
SHIFTS_HIGH: .long 16, 18, 20, 22, 24, 26, 28, 30
MASK_2BIT:   .long 3, 3, 3, 3, 3, 3, 3, 3
VAL_ONE:     .long 1, 1, 1, 1, 1, 1, 1, 1
VAL_TWO:     .long 2, 2, 2, 2, 2, 2, 2, 2

.section .text
ternary_gemv_avx2:
    push %rbp
    mov %rsp, %rbp
    
    # Save scale (xmm0) to ymm8 before clearing ymm0
    vbroadcastss %xmm0, %ymm8
    
    # ymm0: Accumulator (FP32)
    vxorps %ymm0, %ymm0, %ymm0
    
    vmovdqa SHIFTS_LOW(%rip), %ymm10
    vmovdqa SHIFTS_HIGH(%rip), %ymm11
    vmovdqa MASK_2BIT(%rip), %ymm12
    vmovdqa VAL_ONE(%rip), %ymm13
    vmovdqa VAL_TWO(%rip), %ymm14

.loop:
    test %rdi, %rdi
    jle .done
    
    # 1. Load 16 packed weights (1x u32)
    vpbroadcastd (%rdx), %ymm1
    
    # 2. Extract first 8 weights (Lower 16 bits)
    # ymm2 = (ymm1 >> SHIFTS_LOW) & 3
    vpsrlvd %ymm10, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    
    # Create masks: 1 -> +x, 2 -> -x
    vpcmpeqd %ymm13, %ymm2, %ymm3     # ymm3 = mask for +1
    vpcmpeqd %ymm14, %ymm2, %ymm4     # ymm4 = mask for -1 (represented as 2 in bits)
    
    # Load 8 activations
    vmovups (%rsi), %ymm5
    
    # Conditional Add/Sub
    vpand %ymm3, %ymm5, %ymm6         # ymm6 = x where weight is 1, else 0
    vpand %ymm4, %ymm5, %ymm7         # ymm7 = x where weight is -1, else 0
    
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0
    
    # 3. Extract next 8 weights (Upper 16 bits)
    # ymm2 = (ymm1 >> SHIFTS_HIGH) & 3
    vpsrlvd %ymm11, %ymm1, %ymm2
    vpand %ymm12, %ymm2, %ymm2
    
    vpcmpeqd %ymm13, %ymm2, %ymm3
    vpcmpeqd %ymm14, %ymm2, %ymm4
    
    vmovups 32(%rsi), %ymm5           # Load next 8 activations
    
    vpand %ymm3, %ymm5, %ymm6
    vpand %ymm4, %ymm5, %ymm7
    
    vaddps %ymm6, %ymm0, %ymm0
    vsubps %ymm7, %ymm0, %ymm0

    add $4, %rdx                      # Next u32
    add $64, %rsi                     # 16 * 4 bytes
    sub $16, %rdi
    jmp .loop

.done:
    # Scale result: ymm0 *= ymm8 (saved scale)
    vmulps %ymm8, %ymm0, %ymm0

    # Horizontal reduction to scalar
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    
    # Add to existing output (rcx)
    vaddss (%rcx), %xmm0, %xmm0
    vmovss %xmm0, (%rcx)
    
    vzeroupper
    pop %rbp
    ret
