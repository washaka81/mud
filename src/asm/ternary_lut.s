.section .text
.global ternary_gemv_lut_avx2

# System V AMD64 ABI:
# rdi: n (total weights, multiple of 32 for unrolled loop)
# rsi: x (INT8 quantized activations)
# rdx: weights (Unpacked INT8 weights: -1, 0, 1)
# rcx: out (Pointer to FP32 result)
# xmm0: scale (Global layer scale)

ternary_gemv_lut_avx2:
    push %rbp
    mov %rsp, %rbp
    
    # xmm0 contains global scale, broadcast it
    vbroadcastss %xmm0, %ymm8
    
    # ymm9: Accumulator (FP32)
    vxorps %ymm9, %ymm9, %ymm9

    # We will accumulate in 16-bit integers using vpmaddubsw
    # then widen to 32-bit integers, then convert to FP32.
    
    # Zero register for widening
    vxorps %ymm10, %ymm10, %ymm10
    
    # 32-bit int accumulators
    vxorps %ymm11, %ymm11, %ymm11
    vxorps %ymm12, %ymm12, %ymm12

.loop:
    cmp $32, %rdi
    jl .leftover
    
    # Load 32 INT8 activations (Unsigned logic technically but we assume signed/unsigned mix)
    # vpmaddubsw: dest[i] = SATURATE_16(src1[2i]*src2[2i] + src1[2i+1]*src2[2i+1])
    # src1 must be UNSIGNED bytes (0 to 255).
    # src2 must be SIGNED bytes (-128 to 127).
    # Since weights are -1, 0, 1 (SIGNED), they must be src2.
    # But activations can be negative.
    # To fix this, we can shift activations to be positive (add 128), and compensate,
    # or just use vpmaddwd after widening to 16-bit, which is safer and fast.
    
    # Safer approach: widen INT8 to INT16 first, then vpmaddwd
    # Load 16 activations
    vpmovsxbw (%rsi), %ymm1
    # Load 16 weights
    vpmovsxbw (%rdx), %ymm2
    # Multiply and add adjacent pairs -> 32-bit ints
    vpmaddwd %ymm2, %ymm1, %ymm3
    # Accumulate 32-bit ints
    vpaddd %ymm3, %ymm11, %ymm11
    
    # Next 16
    vpmovsxbw 16(%rsi), %ymm1
    vpmovsxbw 16(%rdx), %ymm2
    vpmaddwd %ymm2, %ymm1, %ymm3
    vpaddd %ymm3, %ymm12, %ymm12
    
    add $32, %rsi
    add $32, %rdx
    sub $32, %rdi
    jmp .loop

.leftover:
    # (Skipping leftover logic for simplicity in this prototype)
    
.done_accum:
    # Combine ymm11 and ymm12
    vpaddd %ymm12, %ymm11, %ymm11
    
    # Convert INT32 to FP32
    vcvtdq2ps %ymm11, %ymm11
    
    # Multiply by scale
    vmulps %ymm8, %ymm11, %ymm11
    
    # Horizontal reduction of ymm11 (FP32)
    vextractf128 $1, %ymm11, %xmm1
    vaddps %xmm1, %xmm11, %xmm0
    vshufps $0xEE, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0x11, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    
    vaddss (%rcx), %xmm0, %xmm0
    vmovss %xmm0, (%rcx)
    vzeroupper

    pop %rbp
    ret
