.section .text
.global silu_vectorial_avx2

# Constants
.align 32
log2e:      .float 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
c0:         .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
c1:         .float 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718
c2:         .float 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650
c3:         .float 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411
c4:         .float 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812
i127:       .long 127, 127, 127, 127, 127, 127, 127, 127
neg_zero:   .float -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0
two:        .float 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
clamp_lo:   .float -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0
clamp_hi:   .float 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0
.align 32
exp_mask:   .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000

# silu_vectorial_avx2(n: usize, src: *const f32, dst: *mut f32)
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
# Polish: clamp x for exp domain, prefetch, vrcpps + 1 NR instead of vdivps
silu_vectorial_avx2:
    push %rbp
    mov %rsp, %rbp
    
    # Load constants to YMM registers
    vmovups log2e(%rip), %ymm8
    vmovups c4(%rip), %ymm9
    vmovups c3(%rip), %ymm10
    vmovups c2(%rip), %ymm11
    vmovups c1(%rip), %ymm12
    vmovups c0(%rip), %ymm13
    vmovdqu i127(%rip), %ymm14
    vmovups neg_zero(%rip), %ymm15

.loop:
    cmp $8, %rdi
    jl .leftover

    prefetcht0 256(%rsi)
    prefetcht0 256(%rdx)
    
    # Load 8 floats; clamp for stable exp
    vmovups (%rsi), %ymm0   # ymm0 = x (true, unclamped for final mul)
    vmaxps clamp_lo(%rip), %ymm0, %ymm1
    vminps clamp_hi(%rip), %ymm1, %ymm1  # ymm1 = x_clamped for exp path
    
    # t = -x_clamped
    vxorps %ymm15, %ymm1, %ymm1 # ymm1 = t = -x
    
    # y = t * log2e
    vmulps %ymm8, %ymm1, %ymm2 # ymm2 = y
    
    # n = round(y)
    vroundps $0, %ymm2, %ymm3 # ymm3 = n (float)
    
    # f = y - n
    vsubps %ymm3, %ymm2, %ymm4 # ymm4 = f
    
    # Evaluate polynomial: c4*f^4 + c3*f^3 + c2*f^2 + c1*f + 1
    # p = c4
    vmovups %ymm9, %ymm5
    # p = p*f + c3
    vfmadd213ps %ymm10, %ymm4, %ymm5
    # p = p*f + c2
    vfmadd213ps %ymm11, %ymm4, %ymm5
    # p = p*f + c1
    vfmadd213ps %ymm12, %ymm4, %ymm5
    # p = p*f + 1
    vfmadd213ps %ymm13, %ymm4, %ymm5 # ymm5 = 2^f
    
    # Compute 2^n
    vcvtps2dq %ymm3, %ymm6 # ymm6 = n (int32)
    vpaddd %ymm14, %ymm6, %ymm6 # ymm6 = n + 127
    vpslld $23, %ymm6, %ymm6 # ymm6 = (n + 127) << 23
    
    # exp(t) = 2^f * 2^n
    vmulps %ymm6, %ymm5, %ymm5 # ymm5 = exp(-x)
    
    # d = 1.0 + exp(-x)
    vaddps %ymm13, %ymm5, %ymm5 # ymm5 = d
    
    # inv_d ≈ rcp(d); one Newton-Raphson: r = r*(2 - d*r)
    vrcpps %ymm5, %ymm6
    vmulps %ymm6, %ymm5, %ymm7
    vmovups two(%rip), %ymm4
    vsubps %ymm7, %ymm4, %ymm7
    vmulps %ymm7, %ymm6, %ymm6   # ymm6 = 1/d refined
    # silu = x * inv_d  (use original unclamped x)
    vmulps %ymm6, %ymm0, %ymm7

    # L-08: NaN/Inf lanes → 0
    vmovdqa exp_mask(%rip), %ymm1
    vpand %ymm1, %ymm7, %ymm2
    vpcmpeqd %ymm1, %ymm2, %ymm2
    vpandn %ymm7, %ymm2, %ymm7
    
    # Store 8 floats
    vmovups %ymm7, (%rdx)
    
    add $32, %rsi
    add $32, %rdx
    sub $8, %rdi
    jmp .loop

.leftover:
    cmp $4, %rdi
    jl .leftover_1
    
    # Process 4 floats at a time using xmm (clamp + div path)
    vmovups (%rsi), %xmm0
    vmaxps clamp_lo(%rip), %xmm0, %xmm1
    vminps clamp_hi(%rip), %xmm1, %xmm1
    
    vxorps %xmm15, %xmm1, %xmm1     # t = -x_clamped
    vmulps %xmm8, %xmm1, %xmm2       # y = t * log2e
    vroundps $0, %xmm2, %xmm3        # n = round(y)
    vsubps %xmm3, %xmm2, %xmm4       # f = y - n
    
    vmovaps %xmm9, %xmm5
    vfmadd213ps %xmm10, %xmm4, %xmm5
    vfmadd213ps %xmm11, %xmm4, %xmm5
    vfmadd213ps %xmm12, %xmm4, %xmm5
    vfmadd213ps %xmm13, %xmm4, %xmm5  # p = 2^f
    
    vcvtps2dq %xmm3, %xmm6            # n as int32
    vpaddd %xmm14, %xmm6, %xmm6       # n + 127
    vpslld $23, %xmm6, %xmm6          # (n + 127) << 23
    
    vmulps %xmm6, %xmm5, %xmm5        # exp(-x)
    vaddps %xmm13, %xmm5, %xmm5       # d = 1 + exp(-x)
    vdivps %xmm5, %xmm0, %xmm7        # silu = x_orig / d

    # L-08 sanitize xmm
    vmovdqa exp_mask(%rip), %xmm1
    vpand %xmm1, %xmm7, %xmm2
    vpcmpeqd %xmm1, %xmm2, %xmm2
    vpandn %xmm7, %xmm2, %xmm7
    
    vmovups %xmm7, (%rdx)
    
    add $16, %rsi
    add $16, %rdx
    sub $4, %rdi
    jmp .leftover

.leftover_1:
    cmp $0, %rdi
    je .done
    
    # Scalar fallback for 1-3 remaining elements
    vmovss (%rsi), %xmm0
    vmaxss clamp_lo(%rip), %xmm0, %xmm1
    vminss clamp_hi(%rip), %xmm1, %xmm1
    
    vxorps %xmm15, %xmm1, %xmm1
    vmulss %xmm8, %xmm1, %xmm2
    vroundss $0, %xmm2, %xmm2, %xmm3
    vsubss %xmm3, %xmm2, %xmm4
    
    vmovaps %xmm9, %xmm5
    vfmadd213ss %xmm10, %xmm4, %xmm5
    vfmadd213ss %xmm11, %xmm4, %xmm5
    vfmadd213ss %xmm12, %xmm4, %xmm5
    vfmadd213ss %xmm13, %xmm4, %xmm5
    
    vcvtss2si %xmm3, %eax
    add $127, %eax
    shl $23, %eax
    vmovd %eax, %xmm6
    
    vmulss %xmm6, %xmm5, %xmm5
    vaddss %xmm13, %xmm5, %xmm5
    vdivss %xmm5, %xmm0, %xmm7

    # L-08 scalar finite kill
    vmovd %xmm7, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 1f
    vxorps %xmm7, %xmm7, %xmm7
1:
    vmovss %xmm7, (%rdx)
    
    add $4, %rsi
    add $4, %rdx
    sub $1, %rdi
    jmp .leftover_1

.done:
    vzeroupper
    pop %rbp
    ret
